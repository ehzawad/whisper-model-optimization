"""Streaming ASR engine with LocalAgreement-2 commit policy.

Uses HuggingFace transformers Whisper model directly (the fine-tuned
Bengali model's CTranslate2 conversion is incompatible). Implements
the commit policy, manages the growing audio buffer, handles prompt
conditioning, and coordinates VAD + hallucination filtering.

Inference runs in a background thread so audio keeps accumulating
without blocking.
"""

import asyncio
import logging
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch

# Suppress noisy HF deprecation warnings
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*default values.*")
warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*")
warnings.filterwarnings("ignore", message=".*custom logits processor.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class CommittedSegment:
    """A finalized piece of transcription."""

    text: str
    start: float
    end: float


@dataclass
class ASRConfig:
    """Configuration for the streaming ASR engine."""

    model_dir: str = "."
    device: str = "cuda"
    language: str = "bn"
    beam_size: int = 1  # Greedy decoding — 3x faster than beam=5, same quality

    # Buffer management
    buffer_trim_threshold_sec: float = 25.0
    min_process_chunk_sec: float = 1.5  # Accumulate at least 1.5s before inference
    max_prompt_words: int = 100  # Shorter prompt = fewer tokens = faster decode
    buffer_keep_sec: float = 10.0

    # LocalAgreement
    agreement_n: int = 2

    # VAD
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 30

    # Hallucination filter
    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0


class StreamingASREngine:
    """Real-time streaming ASR with LocalAgreement-2 commit policy.

    Audio flows in via feed_audio(). Inference runs asynchronously —
    results are collected via get_results().
    """

    def __init__(self, config: ASRConfig | None = None):
        self.config = config or ASRConfig()

        torch.backends.cudnn.enabled = False

        logger.info("Loading Whisper model from %s", self.config.model_dir)
        self._load_model()
        logger.info("Model loaded (beam=%d)", self.config.beam_size)

        from .vad import SileroVADWrapper

        self.vad = SileroVADWrapper(
            threshold=self.config.vad_threshold,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
        )

        from .hallucination import HallucinationFilter

        self.hallucination_filter = HallucinationFilter(
            no_speech_prob_threshold=self.config.no_speech_prob_threshold,
            avg_logprob_threshold=self.config.avg_logprob_threshold,
        )

        self._lock = threading.Lock()
        self._inference_running = False
        self._pending_results: list[dict] = []
        self._reset_session_state()

    def _load_model(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(self.config.model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_dir,
            dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(self.config.device)
        self.model.eval()

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.language, task="transcribe"
        )
        self.model.generation_config.forced_decoder_ids = self.forced_decoder_ids

        gen_cfg = self.model.generation_config
        if not hasattr(gen_cfg, "no_timestamps_token_id") or gen_cfg.no_timestamps_token_id is None:
            gen_cfg.no_timestamps_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        # Warmup pass — first inference is always slow due to CUDA kernels
        logger.info("Running warmup inference...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s silence
        with torch.no_grad():
            feats = self.processor(dummy, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            feats = feats.input_features.to(self.config.device, dtype=torch.float16)
            self.model.generate(feats, forced_decoder_ids=self.forced_decoder_ids, max_new_tokens=5)
        torch.cuda.synchronize()
        logger.info("Warmup done")

    def _reset_session_state(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset_samples: int = 0
        self.committed_segments: list[CommittedSegment] = []
        self.prev_hypothesis: list[str] = []
        self._samples_received: int = 0
        self._last_inference_samples: int = 0  # buffer length at last inference

    def start_session(self):
        self._reset_session_state()
        self.vad.reset()
        with self._lock:
            self._pending_results.clear()
            self._inference_running = False
        logger.info("ASR session started")

    def end_session(self) -> list[dict]:
        results = self._force_commit_remaining()
        logger.info(
            "ASR session ended. Total committed: %d segments",
            len(self.committed_segments),
        )
        return results

    def feed_audio(self, pcm_float32: np.ndarray) -> list[dict]:
        """Feed audio. Returns any results ready from background inference."""
        self.vad.feed(pcm_float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm_float32])
        self._samples_received += len(pcm_float32)

        # Collect any results from background inference
        results = self._drain_results()

        buffer_duration = len(self.audio_buffer) / SAMPLE_RATE

        # Not enough audio
        if buffer_duration < self.config.min_process_chunk_sec:
            return results

        # No speech and no pending hypothesis
        if not self.vad.is_speaking and not self.prev_hypothesis:
            if buffer_duration > self.config.buffer_trim_threshold_sec:
                self._trim_buffer()
            return results

        # Don't start new inference if one is already running
        if self._inference_running:
            return results

        # Require at least 0.5s of NEW audio since last inference
        new_samples = len(self.audio_buffer) - self._last_inference_samples
        if new_samples < int(0.5 * SAMPLE_RATE):
            return results

        # Launch inference in background thread
        self._inference_running = True
        self._last_inference_samples = len(self.audio_buffer)
        buffer_snapshot = self.audio_buffer.copy()
        threading.Thread(
            target=self._inference_thread,
            args=(buffer_snapshot,),
            daemon=True,
        ).start()

        return results

    def _drain_results(self) -> list[dict]:
        with self._lock:
            results = self._pending_results.copy()
            self._pending_results.clear()
        return results

    def _inference_thread(self, audio_snapshot: np.ndarray):
        """Run inference in a background thread."""
        try:
            results = self._run_inference_on(audio_snapshot)
            with self._lock:
                self._pending_results.extend(results)
        except Exception as e:
            logger.error("Inference error: %s", e)
        finally:
            self._inference_running = False

    @torch.no_grad()
    def _transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        """Run Whisper inference. Returns transcribed text."""
        feats = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.config.device, dtype=torch.float16)

        max_target = self.model.config.max_target_positions
        reserved = 4

        gen_kwargs = {
            "forced_decoder_ids": self.forced_decoder_ids,
            "num_beams": self.config.beam_size,
        }

        if prompt:
            prompt_ids = self.processor.get_prompt_ids(prompt, return_tensors="pt")
            gen_kwargs["prompt_ids"] = prompt_ids.to(self.config.device)
            prompt_len = prompt_ids.shape[-1]
        else:
            prompt_len = 0

        gen_kwargs["max_new_tokens"] = max(10, max_target - reserved - prompt_len)

        result = self.model.generate(feats, **gen_kwargs)
        text = self.processor.batch_decode(result, skip_special_tokens=True)
        return text[0].strip() if text else ""

    def _run_inference_on(self, audio: np.ndarray) -> list[dict]:
        """Run Whisper + LocalAgreement-2 on an audio buffer snapshot."""
        results = []

        prompt = self._build_prompt()
        t0 = time.perf_counter()
        text = self._transcribe(audio, prompt or None)
        elapsed = time.perf_counter() - t0
        logger.info("Inference: %.2fs for %.1fs audio -> %d chars",
                     elapsed, len(audio) / SAMPLE_RATE, len(text))

        if not text:
            self.prev_hypothesis = []
            return results

        # Hallucination check (phrase + repetition only, since HF doesn't give logprob)
        score = self.hallucination_filter.check_segment(text, 0.0, 0.0)
        if not score.accepted:
            logger.debug("Filtered: '%s' (%s)", text, score.reason)
            self.prev_hypothesis = []
            return results

        current_words = text.split()
        committed, partial = self._local_agreement(current_words)

        buffer_start = self.buffer_offset_samples / SAMPLE_RATE
        duration = len(audio) / SAMPLE_RATE

        if committed:
            committed_text = " ".join(committed)
            seg = CommittedSegment(
                text=committed_text,
                start=round(buffer_start, 2),
                end=round(buffer_start + duration, 2),
            )
            self.committed_segments.append(seg)
            results.append({
                "type": "final",
                "text": committed_text,
                "start": seg.start,
                "end": seg.end,
            })

        if partial:
            results.append({"type": "partial", "text": " ".join(partial)})

        self.prev_hypothesis = current_words

        if len(self.audio_buffer) / SAMPLE_RATE > self.config.buffer_trim_threshold_sec:
            self._trim_buffer()

        return results

    def _local_agreement(
        self, current_words: list[str]
    ) -> tuple[list[str], list[str]]:
        if not self.prev_hypothesis or not current_words:
            return [], current_words

        common_len = 0
        for i in range(min(len(self.prev_hypothesis), len(current_words))):
            if self.prev_hypothesis[i] == current_words[i]:
                common_len = i + 1
            else:
                break

        return current_words[:common_len], current_words[common_len:]

    def _build_prompt(self) -> str:
        all_text = " ".join(seg.text for seg in self.committed_segments)
        words = all_text.split()
        if len(words) > self.config.max_prompt_words:
            words = words[-self.config.max_prompt_words :]
        return " ".join(words)

    def _trim_buffer(self):
        keep_samples = int(self.config.buffer_keep_sec * SAMPLE_RATE)
        if len(self.audio_buffer) > keep_samples:
            trim_samples = len(self.audio_buffer) - keep_samples
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.buffer_offset_samples += trim_samples
            self.prev_hypothesis = []
            self._last_inference_samples = max(0, self._last_inference_samples - trim_samples)
            logger.debug("Buffer trimmed to %.1fs", self.config.buffer_keep_sec)

    def _force_commit_remaining(self) -> list[dict]:
        if len(self.audio_buffer) < 1600:
            return []

        # Wait for any running inference to finish
        for _ in range(50):
            if not self._inference_running:
                break
            time.sleep(0.1)

        results = []
        prompt = self._build_prompt()
        text = self._transcribe(self.audio_buffer, prompt or None)

        if text:
            score = self.hallucination_filter.check_segment(text, 0.0, 0.0)
            if score.accepted:
                buffer_start = self.buffer_offset_samples / SAMPLE_RATE
                duration = len(self.audio_buffer) / SAMPLE_RATE
                seg = CommittedSegment(
                    text=score.text,
                    start=round(buffer_start, 2),
                    end=round(buffer_start + duration, 2),
                )
                self.committed_segments.append(seg)
                results.append({
                    "type": "final",
                    "text": score.text,
                    "start": seg.start,
                    "end": seg.end,
                })

        return results
