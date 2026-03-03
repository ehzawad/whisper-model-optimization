"""Streaming ASR engine with LocalAgreement-2 commit policy.

Architecture:
  ASRSession  — per-client state (buffer, VAD, hallucination filter, local agreement)
  BatchScheduler — singleton owning the shared model, batched GPU inference

The model is NOT loaded here. It is passed in from outside (e.g. a WhisperModel
created in transcribe.py or server startup).

Key optimizations:
  - fp16 + SDPA attention
  - Lightweight preprocessing (HPF + AGC only)
  - Inference deduplication (skip if audio unchanged)
  - Pre-allocated circular buffer per session
  - Batched inference across concurrent sessions
  - VRAM-aware batch cap
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*default values.*")
warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*")
warnings.filterwarnings("ignore", message=".*custom logits processor.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Shared HPF coefficients (computed once)
# ---------------------------------------------------------------------------

_HPF_SOS = None


def _get_hpf_sos(cutoff_hz: float) -> np.ndarray:
    global _HPF_SOS
    if _HPF_SOS is None:
        _HPF_SOS = butter(5, cutoff_hz, btype="high", fs=SAMPLE_RATE, output="sos")
    return _HPF_SOS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    beam_size: int = 1

    # Buffer management
    buffer_trim_threshold_sec: float = 25.0
    min_process_chunk_sec: float = 0.8
    max_prompt_words: int = 100
    buffer_keep_sec: float = 10.0

    # Warmup
    warmup_inferences: int = 5

    # LocalAgreement
    agreement_n: int = 2

    # VAD
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 30

    # Hallucination filter
    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0

    # Lightweight preprocessing (HPF + AGC only)
    preprocessing_enabled: bool = True
    highpass_cutoff_hz: float = 80.0
    agc_target_rms: float = 0.05
    agc_max_gain: float = 10.0

    # Batch scheduling
    collection_window_ms: int = 50
    max_batch_size: int = 0  # 0 = auto from GPU config


# ---------------------------------------------------------------------------
# Batch request / response
# ---------------------------------------------------------------------------

@dataclass
class _BatchRequest:
    """Internal: a single inference request queued for batch processing."""
    session_id: str
    audio: np.ndarray
    prompt: str
    future: Future


# ===================================================================
# BatchScheduler — singleton that owns the shared model
# ===================================================================

class BatchScheduler:
    """Owns the shared Whisper model and serialises GPU access.

    Collects inference requests from multiple ASRSession instances, batches
    them into a single ``model.generate()`` call when possible, and
    dispatches results back through Futures.

    Parameters
    ----------
    model : object
        Any object exposing ``.processor``, ``.model``, ``.gpu_config``
        (a dict with at least ``"vram_gb"``), and ``._cache_impl``
        (str or None). Typically a WhisperModel from the merged
        ``transcribe.py``, but the interface is deliberately loose.
    config : ASRConfig, optional
        Shared configuration. Only ``collection_window_ms``,
        ``max_batch_size``, ``beam_size``, and ``language`` are read here.
    """

    def __init__(self, model: Any, config: ASRConfig | None = None):
        self._model_wrapper = model
        self.config = config or ASRConfig()

        # Resolve convenience attributes from the wrapper
        self._processor = model.processor
        self._model = model.model
        self._cache_impl: str | None = getattr(model, "_cache_impl", None)
        self._gpu_config: dict = getattr(model, "gpu_config", {})

        # Collection window
        self._collection_window_s = self.config.collection_window_ms / 1000.0

        # Max batch size (0 = auto-detect from VRAM)
        self._max_batch: int = self.config.max_batch_size
        if self._max_batch <= 0:
            self._max_batch = self._auto_batch_limit()

        # Pending requests: session_id -> _BatchRequest (latest overwrites)
        self._pending: dict[str, _BatchRequest] = {}
        self._pending_lock = threading.Lock()
        self._collection_event = threading.Event()

        # Daemon thread
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        logger.info(
            "BatchScheduler created: collection_window=%dms, max_batch=%d",
            self.config.collection_window_ms,
            self._max_batch,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the batch processing daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("BatchScheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._batch_loop, name="batch-scheduler", daemon=True,
        )
        self._thread.start()
        logger.info("BatchScheduler started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the batch thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._collection_event.set()  # wake the thread if it is waiting
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("BatchScheduler thread did not exit within %.1fs", timeout)
            else:
                logger.info("BatchScheduler stopped")
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self, session_id: str, audio: np.ndarray, prompt: str = "",
    ) -> Future:
        """Queue an inference request and return a Future for the result.

        If a request for the same ``session_id`` is already pending it is
        overwritten (buffers are cumulative, so the latest snapshot is the
        most complete).

        The Future resolves to a ``str`` (decoded text) or raises on error.
        """
        fut: Future = Future()
        req = _BatchRequest(
            session_id=session_id, audio=audio, prompt=prompt, future=fut,
        )
        with self._pending_lock:
            old = self._pending.get(session_id)
            if old is not None and not old.future.done():
                # Cancel the stale future so the caller does not hang on it
                old.future.cancel()
            self._pending[session_id] = req
        # Signal the batch thread that work is available
        self._collection_event.set()
        return fut

    # ------------------------------------------------------------------
    # Auto batch limit from GPU config
    # ------------------------------------------------------------------

    def _auto_batch_limit(self) -> int:
        """Heuristic: ~1.5 GB VRAM per concurrent Whisper-medium decode."""
        vram_gb = self._gpu_config.get("vram_gb", 4)
        # Reserve 2 GB for the model weights + overhead
        available = max(1, vram_gb - 2)
        per_item_gb = 1.5
        limit = max(1, int(available / per_item_gb))
        logger.info(
            "Auto batch limit: vram=%.1f GB, available=%.1f GB -> max_batch=%d",
            vram_gb, available, limit,
        )
        return limit

    # ------------------------------------------------------------------
    # Batch loop (runs on daemon thread)
    # ------------------------------------------------------------------

    def _batch_loop(self) -> None:
        logger.info("Batch loop started")
        while not self._stop_event.is_set():
            # Wait for work or timeout
            self._collection_event.wait(timeout=self._collection_window_s)
            self._collection_event.clear()

            if self._stop_event.is_set():
                break

            # Snapshot and clear pending
            with self._pending_lock:
                if not self._pending:
                    continue
                batch = dict(self._pending)
                self._pending.clear()

            # Enforce batch cap: excess goes back to pending for next cycle
            items = list(batch.values())
            if len(items) > self._max_batch:
                overflow = items[self._max_batch:]
                items = items[:self._max_batch]
                with self._pending_lock:
                    for req in overflow:
                        # Only re-queue if no newer request arrived
                        if req.session_id not in self._pending:
                            self._pending[req.session_id] = req
                logger.debug(
                    "Batch cap reached: processing %d, deferred %d",
                    len(items), len(overflow),
                )

            try:
                self._process_batch(items)
            except Exception:
                logger.exception("Unhandled error in batch processing")
                for req in items:
                    if not req.future.done():
                        req.future.set_exception(
                            RuntimeError("Batch processing failed")
                        )

        # Drain remaining on shutdown
        with self._pending_lock:
            for req in self._pending.values():
                if not req.future.done():
                    req.future.cancel()
            self._pending.clear()
        logger.info("Batch loop exited")

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def _process_batch(self, items: list[_BatchRequest]) -> None:
        """Run inference for a batch of requests.

        If all prompts are identical (or empty) we can batch the generate
        call. When prompts diverge we fall back to sequential calls within
        this thread (still serialises GPU access so there is no contention).
        """
        import torch

        prompts = [r.prompt for r in items]
        can_batch = len(set(prompts)) <= 1

        if can_batch and len(items) > 1:
            self._batched_generate(items)
        else:
            # Sequential fallback — heterogeneous prompts or single item
            for req in items:
                try:
                    text = self._single_generate(req.audio, req.prompt)
                    req.future.set_result(text)
                except Exception as exc:
                    logger.error(
                        "Inference failed for session %s: %s",
                        req.session_id, exc,
                    )
                    if not req.future.done():
                        req.future.set_exception(exc)

    def _batched_generate(self, items: list[_BatchRequest]) -> None:
        """True batched inference — all items share the same prompt."""
        import torch

        prompt = items[0].prompt
        t0 = time.perf_counter()

        # Feature extraction for each item, then stack
        feature_list = []
        for req in items:
            feats = self._processor(
                req.audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features  # (1, 80, T)
            feature_list.append(feats)

        # Pad to same length along time axis and concatenate
        max_time = max(f.shape[-1] for f in feature_list)
        padded = []
        for f in feature_list:
            if f.shape[-1] < max_time:
                pad_width = max_time - f.shape[-1]
                f = torch.nn.functional.pad(f, (0, pad_width))
            padded.append(f)
        batch_feats = torch.cat(padded, dim=0).to(
            self.config.device, dtype=torch.float16,
        )  # (B, 80, T)

        # Build generation kwargs
        max_target = self._model.config.max_target_positions
        reserved = 4
        gen_kwargs: dict[str, Any] = {
            "num_beams": self.config.beam_size,
        }
        if self._cache_impl:
            gen_kwargs["cache_implementation"] = self._cache_impl
        if prompt:
            prompt_ids = self._processor.get_prompt_ids(prompt, return_tensors="pt")
            gen_kwargs["prompt_ids"] = prompt_ids.to(self.config.device)
            prompt_len = prompt_ids.shape[-1]
        else:
            prompt_len = 0
        gen_kwargs["max_new_tokens"] = max(10, max_target - reserved - prompt_len)

        with torch.no_grad():
            result_ids = self._model.generate(batch_feats, **gen_kwargs)

        texts = self._processor.batch_decode(result_ids, skip_special_tokens=True)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Batched inference: %d items in %.2fs (%.1fs avg audio)",
            len(items), elapsed,
            sum(len(r.audio) for r in items) / len(items) / SAMPLE_RATE,
        )

        # Strip prompt echo and dispatch
        prompt_stripped = prompt.strip() if prompt else ""
        for req, text in zip(items, texts):
            text = text.strip()
            if prompt_stripped and text.startswith(prompt_stripped):
                text = text[len(prompt_stripped):].strip()
            if not req.future.done():
                req.future.set_result(text)

    def _single_generate(self, audio: np.ndarray, prompt: str) -> str:
        """Single-item inference (used when prompts diverge)."""
        import torch

        feats = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        ).input_features.to(self.config.device, dtype=torch.float16)

        max_target = self._model.config.max_target_positions
        reserved = 4
        gen_kwargs: dict[str, Any] = {
            "num_beams": self.config.beam_size,
        }
        if self._cache_impl:
            gen_kwargs["cache_implementation"] = self._cache_impl
        if prompt:
            prompt_ids = self._processor.get_prompt_ids(prompt, return_tensors="pt")
            gen_kwargs["prompt_ids"] = prompt_ids.to(self.config.device)
            prompt_len = prompt_ids.shape[-1]
        else:
            prompt_len = 0
        gen_kwargs["max_new_tokens"] = max(10, max_target - reserved - prompt_len)

        t0 = time.perf_counter()
        with torch.no_grad():
            result_ids = self._model.generate(feats, **gen_kwargs)

        texts = self._processor.batch_decode(result_ids, skip_special_tokens=True)
        text = texts[0].strip() if texts else ""
        elapsed = time.perf_counter() - t0

        if prompt and text.startswith(prompt.strip()):
            text = text[len(prompt.strip()):].strip()

        logger.debug(
            "Single inference: %.2fs for %.1fs audio -> %d chars",
            elapsed, len(audio) / SAMPLE_RATE, len(text),
        )
        return text


# ===================================================================
# ASRSession — per-client session state
# ===================================================================

class ASRSession:
    """Per-client streaming ASR session.

    Owns audio buffer, VAD, hallucination filter, local-agreement state,
    and committed segments. Delegates all GPU work to a shared
    :class:`BatchScheduler`.

    Parameters
    ----------
    config : ASRConfig
        Per-session configuration.
    scheduler : BatchScheduler
        Shared batch scheduler that owns the model.
    session_id : str, optional
        Unique identifier. Auto-generated if not supplied.
    """

    def __init__(
        self,
        config: ASRConfig,
        scheduler: BatchScheduler,
        session_id: str | None = None,
    ):
        self.config = config
        self.scheduler = scheduler
        self.session_id = session_id or uuid.uuid4().hex[:12]

        # VAD (lazy import to avoid import-time side effects)
        from .vad import SileroVADWrapper
        self.vad = SileroVADWrapper(
            threshold=self.config.vad_threshold,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            speech_pad_ms=self.config.vad_speech_pad_ms,
        )

        # Hallucination filter
        from .hallucination import HallucinationFilter
        self.hallucination_filter = HallucinationFilter(
            no_speech_prob_threshold=self.config.no_speech_prob_threshold,
            avg_logprob_threshold=self.config.avg_logprob_threshold,
        )

        # Per-session inference future
        self._pending_future: Future | None = None

        # Session state — initialised by _reset()
        self._buffer_capacity: int = 0
        self._buffer_storage: np.ndarray = np.empty(0, dtype=np.float32)
        self._buffer_len: int = 0
        self.buffer_offset_samples: int = 0
        self.committed_segments: list[CommittedSegment] = []
        self.prev_hypothesis: list[str] = []
        self._samples_received: int = 0
        self._last_inference_samples: int = 0
        self._last_audio_hash: int = 0

        self._reset()

        logger.info("ASRSession created: id=%s", self.session_id)

    # ------------------------------------------------------------------
    # Buffer property (backed by pre-allocated storage)
    # ------------------------------------------------------------------

    @property
    def audio_buffer(self) -> np.ndarray:
        return self._buffer_storage[:self._buffer_len]

    @audio_buffer.setter
    def audio_buffer(self, value: np.ndarray) -> None:
        n = len(value)
        if n > self._buffer_capacity:
            self._buffer_capacity = n * 2
            self._buffer_storage = np.zeros(self._buffer_capacity, dtype=np.float32)
        self._buffer_storage[:n] = value
        self._buffer_len = n

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset all mutable session state."""
        self._buffer_capacity = int(
            self.config.buffer_trim_threshold_sec * SAMPLE_RATE * 2
        )
        self._buffer_storage = np.zeros(self._buffer_capacity, dtype=np.float32)
        self._buffer_len = 0
        self.buffer_offset_samples = 0
        self.committed_segments = []
        self.prev_hypothesis = []
        self._samples_received = 0
        self._last_inference_samples = 0
        self._last_audio_hash = 0
        self._pending_future = None

    def start_session(self) -> None:
        """Prepare the session for a new audio stream."""
        self._reset()
        self.vad.reset()
        logger.info("ASRSession %s started", self.session_id)

    def end_session(self) -> list[dict]:
        """Flush remaining audio and return any final segments."""
        results = self._force_commit_remaining()
        logger.info(
            "ASRSession %s ended. Total committed: %d segments",
            self.session_id, len(self.committed_segments),
        )
        return results

    # ------------------------------------------------------------------
    # Audio feed — main entry point
    # ------------------------------------------------------------------

    def feed_audio(self, pcm_float32: np.ndarray) -> list[dict]:
        """Append audio, run VAD, and return any ready results.

        Returns a list of result dicts (``{"type": "final", ...}`` or
        ``{"type": "partial", ...}``).
        """
        # Feed VAD
        self.vad.feed(pcm_float32)

        # Append to circular buffer
        n = len(pcm_float32)
        new_len = self._buffer_len + n
        if new_len > self._buffer_capacity:
            self._buffer_capacity = new_len * 2
            new_storage = np.zeros(self._buffer_capacity, dtype=np.float32)
            new_storage[:self._buffer_len] = self._buffer_storage[:self._buffer_len]
            self._buffer_storage = new_storage
        self._buffer_storage[self._buffer_len:new_len] = pcm_float32
        self._buffer_len = new_len
        self._samples_received += n

        # Collect results from any previously completed inference
        results = self._collect_completed_results()

        buffer_duration = self._buffer_len / SAMPLE_RATE

        # Not enough audio yet
        if buffer_duration < self.config.min_process_chunk_sec:
            return results

        # No speech and nothing in flight — just manage buffer
        if not self.vad.is_speaking and not self.prev_hypothesis:
            if buffer_duration > self.config.buffer_trim_threshold_sec:
                self._trim_buffer()
            return results

        # An inference is already in flight — wait for it
        if self._pending_future is not None and not self._pending_future.done():
            return results

        # Dedup: skip if not enough new audio since last inference
        new_samples = self._buffer_len - self._last_inference_samples
        if new_samples < int(0.5 * SAMPLE_RATE):
            return results

        # Preprocess and submit to scheduler
        self._last_inference_samples = self._buffer_len
        buffer_snapshot = self.audio_buffer.copy()

        try:
            buffer_snapshot = self._preprocess_audio(buffer_snapshot)
        except Exception as e:
            logger.warning(
                "Session %s: preprocessing failed, using raw audio: %s",
                self.session_id, e,
            )

        # Dedup: hash-based skip if audio unchanged
        audio_hash = hash(
            (len(buffer_snapshot),
             buffer_snapshot[:8].tobytes(),
             buffer_snapshot[-8:].tobytes())
        )
        if audio_hash == self._last_audio_hash and self._last_audio_hash != 0:
            return results
        self._last_audio_hash = audio_hash

        prompt = self._build_prompt()
        self._pending_future = self.scheduler.submit(
            session_id=self.session_id,
            audio=buffer_snapshot,
            prompt=prompt,
        )

        return results

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_completed_results(self) -> list[dict]:
        """Check whether the pending future has completed and process it."""
        if self._pending_future is None or not self._pending_future.done():
            return []

        fut = self._pending_future
        self._pending_future = None

        if fut.cancelled():
            return []

        try:
            raw_text: str = fut.result(timeout=0)
        except Exception as e:
            logger.error(
                "Session %s: inference future raised: %s", self.session_id, e,
            )
            return []

        return self._process_hypothesis(raw_text)

    # ------------------------------------------------------------------
    # Hypothesis processing (pure text, no model calls)
    # ------------------------------------------------------------------

    def _process_hypothesis(self, raw_text: str) -> list[dict]:
        """Apply hallucination filter, local agreement, and commit logic."""
        results: list[dict] = []

        if not raw_text:
            self.prev_hypothesis = []
            return results

        score = self.hallucination_filter.check_segment(raw_text, 0.0, 0.0)
        if not score.accepted:
            logger.debug(
                "Session %s: filtered '%s' (%s)",
                self.session_id, raw_text, score.reason,
            )
            self.prev_hypothesis = []
            return results

        current_words = raw_text.split()
        committed, partial = self._local_agreement(current_words)

        buffer_start = self.buffer_offset_samples / SAMPLE_RATE
        duration = self._buffer_len / SAMPLE_RATE

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

            # Trim buffer proportionally to committed words
            committed_frac = len(committed) / max(1, len(current_words))
            trim_samples = int(committed_frac * self._buffer_len)
            if trim_samples > 0 and self._buffer_len > trim_samples:
                self.audio_buffer = self.audio_buffer[trim_samples:]
                self.buffer_offset_samples += trim_samples
                self._last_inference_samples = max(
                    0, self._last_inference_samples - trim_samples,
                )
                self._last_audio_hash = 0

        if partial:
            results.append({"type": "partial", "text": " ".join(partial)})

        self.prev_hypothesis = current_words

        if self._buffer_len / SAMPLE_RATE > self.config.buffer_trim_threshold_sec:
            self._trim_buffer()

        return results

    # ------------------------------------------------------------------
    # Lightweight preprocessing (HPF + AGC)
    # ------------------------------------------------------------------

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter and automatic gain control."""
        if not self.config.preprocessing_enabled:
            return audio

        t0 = time.perf_counter()
        rms = np.sqrt(np.mean(audio ** 2))

        # Skip if already in target range
        target = self.config.agc_target_rms
        if rms > 1e-8 and 0.5 * target < rms < 2.0 * target:
            elapsed = time.perf_counter() - t0
            logger.debug(
                "Session %s preprocessing: skipped (RMS=%.4f in range) %.3fs",
                self.session_id, rms, elapsed,
            )
            return audio

        sos = _get_hpf_sos(self.config.highpass_cutoff_hz)
        audio = sosfilt(sos, audio).astype(np.float32)

        if rms > 1e-8:
            gain = min(self.config.agc_max_gain, target / rms)
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

        elapsed = time.perf_counter() - t0
        logger.debug(
            "Session %s preprocessing: %.3fs (HPF+AGC) for %.1fs audio",
            self.session_id, elapsed, len(audio) / SAMPLE_RATE,
        )
        return audio

    # ------------------------------------------------------------------
    # LocalAgreement
    # ------------------------------------------------------------------

    def _local_agreement(
        self, current_words: list[str],
    ) -> tuple[list[str], list[str]]:
        """LocalAgreement-N commit policy.

        Returns (committed_words, partial_words).
        """
        if not self.prev_hypothesis or not current_words:
            return [], current_words

        common_len = 0
        for i in range(min(len(self.prev_hypothesis), len(current_words))):
            if self.prev_hypothesis[i] == current_words[i]:
                common_len = i + 1
            else:
                break

        return current_words[:common_len], current_words[common_len:]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """Build a conditioning prompt from recently committed segments."""
        all_text = " ".join(seg.text for seg in self.committed_segments)
        words = all_text.split()
        if len(words) > self.config.max_prompt_words:
            words = words[-self.config.max_prompt_words:]
        return " ".join(words)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _trim_buffer(self) -> None:
        """Trim audio buffer to keep only the trailing window."""
        keep_samples = int(self.config.buffer_keep_sec * SAMPLE_RATE)
        if self._buffer_len > keep_samples:
            trim_samples = self._buffer_len - keep_samples
            self.audio_buffer = self.audio_buffer[trim_samples:]
            self.buffer_offset_samples += trim_samples
            self.prev_hypothesis = []
            self._last_inference_samples = max(
                0, self._last_inference_samples - trim_samples,
            )
            self._last_audio_hash = 0
            logger.debug(
                "Session %s: buffer trimmed to %.1fs",
                self.session_id, self.config.buffer_keep_sec,
            )

    def _force_commit_remaining(self) -> list[dict]:
        """Synchronously flush any remaining audio at session end."""
        if self._buffer_len < 1600:
            return []

        # Wait for any in-flight inference to finish
        if self._pending_future is not None and not self._pending_future.done():
            try:
                self._pending_future.result(timeout=5.0)
            except Exception:
                pass
            # Collect those results first
            results = self._collect_completed_results()
        else:
            results = self._collect_completed_results()

        # Final forced inference
        try:
            audio = self._preprocess_audio(self.audio_buffer.copy())
        except Exception as e:
            logger.warning(
                "Session %s: preprocessing failed at end: %s", self.session_id, e,
            )
            audio = self.audio_buffer.copy()

        prompt = self._build_prompt()
        fut = self.scheduler.submit(
            session_id=self.session_id, audio=audio, prompt=prompt,
        )

        try:
            text = fut.result(timeout=30.0)
        except Exception as e:
            logger.error(
                "Session %s: final inference failed: %s", self.session_id, e,
            )
            return results

        if not text:
            return results

        score = self.hallucination_filter.check_segment(text, 0.0, 0.0)
        if not score.accepted:
            return results

        buffer_start = self.buffer_offset_samples / SAMPLE_RATE
        duration = self._buffer_len / SAMPLE_RATE
        seg = CommittedSegment(
            text=text,
            start=round(buffer_start, 2),
            end=round(buffer_start + duration, 2),
        )
        self.committed_segments.append(seg)
        results.append({
            "type": "final",
            "text": text,
            "start": seg.start,
            "end": seg.end,
        })
        return results
