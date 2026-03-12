"""Live Bengali captioning server using Whisper Medium (faster-whisper / CTranslate2).

Adapted from ct-live-cc/bangla_lcc.py (Wav2Vec2 CTC) for the fine-tuned Whisper Medium
model. Uses faster-whisper for inference (~2x faster than HF, ~800MB VRAM).

Key differences from bangla_lcc.py:
  - Whisper encoder-decoder instead of Wav2Vec2 CTC
  - Hallucination filtering (Whisper-specific)
  - Silero VAD instead of WebRTC VAD
  - HPF + AGC preprocessing (instead of / alongside DeepFilterNet)

Usage:
    python live_caption.py
"""

import os
import json
import asyncio
import time
import torch
import shutil
import numpy as np
import soundfile as sf
import websockets
from loguru import logger
from scipy.signal import resample, butter, sosfilt
from faster_whisper import WhisperModel
from bnunicodenormalizer import Normalizer

from streaming.hallucination import HallucinationFilter

# ======================================================================
# CONSTANTS
# ======================================================================

RATE = 16000                        # Internal processing sample rate
INPUT_SAMPLE_RATE = 48000           # Incoming audio sample rate from client

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct2_model_fp16")
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
PORT = 2700

SAVE_FOLDER = "INCOMING_AUDIO_DATA_FOLDER"
SAVE_INCOMING_AUDIO = False         # Set True to save incoming audio to disk
ACTIVATE_DENOISER = False           # Set True for DeepFilterNet (uses extra VRAM)

LESS_THAN_ONE_SEC = "LESS_THAN_ONE_SEC"
SILENCE_IN_INCOMING_AUDIO = "SILENCE_IN_INCOMING_AUDIO"

INCOMING_AUDIO_DURATION = 0.5       # Expected duration of each incoming chunk (seconds)
MINIMUM_INTERVAL_FOR_INFERENCE = 0.5  # Min new audio before re-transcribing (seconds)
MAXIMUM_BUFFER_TIME = 5             # Max buffer before forced reset (seconds)
MAXIMUM_SILENCE_DURATION = 0.75     # Silence in latest chunk triggers reset (seconds)
MAX_CONSECUTIVE_SILENCE_FRAME_NO = 2  # Consecutive silent chunks before reset

# HPF + AGC parameters
HPF_CUTOFF_HZ = 80.0
AGC_TARGET_RMS = 0.05
AGC_MAX_GAIN = 10.0

# ======================================================================
# INITIALIZATION
# ======================================================================

# cuDNN is completely broken on this system (RTX 2050, cuDNN 9.11)
torch.backends.cudnn.enabled = False

# Optional DeepFilterNet denoiser
if ACTIVATE_DENOISER:
    from df.enhance import enhance, init_df
    denoiser_model, df_state, _ = init_df()

# Save folder management
if os.path.exists(SAVE_FOLDER):
    shutil.rmtree(SAVE_FOLDER)
if SAVE_INCOMING_AUDIO and not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Load faster-whisper model
logger.info(f"Loading Whisper model from {MODEL_PATH} ({DEVICE}/{COMPUTE_TYPE})")
t_load = time.perf_counter()
whisper_model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
logger.info(f"Model loaded in {time.perf_counter() - t_load:.2f}s")

# Hallucination filter (Whisper-specific: no_speech_prob, logprob, phrases, repetition)
halluc_filter = HallucinationFilter()

# Silero VAD for chunk-level speech detection
from silero_vad import load_silero_vad, get_speech_timestamps
vad_model = load_silero_vad()
logger.info("Silero VAD loaded")

# HPF filter coefficients (5th-order Butterworth, 80Hz cutoff)
_hpf_sos = butter(5, HPF_CUTOFF_HZ, btype="high", fs=RATE, output="sos")

# Bengali Unicode normalizer (instantiate once)
_bnorm = Normalizer()

# ======================================================================
# PROCESSING FUNCTIONS
# ======================================================================


def save_audio_data(audio_data):
    """Save audio buffer to disk as WAV."""
    file_name = os.path.join(SAVE_FOLDER, f"{int(time.time())}.wav")
    sf.write(file_name, audio_data, RATE)
    logger.info(f"Saved audio: {file_name}")


def calculate_dbfs(audio_data):
    """Calculate dBFS of audio signal."""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def denoise_audio(audio, sample_rate):
    """Denoise using DeepFilterNet (only when ACTIVATE_DENOISER=True)."""
    if sample_rate != df_state.sr():
        raise ValueError(
            f"Sample rate {sample_rate} != DeepFilterNet expected {df_state.sr()}"
        )
    audio_tensor = torch.from_numpy(audio).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    enhanced = enhance(denoiser_model, df_state, audio_tensor)
    return enhanced.squeeze(0).numpy()


def preprocess_audio(audio):
    """Apply high-pass filter (80Hz) + automatic gain control.

    Adapted from streaming/asr_engine.py ASRSession._preprocess_audio.
    """
    rms = np.sqrt(np.mean(audio ** 2))

    # Skip if already in target range
    if rms > 1e-8 and 0.5 * AGC_TARGET_RMS < rms < 2.0 * AGC_TARGET_RMS:
        return audio

    # High-pass filter
    audio = sosfilt(_hpf_sos, audio).astype(np.float32)

    # Automatic gain control
    if rms > 1e-8:
        gain = min(AGC_MAX_GAIN, AGC_TARGET_RMS / rms)
        audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    return audio


def has_speech(audio_chunk, threshold=0.5):
    """Check if audio chunk contains speech using Silero VAD.

    Replaces bangla_lcc.py's WebRTC VAD-based is_silent() with Silero
    (better quality, already installed, stateless per call).
    """
    if len(audio_chunk) < 512:  # Minimum window size for Silero
        return False
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio_chunk).float(),
        vad_model,
        sampling_rate=RATE,
        threshold=threshold,
    )
    return len(timestamps) > 0


def normalize_sentence(sentence):
    """Bengali Unicode normalization via bnunicodenormalizer."""
    words = sentence.split()
    normalized = []
    for word in words:
        result = _bnorm(word).get("normalized")
        if result is not None:
            normalized.append(result)
    return " ".join(normalized)


# ======================================================================
# TRANSCRIPTION FUNCTION
# ======================================================================


def transcribe_audio_data(data, audio_data):
    """Process incoming audio and return transcription.

    Mirrors bangla_lcc.py's transcribe_audio_data() but uses faster-whisper
    instead of Wav2Vec2 CTC. Key adaptation: model.transcribe() returns a
    generator of segments (with confidence scores) instead of a single
    CTC-decoded string. Each segment is checked for hallucination.

    Returns:
        dict: {"partial": text} or {"text": text} for WebSocket output
        str: SILENCE_IN_INCOMING_AUDIO or LESS_THAN_ONE_SEC sentinel
        None: if transcription produced no valid text
    """
    # Convert incoming int16 PCM to float32 [-1, 1]
    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    logger.info(f"DBFS BEFORE PREPROCESS: {calculate_dbfs(audio_chunk):.1f}")

    # Optional DeepFilterNet denoising (at input sample rate, before resample)
    if ACTIVATE_DENOISER:
        audio_chunk = denoise_audio(audio_chunk, sample_rate=INPUT_SAMPLE_RATE)

    logger.info(f"DBFS AFTER DENOISE: {calculate_dbfs(audio_chunk):.1f}")

    # Resample to 16kHz if needed
    if INPUT_SAMPLE_RATE != RATE:
        n_samples = round(len(audio_chunk) * RATE / INPUT_SAMPLE_RATE)
        resampled = resample(audio_chunk, n_samples)
    else:
        resampled = audio_chunk

    # VAD: skip silent chunks (replaces bangla_lcc.py's is_silent check)
    if not has_speech(resampled):
        return SILENCE_IN_INCOMING_AUDIO

    # Accumulate non-silent audio in buffer
    audio_data.add_data(resampled)

    # Check if enough new data has accumulated for inference
    new_samples = len(audio_data.get_data()) - audio_data.get_last_inference_position()
    if new_samples < (MINIMUM_INTERVAL_FOR_INFERENCE * RATE):
        return LESS_THAN_ONE_SEC

    # Preprocess full buffer (HPF + AGC)
    processed = preprocess_audio(audio_data.get_data())

    logger.info(f"DBFS OF BUFFER: {calculate_dbfs(processed):.1f}")

    # ── Transcribe with faster-whisper ──
    # NOTE: Use model.transcribe() directly, NOT BatchedInferencePipeline.
    # BatchedInferencePipeline is for long audio files. For short buffers (≤5s),
    # direct transcription is correct and avoids overhead.
    t_infer = time.perf_counter()
    segments, info = whisper_model.transcribe(
        processed,
        language="bn",
        beam_size=1,
        vad_filter=False,       # We handle VAD ourselves
        without_timestamps=True,
    )

    # Consume generator immediately and filter hallucinations per segment
    text_parts = []
    for seg in segments:
        score = halluc_filter.check_segment(
            seg.text, seg.no_speech_prob, seg.avg_logprob
        )
        if score.accepted:
            text_parts.append(score.text)
        else:
            logger.debug(f"Hallucination filtered: '{seg.text}' ({score.reason})")

    transcription = " ".join(text_parts).strip()
    elapsed = time.perf_counter() - t_infer
    logger.info(
        f"Inference: {elapsed:.3f}s for {len(processed)/RATE:.1f}s audio"
    )

    # Update inference position to avoid reprocessing same data
    audio_data.update_inference_position()

    # Normalize Bengali text
    if transcription:
        transcription = normalize_sentence(transcription)

    # ── Buffer management: reset on max time or silence ──
    if len(audio_data.get_data()) > (MAXIMUM_BUFFER_TIME * RATE) or not has_speech(
        resampled, threshold=0.3
    ):
        if len(audio_data.get_data()) > (MAXIMUM_BUFFER_TIME * RATE):
            logger.info(f"{MAXIMUM_BUFFER_TIME}s BUFFER FULL.")
        else:
            logger.info("Silence detected in latest chunk, resetting buffer.")

        audio_data.reset_data()
        return {"text": transcription} if transcription else None

    return {"partial": transcription} if transcription else None


# ======================================================================
# AUDIO DATA CLASS
# ======================================================================


class AudioData:
    """Per-connection audio buffer with inference position tracking.

    Direct port from bangla_lcc.py's AudioData class.
    """

    def __init__(self):
        self.data = np.array([], dtype=np.float32)
        self.last_inference_position = 0

    def reset_data(self):
        if SAVE_INCOMING_AUDIO:
            save_audio_data(self.data)
        self.data = np.array([], dtype=np.float32)
        self.last_inference_position = 0

    def get_data(self):
        return self.data

    def set_data(self, input_data):
        self.data = input_data

    def add_data(self, new_data):
        self.data = np.concatenate((self.data, new_data))

    def get_last_inference_position(self):
        return self.last_inference_position

    def update_inference_position(self):
        self.last_inference_position = len(self.data)


# ======================================================================
# WEBSOCKET HANDLER
# ======================================================================


async def recognize(websocket):
    """Handle a single WebSocket connection for live captioning.

    Mirrors bangla_lcc.py's recognize() with additions:
      - "END" text message support for graceful session close
      - Hallucination-aware transcription
      - ensure_ascii=False for proper Bengali output
    """
    addr = getattr(websocket, "remote_address", "unknown")
    logger.info(f"Connection from {addr}")

    audio = AudioData()
    consecutive_silence_count = 0
    last_transcription = None

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                transcription = transcribe_audio_data(message, audio)

                if transcription is SILENCE_IN_INCOMING_AUDIO:
                    consecutive_silence_count += 1
                    if consecutive_silence_count >= MAX_CONSECUTIVE_SILENCE_FRAME_NO:
                        logger.info(
                            f"Consecutive silence ({consecutive_silence_count} frames)"
                        )
                        audio.reset_data()
                        consecutive_silence_count = 0
                        # Promote last partial to final
                        if (
                            last_transcription
                            and "partial" in last_transcription
                        ):
                            transcription = {
                                "text": last_transcription["partial"]
                            }
                        else:
                            continue
                    else:
                        continue
                else:
                    consecutive_silence_count = 0

                if (
                    transcription is not SILENCE_IN_INCOMING_AUDIO
                    and transcription is not LESS_THAN_ONE_SEC
                    and transcription is not None
                ):
                    last_transcription = transcription
                    await websocket.send(
                        json.dumps(transcription, ensure_ascii=False)
                    )

            elif isinstance(message, str):
                if message.strip().upper() == "END":
                    logger.info("Client sent END signal")
                    # Promote remaining partial to final
                    if last_transcription and "partial" in last_transcription:
                        await websocket.send(
                            json.dumps(
                                {"text": last_transcription["partial"]},
                                ensure_ascii=False,
                            )
                        )
                    break

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed from {addr}")
    finally:
        audio.reset_data()


# ======================================================================
# SERVER
# ======================================================================


async def start():
    logger.add("live_transcription_log/{time:YYYY-MM-DD}.log")
    logger.info(
        f"Bengali Whisper live caption server starting on port {PORT}"
    )

    async with websockets.serve(recognize, "0.0.0.0", PORT):
        logger.info(f"Listening on ws://0.0.0.0:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(start())
