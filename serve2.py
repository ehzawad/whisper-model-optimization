#!/usr/bin/env python3
"""FastAPI batch ASR server — Bengali Whisper (faster-whisper / CTranslate2).

Optimized variant: parallel CPU decode, scipy resampling, no VAD, int8_float16.

Usage:
    python serve2.py
    python serve2.py --port 8002 --host 0.0.0.0
"""

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens
from pydantic import BaseModel
from scipy.signal import resample_poly
from typing import List

from loguru import logger

# logger configuration
logger.add(
    "log_folder/{time:YYYY-MM-DD}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = FastAPI()

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CT2_MODEL_DIR = os.path.join(MODEL_DIR, "ct2_model_fp16")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

# ── Model + pipeline (cached singletons) ─────────────────────────────────────
logger.info(f"Loading faster-whisper model from {CT2_MODEL_DIR} ({DEVICE}/{COMPUTE_TYPE})")
t_load = time.perf_counter()
_model = WhisperModel(
    CT2_MODEL_DIR,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=4,
)
_pipeline = BatchedInferencePipeline(model=_model)
logger.info(f"Model + pipeline ready in {time.perf_counter() - t_load:.2f}s")

# Thread pool for parallel base64 decode + resample (CPU work)
_decode_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


# ── Pydantic models (mirrors Java Wav2Vec2 ASR server) ───────────────────────
class Language(BaseModel):
    sourceLanguage: str

class Config(BaseModel):
    language: Language

class AudioContent(BaseModel):
    audioContent: str  # Base64 encoded audio

class AsrRequest(BaseModel):
    config: Config
    audio: List[AudioContent]

class Output(BaseModel):
    source: str

class AsrResponse(BaseModel):
    taskType: str
    output: List[Output]
    time_taken: float


# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio_from_base64(audio_content: str) -> np.ndarray:
    """Decode base64 audio to float32 numpy array at 16kHz."""
    raw = base64.b64decode(audio_content)
    buf = io.BytesIO(raw)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
    return audio


def transcribe_batch(audios: list[np.ndarray], batch_size: int = 16) -> list[str]:
    """Transcribe multiple audio arrays in batched GPU passes.

    Instead of calling _pipeline.transcribe() per audio (which only accepts
    a single input), we call _pipeline.forward() directly with stacked
    features from multiple audios. This is the same codepath the pipeline
    uses internally for VAD-segmented chunks.

    Audio clips longer than 30s are auto-chunked into ≤30s pieces, all
    chunks are batched together, and text is reassembled per original audio.
    """
    if not audios:
        return []

    CHUNK_SAMPLES = 30 * SAMPLE_RATE  # 480000 samples = 30s (Whisper encoder window)

    # ── 1. Split long audios into ≤30s chunks, extract mel features (CPU) ──
    features_list = []
    chunks_metadata = []
    chunk_to_audio = []  # Maps each chunk index → original audio index

    for audio_idx, audio in enumerate(audios):
        for start in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[start : start + CHUNK_SAMPLES]
            feat = _model.feature_extractor(chunk)[..., :-1]
            features_list.append(feat)
            duration = len(chunk) / SAMPLE_RATE
            chunks_metadata.append({
                "offset": 0.0,
                "duration": duration,
                "segments": [{"start": 0, "end": len(chunk)}],
            })
            chunk_to_audio.append(audio_idx)

    # ── 2. Pad/trim to [80, 3000] and stack into [N, 80, 3000] ──
    features = np.stack([pad_or_trim(f) for f in features_list])

    # ── 3. Build tokenizer + options (same pattern as transcribe.py:507-553) ──
    tokenizer = Tokenizer(
        _model.hf_tokenizer,
        _model.model.is_multilingual,
        task="transcribe",
        language="bn",
    )

    options = TranscriptionOptions(
        beam_size=1,
        best_of=1,
        patience=1,
        length_penalty=1,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        prompt_reset_on_temperature=0.5,
        temperatures=[0.0],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=get_suppressed_tokens(tokenizer, [-1]),
        without_timestamps=True,
        max_initial_timestamp=0.0,
        word_timestamps=False,
        prepend_punctuations="\"'\"¿([{-",
        append_punctuations="\"'.。,，!！?？:：\")]}、",
        multilingual=False,
        max_new_tokens=None,
        clip_timestamps="0",
        hallucination_silence_threshold=None,
        hotwords=None,
    )

    # ── 4. Batched GPU forward passes ──
    chunk_texts = []
    for i in range(0, len(features), batch_size):
        batch_features = features[i : i + batch_size]
        batch_metadata = chunks_metadata[i : i + batch_size]

        segmented_outputs = _pipeline.forward(
            batch_features, tokenizer, batch_metadata, options
        )

        for segments in segmented_outputs:
            text = " ".join(
                seg["text"].strip() for seg in segments if seg["text"].strip()
            )
            chunk_texts.append(text)

    # ── 5. Reassemble: concatenate chunk texts per original audio ──
    audio_texts = [""] * len(audios)
    for chunk_idx, audio_idx in enumerate(chunk_to_audio):
        if chunk_texts[chunk_idx]:
            if audio_texts[audio_idx]:
                audio_texts[audio_idx] += " " + chunk_texts[chunk_idx]
            else:
                audio_texts[audio_idx] = chunk_texts[chunk_idx]

    return audio_texts


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": "faster-whisper", "device": DEVICE,
            "compute_type": COMPUTE_TYPE}


@app.post("/asr", response_model=AsrResponse)
async def process_asr(request: AsrRequest):
    start_time = time.time()

    if not request.audio or len(request.audio) == 0:
        raise HTTPException(status_code=400, detail="No audio content provided")

    # Parallel CPU decode: base64 → resample → numpy arrays
    futures = [
        _decode_pool.submit(load_audio_from_base64, item.audioContent)
        for item in request.audio
    ]
    audios = []
    for i, fut in enumerate(futures):
        try:
            audios.append(fut.result())
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error decoding audio item {i}: {e}",
            )

    # Batched GPU transcribe — all audios processed in GPU batches of 16
    try:
        texts = transcribe_batch(audios, batch_size=16)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in batched transcription: {e}",
        )
    outputs = [Output(source=t) for t in texts]

    time_taken = time.time() - start_time

    response = AsrResponse(
        taskType="ASR",
        output=outputs,
        time_taken=time_taken,
    )

    logger.info(response)
    return response


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Bengali Batch ASR Server (optimized)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
