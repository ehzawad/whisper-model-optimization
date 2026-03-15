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


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio using BatchedInferencePipeline, VAD disabled."""
    segments, _info = _pipeline.transcribe(
        audio,
        language="bn",
        beam_size=1,
        vad_filter=False,
        batch_size=8,
        without_timestamps=True,
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


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

    # Sequential GPU transcribe (each call uses batch_size=8 internally)
    outputs = []
    for i, audio in enumerate(audios):
        try:
            transcription = transcribe(audio)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error transcribing audio item {i}: {e}",
            )
        outputs.append(Output(source=transcription))

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
