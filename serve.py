#!/usr/bin/env python3
"""FastAPI batch ASR server — Bengali Whisper (faster-whisper / CTranslate2).

Mirrors the Wav2Vec2 ASR server API contract: accepts base64-encoded audio
in JSON, returns transcriptions. Uses faster-whisper for inference.

Usage:
    python serve.py
    python serve.py --port 8001 --host 0.0.0.0
"""

import base64
import io
import time
import wave
import numpy as np
from scipy.signal import resample
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from loguru import logger
from faster_whisper import WhisperModel

# logger configuration
logger.add(
    "log_folder/{time:YYYY-MM-DD}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

app = FastAPI()

# Constants
RATE = 16000  # Target sampling rate
MODEL_PATH = "ct2_model_fp16"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Initialize the faster-whisper model
logger.info(f"Loading faster-whisper model from {MODEL_PATH} ({DEVICE}/{COMPUTE_TYPE})")
t_load = time.perf_counter()
model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
logger.info(f"Model loaded in {time.perf_counter() - t_load:.2f}s")


# Pydantic models to mirror the Java request/response structure
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


# Function to load and resample audio data
def load_audio_from_base64(audio_content: str, target_rate=RATE):
    """Convert base64 encoded audio to waveform and resample it if necessary."""
    audio_data = base64.b64decode(audio_content)

    with io.BytesIO(audio_data) as wav_file:
        with wave.open(wav_file, 'rb') as wav_file:
            framerate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            audio_data = np.frombuffer(
                wav_file.readframes(num_frames), dtype=np.int16
            ).astype(np.float32) / 32768.0

            if framerate != target_rate:
                num_samples = round(len(audio_data) * target_rate / framerate)
                audio_data = resample(audio_data, num_samples)

            return audio_data


def transcribe_audio(audio_data):
    """Transcribe single audio data using faster-whisper model."""
    segments, info = model.transcribe(
        audio_data,
        language="bn",
        beam_size=1,
        vad_filter=False,
        without_timestamps=True,
    )

    transcription = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return transcription


@app.get("/health")
async def health():
    return {"status": "ok", "model": "faster-whisper", "device": DEVICE}


@app.post("/asr", response_model=AsrResponse)
async def process_asr(request: AsrRequest):
    start_time = time.time()

    if not request.audio or len(request.audio) == 0:
        raise HTTPException(status_code=400, detail="No audio content provided")

    outputs = []
    for i, item in enumerate(request.audio):
        try:
            audio_data = load_audio_from_base64(item.audioContent)
            transcription = transcribe_audio(audio_data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio item {i}: {e}",
            )

        outputs.append(Output(source=transcription))

    end_time = time.time()
    time_taken = end_time - start_time

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

    parser = argparse.ArgumentParser(description="Bengali Batch ASR Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
