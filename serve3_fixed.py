#!/usr/bin/env python3
"""FastAPI batch ASR server — Bengali Whisper with cross-client GPU batching.

Builds on serve2.py: adds a 100ms collection window that groups audio from
multiple concurrent clients into a single GPU batch.  When only one client
is active the extra latency is at most 100ms.

Usage:
    python serve3.py
    python serve3.py --port 8003 --host 0.0.0.0
"""

import asyncio
import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Tuple

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_suppressed_tokens
from pydantic import BaseModel
from scipy.signal import resample_poly

from loguru import logger

# logger configuration
logger.add(
    "log_folder/{time:YYYY-MM-DD}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CT2_MODEL_DIR = os.path.join(MODEL_DIR, "ct2_model_fp16")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"

BATCH_TIMEOUT_S = 0.1   # 100ms collection window
GPU_BATCH_SIZE = 4       # RTX 2050 (4GB); increase to 16 for T4 (16GB)

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

# Batch queue: each item is (audio_ndarray, asyncio.Future)
_batch_queue: asyncio.Queue[Tuple[np.ndarray, asyncio.Future]] = None


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


# ── Audio helpers (reused from serve2.py) ────────────────────────────────────
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


def transcribe_batch(audios: list[np.ndarray], batch_size: int = GPU_BATCH_SIZE) -> list[str]:
    """Transcribe multiple audio arrays in batched GPU passes.

    Calls _pipeline.forward() directly with stacked features from multiple
    audios.  Audio clips longer than 30s are auto-chunked into <=30s pieces,
    all chunks are batched together, and text is reassembled per original audio.
    """
    if not audios:
        return []

    CHUNK_SAMPLES = 30 * SAMPLE_RATE

    # ── 1. Split long audios into <=30s chunks, extract mel features (CPU) ──
    features_list = []
    chunks_metadata = []
    chunk_to_audio = []

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

    # ── 3. Build tokenizer + options ──
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

    # ── 5. Reassemble chunk texts per original audio ──
    audio_texts = [""] * len(audios)
    for chunk_idx, audio_idx in enumerate(chunk_to_audio):
        if chunk_texts[chunk_idx]:
            if audio_texts[audio_idx]:
                audio_texts[audio_idx] += " " + chunk_texts[chunk_idx]
            else:
                audio_texts[audio_idx] = chunk_texts[chunk_idx]

    return audio_texts


# ── Batch worker (background task) ───────────────────────────────────────────
async def _batch_worker():
    """Collect up to GPU_BATCH_SIZE items (100ms window), process, deliver immediately.

    If more items remain in the queue after processing, grab the next batch
    right away (no 100ms wait) so overflow clients aren't penalized.
    """
    loop = asyncio.get_event_loop()

    while True:
        # Block until at least one item arrives
        first_item = await _batch_queue.get()
        batch = [first_item]

        # Collect up to GPU_BATCH_SIZE items within 100ms
        deadline = loop.time() + BATCH_TIMEOUT_S
        while len(batch) < GPU_BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    _batch_queue.get(), timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # Process this batch and deliver results immediately
        audios = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        logger.info(f"Batch worker: processing {len(audios)} items")

        try:
            texts = await loop.run_in_executor(
                None, transcribe_batch, audios
            )
            for fut, text in zip(futures, texts):
                if not fut.done():
                    fut.set_result(text)
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

        # If queue still has items, loop back immediately (no 100ms wait)
        # — the next iteration's await _batch_queue.get() returns instantly


# ── Lifespan: start/stop the batch worker ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _batch_queue
    _batch_queue = asyncio.Queue()
    worker_task = asyncio.create_task(_batch_worker())
    logger.info(
        f"Batch worker started (window={BATCH_TIMEOUT_S*1000:.0f}ms, "
        f"gpu_batch={GPU_BATCH_SIZE})"
    )
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "faster-whisper",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "batch_window_ms": BATCH_TIMEOUT_S * 1000,
    }


@app.post("/asr", response_model=AsrResponse)
async def process_asr(request: AsrRequest):
    start_time = time.time()

    if not request.audio or len(request.audio) == 0:
        raise HTTPException(status_code=400, detail="No audio content provided")

    # Parallel CPU decode: base64 → resample → numpy arrays (per-audio isolation)
    loop = asyncio.get_event_loop()
    decode_futs = [
        loop.run_in_executor(_decode_pool, load_audio_from_base64, item.audioContent)
        for item in request.audio
    ]
    results = await asyncio.gather(*decode_futs, return_exceptions=True)

    # Submit successfully decoded audios to batch queue, track failures
    batch_futs = []
    output_map = {}  # index → future or error string
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Decode failed for audio {i}: {result}")
            output_map[i] = ""
        else:
            fut = loop.create_future()
            await _batch_queue.put((result, fut))
            batch_futs.append(fut)
            output_map[i] = fut

    # Wait for the batch worker to process our items
    if batch_futs:
        try:
            await asyncio.gather(*batch_futs)
        except Exception as e:
            logger.error(f"Batch transcription error: {e}")

    # Assemble outputs preserving original order
    outputs = []
    for i in range(len(request.audio)):
        val = output_map[i]
        if isinstance(val, str):
            outputs.append(Output(source=val))
        elif val.done() and not val.cancelled() and val.exception() is None:
            outputs.append(Output(source=val.result()))
        else:
            outputs.append(Output(source=""))
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

    parser = argparse.ArgumentParser(
        description="Bengali Batch ASR Server (cross-client batching)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
