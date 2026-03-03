"""FastAPI server for Bengali ASR — streaming WebSocket + single-clip REST.

Multi-session architecture:
  - A single BatchScheduler owns the shared model and processes inference
    requests from all active sessions.
  - Each WebSocket connection gets its own ASRSession, providing full
    session isolation (buffers, VAD state, commit history).
  - The REST /transcribe endpoint delegates to transcribe_audio() from the
    parent transcribe module for single-clip inference.
"""

import asyncio
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .asr_engine import ASRConfig, ASRSession, BatchScheduler
from .audio_utils import pcm16_bytes_to_float32

# ---------------------------------------------------------------------------
# Ensure the project root is importable so we can reach transcribe.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transcribe import get_model, detect_gpu_config, transcribe_audio  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals — populated during lifespan
# ---------------------------------------------------------------------------
CLIENTS_DIR = os.path.join(PROJECT_ROOT, "clients")
SAMPLE_RATE = 16000

scheduler: BatchScheduler | None = None
asr_config: ASRConfig | None = None


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, asr_config

    gpu_cfg = detect_gpu_config()
    logger.info("Detected GPU config: %s", gpu_cfg)

    # Build the shared model via the canonical loader in transcribe.py
    model = get_model()

    asr_config = ASRConfig(model_dir=PROJECT_ROOT)

    scheduler = BatchScheduler(model)
    scheduler.start()
    logger.info("BatchScheduler started — ready for connections")

    yield

    # Graceful shutdown
    logger.info("Shutting down BatchScheduler …")
    scheduler.stop()
    logger.info("Shutdown complete")


app = FastAPI(title="Bengali Streaming ASR", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static / index
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(os.path.join(CLIENTS_DIR, "index.html"))


# ---------------------------------------------------------------------------
# REST single-clip transcription
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe a single audio clip.

    Accepts any format soundfile can read (wav, flac, ogg, etc.).
    Returns JSON: {"text", "duration_s", "inference_ms"}.
    """
    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        return JSONResponse({"error": f"Cannot decode audio: {e}"}, status_code=400)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    duration = len(audio) / SAMPLE_RATE
    model = get_model()

    t0 = time.perf_counter()
    text = await asyncio.to_thread(transcribe_audio, audio, model=model)
    inference_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "text": text,
        "duration_s": round(duration, 2),
        "inference_ms": round(inference_ms, 1),
    })


# ---------------------------------------------------------------------------
# WebSocket streaming transcription
# ---------------------------------------------------------------------------
@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """Streaming transcription over WebSocket.

    Each connection gets its own ASRSession backed by the shared
    BatchScheduler.  Client sends binary PCM16 frames; server sends
    JSON partial/final results.
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    session = ASRSession(config=asr_config, scheduler=scheduler)
    session.start()
    await ws.send_text(json.dumps({
        "type": "info",
        "message": "Session started. Send PCM16 audio at 16 kHz mono.",
    }))

    async def send_pending():
        """Drain any results produced by background inference and forward them."""
        results = session.drain_results()
        for result in results:
            await ws.send_text(json.dumps(result, ensure_ascii=False))

    try:
        while True:
            # Short timeout lets us poll for inference results between chunks
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                await send_pending()
                continue

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                audio_chunk = pcm16_bytes_to_float32(data["bytes"])
                results = session.feed_audio(audio_chunk)
                for result in results:
                    await ws.send_text(json.dumps(result, ensure_ascii=False))

            elif "text" in data and data["text"]:
                if data["text"] == "END":
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass

    # Finalize the session and send any remaining committed text
    final_results = session.end()
    for result in final_results:
        try:
            await ws.send_text(json.dumps(result, ensure_ascii=False))
        except Exception:
            break

    try:
        await ws.close()
    except Exception:
        pass

    logger.info("Session closed")


# ---------------------------------------------------------------------------
# Direct execution (dev convenience)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "streaming.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
