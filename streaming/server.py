"""FastAPI WebSocket server for real-time Bengali streaming ASR."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .asr_engine import ASRConfig, StreamingASREngine
from .audio_utils import pcm16_bytes_to_float32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENTS_DIR = os.path.join(PROJECT_ROOT, "clients")

engine: StreamingASREngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    config = ASRConfig(model_dir=PROJECT_ROOT)
    engine = StreamingASREngine(config)
    logger.info("ASR engine ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Bengali Streaming ASR", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(os.path.join(CLIENTS_DIR, "index.html"))


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """Streaming transcription over WebSocket.

    Client sends binary PCM16 frames. Server sends JSON partial/final.
    Inference runs in a background thread; we poll for results while
    processing incoming audio to keep latency low.
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    engine.start_session()
    await ws.send_text(json.dumps({"type": "info", "message": "Session started. Send PCM16 audio at 16kHz mono."}))

    running = True

    async def send_pending():
        """Drain any results from background inference and send them."""
        results = engine._drain_results()
        for result in results:
            await ws.send_text(json.dumps(result, ensure_ascii=False))

    try:
        while running:
            # Use a short timeout so we can poll for inference results
            # even between audio chunks
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                # No data from client — just check for inference results
                await send_pending()
                continue

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                audio_chunk = pcm16_bytes_to_float32(data["bytes"])
                # feed_audio returns any immediately available results
                results = engine.feed_audio(audio_chunk)
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

    # Finalize
    final_results = engine.end_session()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "streaming.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
