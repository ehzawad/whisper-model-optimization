#!/usr/bin/env python3
"""FastAPI batch ASR server — Bengali Whisper with custom Triton GPU kernels.

HF Transformers + Flash Attention 2 backend with fused Triton kernels that
reduce global memory round-trips on bandwidth-limited GPUs (RTX 2050: 96 GB/s).

Custom kernels:
    fused_residual_layernorm  — 1.29x faster than PyTorch (saves 33% bandwidth)
    fused_residual_add_clamp  — 1.69x faster than PyTorch (encoder epilogue)

Cross-client batching (100ms window) groups audio from multiple concurrent
clients into a single GPU batch.

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
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import resample_poly

from loguru import logger

# cuDNN is broken on this system (RTX 2050, cuDNN 9.11)
torch.backends.cudnn.enabled = False

# logger configuration
logger.add(
    "log_folder/{time:YYYY-MM-DD}.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE  # 480000 samples = 30s Whisper encoder window
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"

BATCH_TIMEOUT_S = 0.1   # 100ms collection window
GPU_BATCH_SIZE = 4       # RTX 2050 (4GB); increase to 16 for T4 (16GB)
CLIENT_TIMEOUT_S = 300   # Max wait for client futures (5 min)
BATCH_FORWARD_TIMEOUT_S = 120  # Max time for a single GPU batch

# ── Custom Triton kernels ───────────────────────────────────────────────────
from triton_kernels import fused_residual_layernorm, fused_residual_add_clamp, triton_gelu

# ── Model loading (HF Transformers + Flash Attention 2) ──────────────────────
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger.info(f"Loading HF Whisper model from {MODEL_DIR} (fp16/flash_attention_2)")
t_load = time.perf_counter()

_processor = WhisperProcessor.from_pretrained(MODEL_DIR)
_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
).to(DEVICE)
_model.eval()

# Set forced decoder IDs for Bengali transcription
_forced_ids = _processor.get_decoder_prompt_ids(language="bn", task="transcribe")
_model.generation_config.forced_decoder_ids = _forced_ids

logger.info(f"Model ready in {time.perf_counter() - t_load:.2f}s")


# ── Monkey-patch: inject Triton kernels into encoder layers ─────────────────
def _make_fused_encoder_forward(layer):
    """Create a fused forward for WhisperEncoderLayer using Triton kernels."""
    self_attn = layer.self_attn
    fc1 = layer.fc1
    fc2 = layer.fc2
    sa_ln_w = layer.self_attn_layer_norm.weight
    sa_ln_b = layer.self_attn_layer_norm.bias
    ff_ln_w = layer.final_layer_norm.weight
    ff_ln_b = layer.final_layer_norm.bias

    def forward(hidden_states, attention_mask, layer_head_mask, output_attentions=False):
        # Pre-norm for self-attention (no residual add yet, just LayerNorm)
        residual = hidden_states
        hidden_states = nn.functional.layer_norm(
            hidden_states, (hidden_states.size(-1),), sa_ln_w, sa_ln_b,
        )

        # Self-attention (Flash Attention 2)
        hidden_states, attn_weights = self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # dropout is no-op in eval — skip kernel launch

        # FUSED: residual add + final_layer_norm (Triton kernel, 1.29x faster)
        hidden_states, residual = fused_residual_layernorm(
            hidden_states, residual, ff_ln_w, ff_ln_b,
        )

        # FFN
        hidden_states = triton_gelu(fc1(hidden_states))
        hidden_states = fc2(hidden_states)
        # dropout is no-op in eval — skip kernel launch

        # FUSED: residual add + fp16 clamp (Triton kernel, 1.69x faster)
        hidden_states = fused_residual_add_clamp(hidden_states, residual)

        return hidden_states, attn_weights

    return forward


def _make_fused_decoder_forward(layer):
    """Create a fused forward for WhisperDecoderLayer using Triton kernels."""
    self_attn = layer.self_attn
    encoder_attn = layer.encoder_attn
    fc1 = layer.fc1
    fc2 = layer.fc2
    sa_ln_w = layer.self_attn_layer_norm.weight
    sa_ln_b = layer.self_attn_layer_norm.bias
    ca_ln_w = layer.encoder_attn_layer_norm.weight
    ca_ln_b = layer.encoder_attn_layer_norm.bias
    ff_ln_w = layer.final_layer_norm.weight
    ff_ln_b = layer.final_layer_norm.bias

    def forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=True,
        cache_position=None,
    ):
        # ── Self-attention sub-block ──
        residual = hidden_states
        hidden_states = nn.functional.layer_norm(
            hidden_states, (hidden_states.size(-1),), sa_ln_w, sa_ln_b,
        )
        hidden_states, self_attn_weights = self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # ── Cross-attention sub-block ──
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = nn.functional.layer_norm(
                hidden_states, (hidden_states.size(-1),), ca_ln_w, ca_ln_b,
            )
            hidden_states, cross_attn_weights = encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = residual + hidden_states

        # ── FFN sub-block ──
        residual = hidden_states
        hidden_states = nn.functional.layer_norm(
            hidden_states, (hidden_states.size(-1),), ff_ln_w, ff_ln_b,
        )
        hidden_states = triton_gelu(fc1(hidden_states))
        hidden_states = fc2(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs

    return forward


# Apply monkey-patches
_n_patched = 0
for layer in _model.model.encoder.layers:
    layer.forward = _make_fused_encoder_forward(layer)
    _n_patched += 1
for layer in _model.model.decoder.layers:
    layer.forward = _make_fused_decoder_forward(layer)
    _n_patched += 1
logger.info(f"Patched {_n_patched} layers with Triton kernels")

# Thread pool for parallel CPU work (base64 decode, resample, feature extraction)
_decode_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# Batch queue: each item is (audio_ndarray, asyncio.Future)
_batch_queue: asyncio.Queue[Tuple[np.ndarray, asyncio.Future]] = None

# ── Model warm-up ────────────────────────────────────────────────────────────
logger.info("Warming up model with Triton kernels...")
_warmup = _processor(
    np.zeros(SAMPLE_RATE, dtype=np.float32),
    sampling_rate=SAMPLE_RATE, return_tensors="pt",
).input_features.to(DEVICE, dtype=torch.float16)
with torch.inference_mode():
    _model.generate(_warmup, max_new_tokens=10)
del _warmup
logger.info("Warm-up complete")


# ── Pydantic models (same API contract) ─────────────────────────────────────
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


# ── Audio helpers ────────────────────────────────────────────────────────────
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


def _extract_features_for_audio(audio_idx: int, audio: np.ndarray):
    """Extract features for one audio, chunking if >30s. Per-audio error isolation."""
    try:
        chunks = []
        for start in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[start : start + CHUNK_SAMPLES]
            feats = _processor(
                chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features  # [1, 80, 3000]
            chunks.append(feats)
        return (audio_idx, chunks, None)
    except Exception as e:
        return (audio_idx, None, e)


def transcribe_batch(audios: list[np.ndarray], batch_size: int = GPU_BATCH_SIZE) -> list[str]:
    """Transcribe multiple audios with batched model.generate() + Triton kernels.

    Extracts features for all audios in parallel, stacks into one tensor,
    runs ONE model.generate() call per GPU batch.
    """
    if not audios:
        return []

    # ── 1. Parallel feature extraction with per-audio error isolation ──
    futures = [
        _decode_pool.submit(_extract_features_for_audio, i, audio)
        for i, audio in enumerate(audios)
    ]

    all_features = []  # List of [1, 80, 3000] tensors
    chunk_to_audio = []

    for fut in futures:
        audio_idx, chunks, error = fut.result()
        if error is not None:
            logger.error(f"Feature extraction failed for audio {audio_idx}: {error}")
            continue
        for feat in chunks:
            all_features.append(feat)
            chunk_to_audio.append(audio_idx)

    if not all_features:
        return [""] * len(audios)

    # ── 2. Stack on CPU ──
    features = torch.cat(all_features, dim=0)

    # ── 3. Batched GPU inference with Triton kernels ──
    chunk_texts = []
    for i in range(0, len(features), batch_size):
        batch_feats = features[i : i + batch_size].to(DEVICE, dtype=torch.float16)

        with torch.inference_mode():
            ids = _model.generate(batch_feats, max_new_tokens=444)

        texts = _processor.batch_decode(ids, skip_special_tokens=True)
        chunk_texts.extend([t.strip() for t in texts])

    # ── 4. Reassemble chunk texts per original audio ──
    audio_texts = [""] * len(audios)
    for chunk_idx, audio_idx in enumerate(chunk_to_audio):
        if chunk_texts[chunk_idx]:
            if audio_texts[audio_idx]:
                audio_texts[audio_idx] += " " + chunk_texts[chunk_idx]
            else:
                audio_texts[audio_idx] = chunk_texts[chunk_idx]

    return audio_texts


# ── Batch worker — cross-client batching ────────────────────────────────────
async def _batch_worker():
    """Collect up to GPU_BATCH_SIZE items (100ms window), process, deliver."""
    loop = asyncio.get_event_loop()

    while True:
        first_item = await _batch_queue.get()
        batch = [first_item]

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

        audios = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        logger.info(f"Batch worker: processing {len(audios)} items (queue={_batch_queue.qsize()})")

        try:
            texts = await asyncio.wait_for(
                loop.run_in_executor(None, transcribe_batch, audios),
                timeout=BATCH_FORWARD_TIMEOUT_S,
            )
            for fut, text in zip(futures, texts):
                if not fut.done():
                    fut.set_result(text)
        except asyncio.TimeoutError:
            logger.error(f"Batch transcription timed out after {BATCH_FORWARD_TIMEOUT_S}s")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(TimeoutError("GPU transcription timed out"))
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _batch_queue
    _batch_queue = asyncio.Queue(maxsize=200)
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
        "model": "whisper-medium-bn (HF + FA2 + Triton kernels)",
        "device": DEVICE,
        "backend": "transformers + flash_attn + triton_kernels",
        "gpu_batch_size": GPU_BATCH_SIZE,
        "batch_window_ms": BATCH_TIMEOUT_S * 1000,
        "queue_depth": _batch_queue.qsize() if _batch_queue else 0,
    }


@app.post("/asr", response_model=AsrResponse)
async def process_asr(request: AsrRequest):
    start_time = time.time()

    if not request.audio or len(request.audio) == 0:
        raise HTTPException(status_code=400, detail="No audio content provided")

    # Parallel CPU decode with per-audio isolation
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
            try:
                await asyncio.wait_for(_batch_queue.put((result, fut)), timeout=10.0)
            except asyncio.TimeoutError:
                for prev_fut in batch_futs:
                    prev_fut.cancel()
                raise HTTPException(
                    status_code=503, detail="Server overloaded, try again later"
                )
            batch_futs.append(fut)
            output_map[i] = fut

    # Wait for the batch worker to process our items
    if batch_futs:
        try:
            await asyncio.wait_for(
                asyncio.gather(*batch_futs),
                timeout=CLIENT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504, detail="Transcription timed out"
            )
        except Exception as e:
            logger.error(f"Batch transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

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

    logger.info(f"Processed {len(outputs)} audio(s) in {time_taken:.3f}s")
    return response


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Bengali ASR Server (HF + FA2 + Triton kernels, batched)"
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
