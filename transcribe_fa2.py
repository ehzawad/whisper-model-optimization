#!/usr/bin/env python3
"""Bengali ASR — HF Transformers + Flash Attention 2 CLI.

Batched inference using model.generate() with Flash Attention 2.
Same engine as serve4.py but as a direct CLI tool — no server, no queue.

Requires: pip install flash-attn

Usage:
    python transcribe_fa2.py audio.wav
    python transcribe_fa2.py file1.wav file2.wav
    python transcribe_fa2.py meeting_22_11_23/
    python transcribe_fa2.py meeting_22_11_23/ --json
    python transcribe_fa2.py ec-audio/ --batch-size 8
    python transcribe_fa2.py audio.wav --device cpu
"""

import argparse
import json
import os
import sys
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import torch

# cuDNN is broken on this system (RTX 2050, cuDNN 9.11)
torch.backends.cudnn.enabled = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE  # 30s Whisper encoder window


# ── Verify flash-attn is installed ──────────────────────────────────────────
try:
    import flash_attn  # noqa: F401
except ImportError:
    print(
        "Error: flash-attn is not installed.\n"
        "Install with: pip install flash-attn --no-build-isolation\n"
        "Or pre-built wheel: pip install 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl'",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Audio loading (from transcribe_fw.py) ────────────────────────────────────
def collect_audio_files(paths: list[str]) -> list[str]:
    """Collect audio files from paths (files and/or directories)."""
    import glob as globmod
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
                files.extend(globmod.glob(os.path.join(p, ext)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: {p} not found, skipping", file=sys.stderr)
    return sorted(set(files))


def load_audio(path: str) -> np.ndarray:
    """Load audio as float32 numpy array at 16kHz."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
    return audio


# ── GPU auto-detection (from transcribe.py) ──────────────────────────────────
def detect_batch_size(device: str) -> int:
    """Auto-detect batch size from GPU VRAM."""
    if device == "cpu" or not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    MODEL_GB = 1.5
    HEADROOM_GB = 0.5
    PER_ITEM_GB = 0.3
    available = vram_gb - MODEL_GB - HEADROOM_GB
    return max(1, min(64, int(available / PER_ITEM_GB)))


# ── Model loading ────────────────────────────────────────────────────────────
def load_model(device: str = "cuda"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    forced_ids = processor.get_decoder_prompt_ids(language="bn", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_ids

    # Verify Flash Attention 2 is active
    attn_impl = model.config._attn_implementation
    from transformers.models.whisper.modeling_whisper import ALL_ATTENTION_FUNCTIONS
    attn_fn = ALL_ATTENTION_FUNCTIONS.get(attn_impl, None)
    print(f"Attention: {attn_impl} → {attn_fn.__name__ if attn_fn else 'NOT FOUND'}", file=sys.stderr)

    return model, processor


# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(audio: np.ndarray, processor) -> list:
    """Extract features for one audio, chunking if >30s."""
    chunks = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        feats = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        ).input_features  # [1, 80, 3000]
        chunks.append(feats)
    return chunks


# ── Batch transcription ─────────────────────────────────────────────────────
def transcribe_batch(
    audios: list[np.ndarray],
    model,
    processor,
    batch_size: int,
    device: str,
) -> list[str]:
    """Transcribe multiple audios with batched model.generate() + Flash Attention 2."""
    if not audios:
        return []

    pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    # Parallel feature extraction
    def _extract(args):
        idx, audio = args
        try:
            return (idx, extract_features(audio, processor), None)
        except Exception as e:
            return (idx, None, e)

    futures = [pool.submit(_extract, (i, a)) for i, a in enumerate(audios)]

    all_features = []
    chunk_to_audio = []
    for fut in futures:
        idx, chunks, err = fut.result()
        if err is not None:
            print(f"Warning: feature extraction failed for audio {idx}: {err}", file=sys.stderr)
            continue
        for feat in chunks:
            all_features.append(feat)
            chunk_to_audio.append(idx)

    pool.shutdown(wait=False)

    if not all_features:
        return [""] * len(audios)

    features = torch.cat(all_features, dim=0)
    total_chunks = len(features)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    # Batched GPU inference
    chunk_texts = []
    for batch_num, i in enumerate(range(0, len(features), batch_size), 1):
        batch_feats = features[i : i + batch_size].to(device, dtype=torch.float16)
        cur_size = len(batch_feats)
        print(
            f"\r  Batch {batch_num}/{total_batches} [bs={cur_size}] ({i + cur_size}/{total_chunks} chunks)",
            end="", file=sys.stderr, flush=True,
        )
        with torch.no_grad():
            ids = model.generate(batch_feats, max_new_tokens=444)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        chunk_texts.extend([t.strip() for t in texts])
    print(file=sys.stderr)  # newline after progress

    # Reassemble per original audio
    audio_texts = [""] * len(audios)
    for chunk_idx, audio_idx in enumerate(chunk_to_audio):
        if chunk_texts[chunk_idx]:
            if audio_texts[audio_idx]:
                audio_texts[audio_idx] += " " + chunk_texts[chunk_idx]
            else:
                audio_texts[audio_idx] = chunk_texts[chunk_idx]

    return audio_texts


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bengali ASR (HF Transformers + Flash Attention 2)"
    )
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=0,
                        help="GPU batch size (0 = auto-detect from VRAM)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--language", default="bn")
    args = parser.parse_args()

    # Collect files
    audio_files = collect_audio_files(args.inputs)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect batch size
    batch_size = args.batch_size if args.batch_size > 0 else detect_batch_size(args.device)

    # Load model
    t_load = time.perf_counter()
    model, processor = load_model(device=args.device)
    t_load = time.perf_counter() - t_load

    if not args.json:
        print(f"Backend: HF Transformers + Flash Attention 2", file=sys.stderr)
        print(f"Device: {args.device} | Batch size: {batch_size}", file=sys.stderr)
        print(f"Model load: {t_load:.2f}s", file=sys.stderr)
        print(f"Files: {len(audio_files)}", file=sys.stderr)
        print(file=sys.stderr)

    # Load audio files in parallel
    print(f"Loading {len(audio_files)} audio file(s)...", file=sys.stderr)
    pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    audio_futures = [(pool.submit(load_audio, p), p) for p in audio_files]
    audios = []
    for i, (fut, path) in enumerate(audio_futures, 1):
        audios.append(fut.result())
        print(f"\r  Loaded {i}/{len(audio_files)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    pool.shutdown(wait=False)

    # Transcribe
    t_infer = time.perf_counter()
    texts = transcribe_batch(audios, model, processor, batch_size, args.device)
    t_infer = time.perf_counter() - t_infer

    total_audio = sum(len(a) / SAMPLE_RATE for a in audios)

    # Output
    if args.json:
        results = []
        for i, (path, text) in enumerate(zip(audio_files, texts)):
            duration = len(audios[i]) / SAMPLE_RATE
            results.append({
                "file": path,
                "text": text,
                "duration": round(duration, 2),
            })
        output = {
            "backend": "transformers + flash_attention_2",
            "batch_size": batch_size,
            "total_audio_s": round(total_audio, 2),
            "total_inference_s": round(t_infer, 3),
            "overall_rtf": round(t_infer / max(total_audio, 0.01), 4),
            "throughput_x": round(total_audio / max(t_infer, 0.001), 2),
            "results": results,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for i, (path, text) in enumerate(zip(audio_files, texts)):
            duration = len(audios[i]) / SAMPLE_RATE
            item_time = t_infer * (duration / max(total_audio, 0.01))
            rtf = item_time / max(duration, 0.01)
            throughput = duration / max(item_time, 0.001)
            print(f"--- {os.path.basename(path)} ({duration:.1f}s, RTF={rtf:.3f}, {throughput:.1f}x) ---")
            print(text)
            print()

        rtf = t_infer / max(total_audio, 0.01)
        throughput = total_audio / max(t_infer, 0.001)
        print(
            f"=== Total: {total_audio:.0f}s audio in {t_infer:.1f}s "
            f"(RTF={rtf:.4f}, {throughput:.1f}x real-time) ===",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
