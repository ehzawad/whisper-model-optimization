#!/usr/bin/env python3
"""Bengali ASR — Naive HuggingFace pipeline baseline.

Uses transformers.pipeline('automatic-speech-recognition') with sequential
per-chunk decoding. This is the unoptimized baseline for benchmarking against
faster-whisper (transcribe_fw.py).

Usage:
    python transcribe_naive.py audio.wav
    python transcribe_naive.py audio.wav --chunk-length 15
    python transcribe_naive.py meeting_22_11_23/ --json
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_pipe = None


def get_pipeline(chunk_length_s: int = 30, device: str = "cuda"):
    """Load (or return cached) transformers ASR pipeline."""
    global _pipe
    if _pipe is None or _pipe.model.device.type != device:
        import torch
        from transformers import pipeline

        _pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            chunk_length_s=chunk_length_s,
            device=0 if device == "cuda" else -1,
            batch_size=1,
            torch_dtype=torch.float16,
            ignore_warning=True,
        )
        _pipe.model.config.forced_decoder_ids = (
            _pipe.tokenizer.get_decoder_prompt_ids(language="bn", task="transcribe")
        )
    return _pipe


def collect_audio_files(paths: list[str]) -> list[str]:
    """Collect audio files from paths (files and/or directories)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
                files.extend(glob.glob(os.path.join(p, ext)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: {p} not found, skipping", file=sys.stderr)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(
        description="Bengali ASR — Naive HF pipeline baseline"
    )
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--chunk-length", type=int, default=30,
                        help="chunk_length_s passed to pipeline (default: 30)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    audio_files = collect_audio_files(args.inputs)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    t_load = time.perf_counter()
    pipe = get_pipeline(chunk_length_s=args.chunk_length, device=args.device)
    t_load = time.perf_counter() - t_load

    if not args.json:
        print(f"Backend: transformers.pipeline (naive baseline)", file=sys.stderr)
        print(f"Device: {args.device} | chunk_length_s: {args.chunk_length}", file=sys.stderr)
        print(f"Model load: {t_load:.2f}s", file=sys.stderr)
        print(f"Files: {len(audio_files)}", file=sys.stderr)
        print(file=sys.stderr)

    results = []
    total_audio = 0.0
    total_inference = 0.0
    output_lines = []

    for path in audio_files:
        import soundfile as sf
        info = sf.info(path)
        duration = info.duration

        t0 = time.perf_counter()
        out = pipe(path)
        elapsed = time.perf_counter() - t0

        text = out["text"].strip()
        total_audio += duration
        total_inference += elapsed

        if args.json:
            results.append({
                "file": path,
                "text": text,
                "duration": round(duration, 2),
                "inference_ms": round(elapsed * 1000, 1),
                "rtf": round(elapsed / max(duration, 0.01), 4),
            })
        else:
            rtf = elapsed / max(duration, 0.01)
            throughput = duration / max(elapsed, 0.001)
            output_lines.append(
                f"--- {os.path.basename(path)} "
                f"({duration:.1f}s, {elapsed*1000:.0f}ms, "
                f"RTF={rtf:.3f}, {throughput:.1f}x) ---"
            )
            output_lines.append(text)
            output_lines.append("")

    if args.json:
        overall_rtf = total_inference / max(total_audio, 0.01)
        print(json.dumps({
            "model": "Naive HF pipeline (transformers.pipeline)",
            "backend": "huggingface",
            "chunk_length_s": args.chunk_length,
            "total_audio_s": round(total_audio, 2),
            "total_inference_s": round(total_inference, 3),
            "overall_rtf": round(overall_rtf, 4),
            "throughput_x": round(total_audio / max(total_inference, 0.001), 2),
            "results": results,
        }, ensure_ascii=False, indent=2))
    else:
        for line in output_lines:
            if line.startswith("=== Total:"):
                print(line, file=sys.stderr)
            else:
                print(line)

        if len(audio_files) >= 1:
            rtf = total_inference / max(total_audio, 0.01)
            throughput = total_audio / max(total_inference, 0.001)
            print(
                f"=== Total: {total_audio:.0f}s audio in {total_inference:.1f}s "
                f"(RTF={rtf:.4f}, {throughput:.1f}x real-time) ===",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
