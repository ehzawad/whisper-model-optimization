#!/usr/bin/env python3
"""Bengali ASR — Faster Whisper (CTranslate2) inference pipeline.

Uses SYSTRAN/faster-whisper, a CTranslate2-based reimplementation of Whisper
that is up to 4x faster than the HuggingFace pipeline with lower VRAM usage.

The model must first be converted from HuggingFace format:
    ct2-transformers-converter --model . --output_dir ct2_model_fp16 \
        --copy_files tokenizer.json preprocessor_config.json --quantization float16

Usage:
    python transcribe_fw.py audio.wav
    python transcribe_fw.py long_meeting.wav
    python transcribe_fw.py meeting_22_11_23/ --json
    python transcribe_fw.py file1.wav file2.wav --json
    python transcribe_fw.py audio.wav --device cpu --compute-type int8
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CT2_MODEL_DIR = os.path.join(MODEL_DIR, "ct2_model_fp16")
SAMPLE_RATE = 16000


_model = None


def get_model(
    model_dir: str = CT2_MODEL_DIR,
    device: str = "cuda",
    compute_type: str = "float16",
    cpu_threads: int = 4,
) -> "WhisperModel":
    """Load (or return cached) faster-whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            model_dir,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
    return _model



def load_audio(path: str) -> np.ndarray:
    """Load audio as float32 numpy array at 16kHz."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


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



def transcribe_audio(
    audio: np.ndarray,
    model=None,
    language: str = "bn",
    beam_size: int = 1,
    vad_filter: bool = True,
    batch_size: int = 8,
) -> str:
    if model is None:
        model = get_model()

    from faster_whisper import BatchedInferencePipeline

    batched = BatchedInferencePipeline(model=model)
    segments, info = batched.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        batch_size=batch_size,
        without_timestamps=True,
    )

    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


def transcribe_file(
    audio_path: str,
    model=None,
    language: str = "bn",
    beam_size: int = 1,
    vad_filter: bool = True,
    batch_size: int = 8,
) -> str:
    """Transcribe a single audio file."""
    if model is None:
        model = get_model()
    audio = load_audio(audio_path)
    return transcribe_audio(
        audio, model=model, language=language,
        beam_size=beam_size, vad_filter=vad_filter, batch_size=batch_size,
    )


def transcribe_batch(
    audio_paths: list[str],
    model=None,
    language: str = "bn",
    beam_size: int = 1,
    vad_filter: bool = True,
    batch_size: int = 8,
    json_output: bool = False,
) -> dict | str:
    if model is None:
        model = get_model()

    from faster_whisper import BatchedInferencePipeline

    batched = BatchedInferencePipeline(model=model)

    results = []
    total_audio = 0.0
    total_inference = 0.0
    output_lines = []

    for path in audio_paths:
        audio = load_audio(path)
        duration = len(audio) / SAMPLE_RATE

        t0 = time.perf_counter()
        segments, info = batched.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            batch_size=batch_size,
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        elapsed = time.perf_counter() - t0

        total_audio += duration
        total_inference += elapsed

        if json_output:
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
                f"({duration:.1f}s, {elapsed * 1000:.0f}ms, "
                f"RTF={rtf:.3f}, {throughput:.1f}x) ---"
            )
            output_lines.append(text)
            output_lines.append("")

    if json_output:
        overall_rtf = total_inference / max(total_audio, 0.01)
        return {
            "model": "faster-whisper (CTranslate2)",
            "backend": "ctranslate2",
            "compute_type": "float16",
            "total_audio_s": round(total_audio, 2),
            "total_inference_s": round(total_inference, 3),
            "overall_rtf": round(overall_rtf, 4),
            "throughput_x": round(total_audio / max(total_inference, 0.001), 2),
            "results": results,
        }

    if len(audio_paths) > 1:
        rtf = total_inference / max(total_audio, 0.01)
        throughput = total_audio / max(total_inference, 0.001)
        output_lines.append(
            f"=== Total: {total_audio:.0f}s audio in {total_inference:.1f}s "
            f"(RTF={rtf:.4f}, {throughput:.1f}x real-time) ==="
        )

    return "\n".join(output_lines)



def main():
    parser = argparse.ArgumentParser(
        description="Bengali ASR (faster-whisper / CTranslate2)"
    )
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compute-type", default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
    )
    parser.add_argument("--language", default="bn")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--ct2-model", default=CT2_MODEL_DIR,
        help="Path to CTranslate2 converted model dir",
    )
    args = parser.parse_args()

    audio_files = collect_audio_files(args.inputs)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    # Load model
    t_load = time.perf_counter()
    model = get_model(
        model_dir=args.ct2_model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
    )
    t_load = time.perf_counter() - t_load

    if not args.json:
        print(f"Backend: faster-whisper (CTranslate2)", file=sys.stderr)
        print(f"Device: {args.device} | Compute: {args.compute_type}", file=sys.stderr)
        print(f"Model load: {t_load:.2f}s", file=sys.stderr)
        print(f"Files: {len(audio_files)} | VAD: {'ON' if not args.no_vad else 'OFF'}", file=sys.stderr)
        print(file=sys.stderr)

    result = transcribe_batch(
        audio_files,
        model=model,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        batch_size=args.batch_size,
        json_output=args.json,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        lines = result.split("\n") if result else []
        for line in lines:
            if line.startswith("=== Total:"):
                print(line, file=sys.stderr)
            else:
                print(line)


if __name__ == "__main__":
    main()
