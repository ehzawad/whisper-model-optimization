#!/usr/bin/env python3
"""Bengali ASR — HuggingFace inference pipeline.

Used by the streaming server (streaming/server.py).
For offline transcription, use transcribe_fw.py (faster-whisper) instead.
"""

import argparse
import glob
import json
import logging
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000


@dataclass
class GPUConfig:
    """Runtime GPU configuration — drives all optimization decisions."""
    device: str = "cpu"
    gpu_name: str = "CPU"
    vram_gb: float = 0.0
    sms: int = 0
    cudnn: bool = False
    use_compile: bool = False
    batch_size: int = 1
    static_cache: bool = False
    cache_impl: str | None = None


def test_cudnn() -> bool:
    """Test if cuDNN works with Whisper-like Conv1d (80->1024, kernel=3, fp16)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.backends.cudnn.enabled = True
        conv = torch.nn.Conv1d(80, 1024, 3, padding=1).half().cuda()
        x = torch.randn(1, 80, 3000, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            conv(x)
        torch.cuda.synchronize()
        del x, conv
        return True
    except Exception:
        torch.backends.cudnn.enabled = False
        return False


def detect_gpu_config(
    device: str = "cuda",
    batch_size: int = 0,
    no_compile: bool = False,
    no_cudnn: bool = False,
    no_batch: bool = False,
) -> GPUConfig:
    cfg = GPUConfig()

    if not torch.cuda.is_available() or device == "cpu":
        return cfg

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    sms = props.multi_processor_count

    cfg.device = "cuda"
    cfg.gpu_name = props.name
    cfg.vram_gb = round(vram_gb, 1)
    cfg.sms = sms

    if no_cudnn:
        torch.backends.cudnn.enabled = False
        cfg.cudnn = False
    else:
        cudnn_ok = test_cudnn()
        cfg.cudnn = cudnn_ok
        if cudnn_ok:
            torch.backends.cudnn.benchmark = True

    if not no_compile and sms >= 30:
        cfg.use_compile = True
        cfg.static_cache = True
        cfg.cache_impl = "static"

    MODEL_GB = 1.5
    HEADROOM_GB = 1.5
    PER_ITEM_GB = 0.3
    if batch_size > 0:
        cfg.batch_size = batch_size
    elif not no_batch:
        available = vram_gb - MODEL_GB - HEADROOM_GB
        cfg.batch_size = max(1, min(64, int(available / PER_ITEM_GB)))
    return cfg


class WhisperModel:
    """Wraps HuggingFace Whisper model + processor + generation config."""

    def __init__(
        self,
        model_dir: str,
        gpu_config: GPUConfig,
        language: str = "bn",
        warmup: int = 5,
    ):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.gpu_config = gpu_config
        device = gpu_config.device

        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16, attn_implementation="sdpa",
        ).to(device)
        self.model.eval()

        self.forced_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe",
        )
        self.model.generation_config.forced_decoder_ids = self.forced_ids

        self._warmup(warmup)

    def _warmup(self, n: int):
        device = self.gpu_config.device
        cache_impl = self.gpu_config.cache_impl
        dummy = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

        for _ in range(n):
            feats = self.processor(
                dummy, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features.to(device, dtype=torch.float16)
            gen_args = {"max_new_tokens": 10}
            if cache_impl:
                gen_args["cache_implementation"] = cache_impl
            with torch.no_grad():
                self.model.generate(feats, **gen_args)

        batch_size = self.gpu_config.batch_size
        if batch_size > 1:
            batch_feats = self.processor(
                dummy, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features.to(device, dtype=torch.float16)
            batch_feats = batch_feats.expand(min(batch_size, 4), -1, -1)
            gen_args = {"max_new_tokens": 10}
            if cache_impl:
                gen_args["cache_implementation"] = cache_impl
            with torch.no_grad():
                self.model.generate(batch_feats, **gen_args)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def generate(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        gen_args = {"max_new_tokens": 444}
        if self.gpu_config.cache_impl:
            gen_args["cache_implementation"] = self.gpu_config.cache_impl
        gen_args.update(kwargs)
        with torch.no_grad():
            return self.model.generate(features, **gen_args)

    def extract_features(self, audio_arrays: list[np.ndarray]) -> torch.Tensor:
        feats_list = []
        for audio in audio_arrays:
            feats = self.processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features
            feats_list.append(feats)
        return torch.cat(feats_list, dim=0).to(
            self.gpu_config.device, dtype=torch.float16,
        )

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)


_model_lock = threading.Lock()
_model_instance: WhisperModel | None = None


def get_model(
    model_dir: str = MODEL_DIR,
    gpu_config: GPUConfig | None = None,
    language: str = "bn",
    warmup: int = 5,
) -> WhisperModel:
    global _model_instance
    with _model_lock:
        if _model_instance is None:
            if gpu_config is None:
                gpu_config = detect_gpu_config()
            _model_instance = WhisperModel(
                model_dir, gpu_config, language=language, warmup=warmup,
            )
        return _model_instance



def load_audio(path: str) -> np.ndarray:
    """Load audio as float32 numpy array at 16kHz."""
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


def deduplicate_overlap(
    all_chunk_texts: list[tuple[str, bool]],
    overlap_ratio: float,
) -> str:
    all_texts = []
    prev_words = []
    for chunk_text, is_first in all_chunk_texts:
        if not chunk_text:
            continue
        cur_words = chunk_text.split()
        if is_first or not prev_words:
            all_texts.extend(cur_words)
        else:
            # Pass 1: exact suffix-prefix match
            best = 0
            max_check = min(len(prev_words), len(cur_words),
                            max(10, len(cur_words) // 3))
            for k in range(1, max_check + 1):
                if prev_words[-k:] == cur_words[:k]:
                    best = k
            # Pass 2: if no exact match, use proportional trim
            if best == 0:
                best = max(1, int(len(cur_words) * overlap_ratio))
            all_texts.extend(cur_words[best:])
        prev_words = cur_words
    return " ".join(all_texts)


def transcribe_audio(audio: np.ndarray, model: WhisperModel | None = None) -> str:
    if model is None:
        model = get_model()

    chunk_length = 30
    chunk_samples = chunk_length * SAMPLE_RATE
    overlap_samples = 5 * SAMPLE_RATE
    stride_samples = chunk_samples - overlap_samples
    duration = len(audio) / SAMPLE_RATE
    batch_size = model.gpu_config.batch_size

    # Short audio — single inference
    if duration <= chunk_length:
        feats = model.extract_features([audio])
        ids = model.generate(feats)
        texts = model.decode(ids)
        return texts[0].strip() if texts else ""

    # Long audio — batched chunking with overlap dedup
    chunks = []
    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunks.append((audio[pos:end], pos == 0))
        pos += stride_samples

    all_chunk_texts = []
    chunk_audios = [c[0] for c in chunks]
    chunk_is_first = [c[1] for c in chunks]

    for b_start in range(0, len(chunk_audios), batch_size):
        b_audio = chunk_audios[b_start:b_start + batch_size]
        b_feats = model.extract_features(b_audio)
        ids = model.generate(b_feats)
        texts = model.decode(ids)
        for j, t in enumerate(texts):
            idx = b_start + j
            all_chunk_texts.append((t.strip(), chunk_is_first[idx]))

    overlap_ratio = overlap_samples / chunk_samples
    return deduplicate_overlap(all_chunk_texts, overlap_ratio)


def transcribe_file(
    audio_path: str,
    model: WhisperModel | None = None,
    device: str = "cuda",
) -> str:
    if model is None:
        gpu_config = detect_gpu_config(device=device)
        model = get_model(gpu_config=gpu_config)

    audio = load_audio(audio_path)
    return transcribe_audio(audio, model=model)


def transcribe_batch(
    audio_paths: list[str],
    model: WhisperModel | None = None,
    json_output: bool = False,
    chunk_length: int = 30,
    no_async: bool = False,
) -> list[dict] | str:
    if model is None:
        model = get_model()

    gpu_config = model.gpu_config
    batch_size = gpu_config.batch_size
    cache_impl = gpu_config.cache_impl
    device = gpu_config.device

    executor = ThreadPoolExecutor(max_workers=4) if not no_async else None
    if executor:
        futures = {executor.submit(load_audio, p): p for p in audio_paths}
        audio_map = {}
        for fut in futures:
            path = futures[fut]
            audio_map[path] = fut.result()
    else:
        audio_map = {p: load_audio(p) for p in audio_paths}

    short_clips = []
    long_clips = []
    for path in audio_paths:
        audio = audio_map[path]
        duration = len(audio) / SAMPLE_RATE
        if duration > chunk_length:
            long_clips.append((path, audio, duration))
        else:
            short_clips.append((path, audio, duration))

    results = []
    total_audio = 0.0
    total_inference = 0.0
    output_lines = []

    for batch_start in range(0, max(len(short_clips), 1), batch_size):
        batch = short_clips[batch_start:batch_start + batch_size]
        if not batch:
            break

        batch_feats = model.extract_features([audio for _, audio, _ in batch])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        ids = model.generate(batch_feats)
        texts = model.decode(ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        batch_duration = sum(d for _, _, d in batch)
        total_inference += elapsed
        total_audio += batch_duration

        for j, (path, audio, duration) in enumerate(batch):
            text = texts[j].strip() if j < len(texts) else ""
            clip_time = (
                elapsed * (duration / batch_duration)
                if batch_duration > 0 else elapsed / len(batch)
            )

            if json_output:
                results.append({
                    "file": path,
                    "text": text,
                    "duration": round(duration, 2),
                    "inference_ms": round(clip_time * 1000, 1),
                    "rtf": round(clip_time / max(duration, 0.01), 4),
                    "batch_size": len(batch),
                })
            else:
                rtf = clip_time / max(duration, 0.01)
                throughput = duration / max(clip_time, 0.001)
                tag = f" [batch={len(batch)}]" if len(batch) > 1 else ""
                output_lines.append(
                    f"--- {os.path.basename(path)} "
                    f"({duration:.1f}s, {clip_time*1000:.0f}ms, "
                    f"RTF={rtf:.3f}, {throughput:.1f}x){tag} ---"
                )
                output_lines.append(text)
                output_lines.append("")

    chunk_samples = chunk_length * SAMPLE_RATE
    overlap_samples = 5 * SAMPLE_RATE
    stride_samples = chunk_samples - overlap_samples

    for path, audio, duration in long_clips:
        chunks = []
        pos = 0
        while pos < len(audio):
            end = min(pos + chunk_samples, len(audio))
            chunks.append((audio[pos:end], pos == 0))
            pos += stride_samples

        chunk_feats_list = []
        chunk_is_first = []
        for chunk_audio, is_first in chunks:
            feats = model.processor(
                chunk_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features
            chunk_feats_list.append(feats)
            chunk_is_first.append(is_first)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        all_chunk_texts = []
        idx = 0
        for b_start in range(0, len(chunk_feats_list), batch_size):
            b_feats = torch.cat(
                chunk_feats_list[b_start:b_start + batch_size], dim=0
            ).to(device, dtype=torch.float16)

            ids = model.generate(b_feats)
            texts = model.decode(ids)
            for t in texts:
                all_chunk_texts.append((t.strip(), chunk_is_first[idx]))
                idx += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        overlap_ratio = overlap_samples / chunk_samples
        text = deduplicate_overlap(all_chunk_texts, overlap_ratio)
        n_chunks = len(chunks)
        total_inference += elapsed
        total_audio += duration

        if json_output:
            results.append({
                "file": path,
                "text": text,
                "duration": round(duration, 2),
                "inference_ms": round(elapsed * 1000, 1),
                "rtf": round(elapsed / max(duration, 0.01), 4),
                "batch_size": min(batch_size, n_chunks),
                "chunks": n_chunks,
            })
        else:
            rtf = elapsed / max(duration, 0.01)
            throughput = duration / max(elapsed, 0.001)
            output_lines.append(
                f"--- {os.path.basename(path)} "
                f"({duration:.1f}s, {elapsed*1000:.0f}ms, "
                f"RTF={rtf:.3f}, {throughput:.1f}x) "
                f"[{n_chunks} chunks, batch={min(batch_size, n_chunks)}] ---"
            )
            output_lines.append(text)
            output_lines.append("")

    if executor:
        executor.shutdown(wait=False)

    if json_output:
        overall_rtf = total_inference / max(total_audio, 0.01)
        return {
            "model": "HF Whisper Medium fp16+SDPA",
            "backend": "huggingface",
            "gpu": gpu_config.gpu_name,
            "gpu_config": {
                "vram_gb": gpu_config.vram_gb,
                "sms": gpu_config.sms,
                "cudnn": gpu_config.cudnn,
                "static_cache": gpu_config.static_cache,
                "compile": gpu_config.use_compile,
                "batch_size": batch_size,
            },
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
    parser = argparse.ArgumentParser(description="Bengali ASR (HF fp16+SDPA)")
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--language", default="bn")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup inferences")
    parser.add_argument("--no-async", action="store_true", help="Disable async I/O")
    parser.add_argument("--chunk-length", type=int, default=30,
                        help="Chunk length in seconds for long audio")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size for short clips (0=auto)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile + static cache")
    parser.add_argument("--no-cudnn", action="store_true",
                        help="Force-disable cuDNN")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batch inference")
    args = parser.parse_args()

    audio_files = collect_audio_files(args.inputs)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    gpu_config = detect_gpu_config(
        device=args.device,
        batch_size=args.batch_size,
        no_compile=args.no_compile,
        no_cudnn=args.no_cudnn,
        no_batch=args.no_batch,
    )

    t_load = time.perf_counter()
    wm = get_model(
        model_dir=MODEL_DIR,
        gpu_config=gpu_config,
        language=args.language,
        warmup=args.warmup,
    )
    t_load = time.perf_counter() - t_load

    if not args.json:
        cudnn_s = "ON" if gpu_config.cudnn else "OFF"
        compile_s = "ON" if gpu_config.use_compile else "OFF"
        cache_s = "static" if gpu_config.static_cache else "dynamic"
        print(f"GPU: {gpu_config.gpu_name} ({gpu_config.vram_gb}GB, "
              f"{gpu_config.sms} SMs)", file=sys.stderr)
        print(f"cuDNN: {cudnn_s} | compile: {compile_s} | cache: {cache_s} | "
              f"batch: {gpu_config.batch_size}", file=sys.stderr)
        print(f"Load+warmup: {t_load:.2f}s ({args.warmup}x warmup)",
              file=sys.stderr)
        print(f"Files: {len(audio_files)} | Chunk: {args.chunk_length}s",
              file=sys.stderr)
        print(file=sys.stderr)

    result = transcribe_batch(
        audio_files,
        model=wm,
        json_output=args.json,
        chunk_length=args.chunk_length,
        no_async=args.no_async,
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
