#!/usr/bin/env python3
"""Comprehensive ASR benchmark runner.

Auto-detects GPU, runs all backend x chunk_size x batch_size combinations
as isolated subprocesses, handles OOM gracefully, and outputs structured
results + markdown tables.

Usage:
    python benchmark.py                                    # full matrix
    python benchmark.py audio.wav                          # custom audio
    python benchmark.py --backends naive faster_whisper     # subset
    python benchmark.py --chunks 30 --batches 1 2          # subset matrix
    python benchmark.py --load results.json                # regenerate tables
    python benchmark.py --update-readme                    # update README.md
"""

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_AUDIO = os.path.join(
    "meeting_22_11_23", "single_audio",
    "2023-11-22T07_37_42.518693Z_65f437bd-53d0-4ff2-a667-9b5a6935c52d.wav",
)

BACKENDS = {
    "naive": {
        "script": "transcribe_naive.py",
        "label": "Naive HF pipeline",
    },
    "optimized_hf": {
        "script": "transcribe.py",
        "label": "Optimized HF (SDPA)",
    },
    "faster_whisper": {
        "script": "transcribe_fw.py",
        "label": "faster-whisper (CT2)",
    },
}

BACKEND_ORDER = ["naive", "optimized_hf", "faster_whisper"]

OOM_PATTERNS = [
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.OutOfMemoryError",
    "failed to allocate",
    "out of memory",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify_gpu(name: str) -> str:
    """'NVIDIA GeForce RTX 2050' -> 'rtx_2050'."""
    s = name.lower()
    for prefix in ("nvidia geforce ", "nvidia ", "tesla "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def gpu_short_name(name: str) -> str:
    """'NVIDIA GeForce RTX 2050' -> 'RTX 2050'."""
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ---------------------------------------------------------------------------
# GPU detection (subprocess — no torch import in this process)
# ---------------------------------------------------------------------------

def detect_gpu() -> dict:
    """Detect GPU info via a subprocess that imports torch."""
    script = (
        "import torch, json, sys\n"
        "if not torch.cuda.is_available():\n"
        "    json.dump({'name':'CPU','vram_gb':0,'sms':0},sys.stdout)\n"
        "else:\n"
        "    p=torch.cuda.get_device_properties(0)\n"
        "    json.dump({'name':p.name,'vram_gb':round(p.total_memory/(1024**3),1),"
        "'sms':p.multi_processor_count},sys.stdout)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"GPU detection failed: {result.stderr}", file=sys.stderr)
        return {"name": "CPU", "vram_gb": 0, "sms": 0}
    return json.loads(result.stdout)


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via subprocess."""
    abs_path = os.path.join(PROJECT_DIR, audio_path)
    script = (
        "import soundfile, json, sys\n"
        f"info = soundfile.info(r'{abs_path}')\n"
        "json.dump({'duration': round(info.duration, 2)}, sys.stdout)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(f"Could not get audio duration: {result.stderr}", file=sys.stderr)
        return 0.0
    return json.loads(result.stdout)["duration"]


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def build_cmd(backend_key: str, audio: str, chunk_s: int, batch_size: int) -> list[str]:
    """Build the subprocess command for one benchmark run."""
    cfg = BACKENDS[backend_key]
    script_path = os.path.join(PROJECT_DIR, cfg["script"])
    cmd = [
        sys.executable, script_path, audio,
        "--json",
        "--chunk-length", str(chunk_s),
        "--batch-size", str(batch_size),
    ]
    return cmd


def run_single(
    backend_key: str,
    audio: str,
    chunk_s: int,
    batch_size: int,
    timeout: int,
) -> dict:
    """Run one benchmark configuration in an isolated subprocess."""
    cfg = BACKENDS[backend_key]
    cmd = build_cmd(backend_key, audio, chunk_s, batch_size)

    entry = {
        "backend": backend_key,
        "label": cfg["label"],
        "chunk_s": chunk_s,
        "batch_size": batch_size,
        "status": "error",
        "inference_s": None,
        "rtf": None,
        "throughput_x": None,
        "error_msg": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_DIR,
        )
    except subprocess.TimeoutExpired:
        entry["status"] = "timeout"
        entry["error_msg"] = f"Timed out after {timeout}s"
        return entry

    stderr = proc.stderr or ""

    if proc.returncode != 0:
        stderr_lower = stderr.lower()
        if any(p.lower() in stderr_lower for p in OOM_PATTERNS):
            entry["status"] = "oom"
            entry["error_msg"] = "CUDA out of memory"
        else:
            entry["error_msg"] = stderr[:500].strip()
        return entry

    # Parse JSON from stdout
    try:
        data = json.loads(proc.stdout)
        entry["status"] = "ok"
        entry["inference_s"] = data["total_inference_s"]
        entry["rtf"] = data["overall_rtf"]
        entry["throughput_x"] = data["throughput_x"]
    except (json.JSONDecodeError, KeyError) as e:
        entry["error_msg"] = f"JSON parse error: {e}"

    return entry


# ---------------------------------------------------------------------------
# Full benchmark matrix
# ---------------------------------------------------------------------------

def run_benchmark(
    audio: str,
    backends: list[str],
    chunk_sizes: list[int],
    batch_sizes: list[int],
    timeout: int,
    output_path: str,
) -> dict:
    """Run the full test matrix and return results dict."""
    print("Detecting GPU...", flush=True)
    gpu_info = detect_gpu()
    print(f"  GPU: {gpu_info['name']} ({gpu_info['vram_gb']}GB, {gpu_info['sms']} SMs)")

    print("Getting audio duration...", flush=True)
    audio_duration = get_audio_duration(audio)
    print(f"  Audio: {os.path.basename(audio)} ({audio_duration:.1f}s)")

    full_results = {
        "gpu": gpu_info["name"],
        "vram_gb": gpu_info["vram_gb"],
        "sms": gpu_info["sms"],
        "audio_file": os.path.basename(audio),
        "audio_duration_s": audio_duration,
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "results": [],
    }

    # Build ordered run list
    runs = []
    for chunk_s in chunk_sizes:
        for batch_size in batch_sizes:
            for backend in backends:
                runs.append((backend, chunk_s, batch_size))

    total = len(runs)
    t_start = time.perf_counter()
    print(f"\nRunning {total} benchmark configurations...\n", flush=True)

    for i, (backend, chunk_s, batch_size) in enumerate(runs, 1):
        label = BACKENDS[backend]["label"]
        tag = f"[{i}/{total}]"
        print(
            f"  {tag} {label}, chunk={chunk_s}s, batch={batch_size}... ",
            end="", flush=True,
        )

        t0 = time.perf_counter()
        entry = run_single(backend, audio, chunk_s, batch_size, timeout)
        wall = time.perf_counter() - t0

        full_results["results"].append(entry)

        if entry["status"] == "ok":
            print(f"{entry['inference_s']:.1f}s ({entry['throughput_x']:.1f}x) [{wall:.0f}s wall]")
        elif entry["status"] == "oom":
            print(f"OOM [{wall:.0f}s wall]")
        elif entry["status"] == "timeout":
            print(f"TIMEOUT [{timeout}s]")
        else:
            msg = (entry["error_msg"] or "unknown")[:80]
            print(f"ERROR: {msg}")

        # Save intermediate results
        save_results(full_results, output_path)

    elapsed = time.perf_counter() - t_start
    ok = sum(1 for r in full_results["results"] if r["status"] == "ok")
    oom = sum(1 for r in full_results["results"] if r["status"] == "oom")
    fail = total - ok - oom

    print(f"\nDone in {elapsed:.0f}s — {ok} ok, {oom} OOM, {fail} errors")
    print(f"Results saved to {output_path}")

    return full_results


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    """Write results dict to JSON atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_results(path: str) -> dict:
    """Read results dict from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

def format_cell(entry: dict | None) -> str:
    """Format one table cell."""
    if entry is None:
        return "---"
    if entry["status"] == "oom":
        return "OOM"
    if entry["status"] == "timeout":
        return "TIMEOUT"
    if entry["status"] != "ok":
        return "ERROR"
    return f"{entry['inference_s']:.1f}s ({entry['throughput_x']:.1f}x)"


def generate_markdown(results: dict, batch_sizes: list[int] | None = None) -> str:
    """Generate markdown benchmark tables from results."""
    gpu_name = results["gpu"]
    vram = results["vram_gb"]
    audio_file = results["audio_file"]
    audio_dur = results["audio_duration_s"]

    # Build lookup: (backend, chunk_s, batch_size) -> entry
    lookup = {}
    for r in results["results"]:
        key = (r["backend"], r["chunk_s"], r["batch_size"])
        lookup[key] = r

    # Discover dimensions from data
    chunk_sizes = sorted(set(r["chunk_s"] for r in results["results"]))
    if batch_sizes is None:
        batch_sizes = sorted(set(r["batch_size"] for r in results["results"]))
    backends_present = [b for b in BACKEND_ORDER if b in set(r["backend"] for r in results["results"])]

    short_gpu = gpu_short_name(gpu_name)

    lines = []
    lines.append(f"**Audio:** `{audio_file}` — {audio_dur:.0f}s ({audio_dur/60:.0f}m{audio_dur%60:.0f}s) Bengali meeting recording")
    lines.append("")
    lines.append("```bash")
    lines.append(f"AUDIO={DEFAULT_AUDIO}")
    lines.append(f"python benchmark.py $AUDIO    # run full benchmark matrix")
    lines.append("```")
    lines.append("")

    for chunk_s in chunk_sizes:
        lines.append(f"### {short_gpu} ({vram}GB) — {chunk_s}s chunks")
        lines.append("")

        # Header
        batch_cols = " | ".join(f"batch={b}" for b in batch_sizes)
        lines.append(f"| Approach | {batch_cols} |")
        lines.append(f"|---|{'---|' * len(batch_sizes)}")

        for backend in backends_present:
            label = BACKENDS[backend]["label"]
            cells = []
            for b in batch_sizes:
                entry = lookup.get((backend, chunk_s, b))
                cells.append(format_cell(entry))
            row = " | ".join(cells)
            lines.append(f"| {label} | {row} |")

        lines.append("")

    # Key observations
    lines.append("### Key observations")
    lines.append("")
    lines.append("- **Naive HF pipeline** — `transformers.pipeline(chunk_length_s=N, batch_size=N)`. "
                 "No SDPA, no manual optimization.")
    lines.append("- **Optimized HF** — fp16 + SDPA + batched chunks with 5s overlap deduplication.")
    lines.append("- **faster-whisper** — CTranslate2 C++ kernels + Silero VAD. "
                 "Half the VRAM (~800MB vs ~1700MB).")
    lines.append("- **15s vs 30s chunks** — More chunks = more decoder passes = slower.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def update_readme(markdown: str, readme_path: str) -> None:
    """Replace the ## Benchmarks section in README.md."""
    with open(readme_path) as f:
        content = f.read()

    # Find ## Benchmarks
    bench_match = re.search(r"^## Benchmarks\s*$", content, re.MULTILINE)
    if not bench_match:
        print("WARNING: '## Benchmarks' section not found in README.md", file=sys.stderr)
        return

    # Find the next ## heading after Benchmarks
    rest = content[bench_match.end():]
    next_heading = re.search(r"^## ", rest, re.MULTILINE)

    if next_heading:
        end_pos = bench_match.end() + next_heading.start()
    else:
        end_pos = len(content)

    new_section = f"## Benchmarks\n\n{markdown}\n"
    new_content = content[:bench_match.start()] + new_section + content[end_pos:]

    with open(readme_path, "w") as f:
        f.write(new_content)

    print(f"Updated {readme_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ASR benchmark — runs all backends x chunks x batches"
    )
    parser.add_argument(
        "audio", nargs="?", default=DEFAULT_AUDIO,
        help=f"Audio file to benchmark (default: {os.path.basename(DEFAULT_AUDIO)})",
    )
    parser.add_argument(
        "--backends", nargs="+",
        choices=list(BACKENDS.keys()),
        default=list(BACKEND_ORDER),
        help="Backends to test (default: all three)",
    )
    parser.add_argument(
        "--chunks", nargs="+", type=int, default=[15, 30],
        help="Chunk sizes in seconds (default: 15 30)",
    )
    parser.add_argument(
        "--batches", nargs="+", type=int, default=[1, 2, 4],
        help="Batch sizes to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="Timeout per run in seconds (default: 900)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: benchmark_results_{gpu}.json)",
    )
    parser.add_argument(
        "--update-readme", action="store_true",
        help="Replace Benchmarks section in README.md",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Load existing results JSON and regenerate tables (skip running)",
    )
    args = parser.parse_args()

    # --load mode: just regenerate tables
    if args.load:
        results = load_results(args.load)
        md = generate_markdown(results, batch_sizes=args.batches)
        print(md)
        if args.update_readme:
            readme_path = os.path.join(PROJECT_DIR, "README.md")
            update_readme(md, readme_path)
        return

    # Verify audio file exists
    audio_abs = os.path.join(PROJECT_DIR, args.audio) if not os.path.isabs(args.audio) else args.audio
    if not os.path.isfile(audio_abs):
        print(f"Error: audio file not found: {audio_abs}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output is None:
        gpu_info = detect_gpu()
        slug = slugify_gpu(gpu_info["name"])
        args.output = os.path.join(PROJECT_DIR, f"benchmark_results_{slug}.json")

    # Preserve backend ordering
    ordered_backends = [b for b in BACKEND_ORDER if b in args.backends]

    # Run the full matrix
    results = run_benchmark(
        audio=args.audio,
        backends=ordered_backends,
        chunk_sizes=sorted(args.chunks),
        batch_sizes=sorted(args.batches),
        timeout=args.timeout,
        output_path=args.output,
    )

    # Generate and print markdown
    md = generate_markdown(results, batch_sizes=sorted(args.batches))
    print("\n" + "=" * 60)
    print("MARKDOWN TABLES")
    print("=" * 60 + "\n")
    print(md)

    if args.update_readme:
        readme_path = os.path.join(PROJECT_DIR, "README.md")
        update_readme(md, readme_path)


if __name__ == "__main__":
    main()
