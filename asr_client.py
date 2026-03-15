#!/usr/bin/env python3
"""CLI client for the Bengali ASR server (serve.py).

Reads audio files, base64-encodes them, and POSTs to the /asr endpoint.

Usage:
    python asr_client.py audio.wav
    python asr_client.py file1.wav file2.wav
    python asr_client.py meeting_22_11_23/
    python asr_client.py audio.wav --json
    python asr_client.py audio.wav --server http://192.168.1.5:8001
"""

import argparse
import base64
import json
import os
import sys

import httpx

# Import collect_audio_files from the existing transcribe_fw module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transcribe_fw import collect_audio_files


def main():
    parser = argparse.ArgumentParser(description="Bengali ASR CLI client")
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or directory")
    parser.add_argument(
        "--server", default="http://localhost:8001",
        help="ASR server URL (default: http://localhost:8001)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    parser.add_argument("--language", default="bn", help="Source language (default: bn)")
    args = parser.parse_args()

    # Collect audio files
    audio_files = collect_audio_files(args.inputs)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    # Health check
    try:
        resp = httpx.get(f"{args.server}/health", timeout=5.0)
        resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException):
        print(f"Error: cannot reach server at {args.server}", file=sys.stderr)
        print("Is the server running? Start it with: python serve.py", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"Server: {args.server}", file=sys.stderr)
        print(f"Files: {len(audio_files)}", file=sys.stderr)
        print(file=sys.stderr)

    # Read and base64-encode each file
    audio_items = []
    for path in audio_files:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        audio_items.append({"audioContent": b64})

    # Build request
    payload = {
        "config": {"language": {"sourceLanguage": args.language}},
        "audio": audio_items,
    }

    # POST to server
    try:
        resp = httpx.post(
            f"{args.server}/asr",
            json=payload,
            timeout=600.0,  # 10 min for large batches
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        print("Error: request timed out", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        print(f"Error {e.response.status_code}: {detail or e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # Pretty-print results
    for i, output in enumerate(data.get("output", [])):
        fname = os.path.basename(audio_files[i]) if i < len(audio_files) else f"Audio {i}"
        print(f"--- {fname} ---")
        print(output.get("source", "(no speech detected)"))
        print()

    print(f"Total time: {data.get('time_taken', 0):.2f}s (server-reported)", file=sys.stderr)


if __name__ == "__main__":
    main()
