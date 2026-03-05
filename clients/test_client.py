#!/usr/bin/env python3
"""Python WebSocket test client — streams a .wav file to the ASR server."""

import asyncio
import json
import sys
import os

import websockets

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.audio_utils import read_wav_file, resample_if_needed, float32_to_pcm16_bytes

SERVER_URL = "ws://localhost:8000/ws/transcribe"
CHUNK_DURATION_SEC = 0.1  # send 100ms chunks
TARGET_SR = 16000


async def stream_file(wav_path: str):
    """Stream a WAV file to the server and print results."""
    print(f"Loading: {wav_path}")
    audio, sr = read_wav_file(wav_path)
    print(f"  Sample rate: {sr} Hz, Duration: {len(audio)/sr:.1f}s")

    if sr != TARGET_SR:
        print(f"  Resampling {sr} -> {TARGET_SR} Hz...")
        audio = resample_if_needed(audio, sr, TARGET_SR)
        print(f"  Resampled duration: {len(audio)/TARGET_SR:.1f}s")

    chunk_samples = int(CHUNK_DURATION_SEC * TARGET_SR)  # 1600 samples per chunk

    print(f"\nConnecting to {SERVER_URL}")
    async with websockets.connect(SERVER_URL) as ws:
        # Start a task to receive messages
        receive_task = asyncio.create_task(receive_messages(ws))

        # Stream audio chunks with realistic timing
        total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            pcm_bytes = float32_to_pcm16_bytes(chunk)
            await ws.send(pcm_bytes)

            chunk_num = i // chunk_samples + 1
            elapsed = i / TARGET_SR
            if chunk_num % 10 == 0:  # status every second
                print(f"  Sent: {elapsed:.1f}s / {len(audio)/TARGET_SR:.1f}s", end="\r")

            # Simulate real-time pacing
            await asyncio.sleep(CHUNK_DURATION_SEC)

        print(f"\n  All audio sent ({len(audio)/TARGET_SR:.1f}s)")

        # Signal end of stream
        await ws.send("END")

        # Wait for remaining messages
        await asyncio.sleep(3.0)
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

    print("\nDone.")


async def receive_messages(ws):
    """Receive and display messages from the server."""
    try:
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "final":
                print(f"\n  [FINAL] {data['text']}")
                print(f"          ({data.get('start', '?')}s - {data.get('end', '?')}s)")
            elif msg_type == "partial":
                print(f"  [partial] {data['text']}", end="\r")
            elif msg_type == "info":
                print(f"  [info] {data['message']}")
            elif msg_type == "error":
                print(f"  [ERROR] {data['message']}")
    except asyncio.CancelledError:
        pass
    except websockets.exceptions.ConnectionClosed:
        pass


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python -m clients.test_client <audio_file.wav>")
        print(f"\nExample:")
        print(f"  python -m clients.test_client meeting_22_11_23/07_37_44.683093e99f8d5esaiful.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.isfile(wav_path):
        print(f"Error: file not found: {wav_path}")
        sys.exit(1)

    asyncio.run(stream_file(wav_path))


if __name__ == "__main__":
    main()
