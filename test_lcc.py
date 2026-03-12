#!/usr/bin/env python3
"""Test client for live_caption.py — streams a WAV file over WebSocket.

Usage:
    python test_lcc.py meeting_22_11_23/07_37_44.683093e99f8d5esaiful.wav
    python test_lcc.py some_audio.wav --rate 16000
"""

import asyncio
import json
import sys
import os

import numpy as np
import soundfile as sf
import websockets

SERVER_URL = "ws://localhost:2700"
CHUNK_DURATION = 0.5  # seconds, matches live_caption.py's INCOMING_AUDIO_DURATION


async def stream_file(wav_path: str, send_rate: int = 48000):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to send_rate if file SR differs
    if sr != send_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=send_rate)

    duration = len(audio) / send_rate
    chunk_samples = int(CHUNK_DURATION * send_rate)
    print(f"Audio: {wav_path} ({duration:.1f}s at {send_rate}Hz)")
    print(f"Sending {CHUNK_DURATION}s chunks to {SERVER_URL}\n")

    async with websockets.connect(SERVER_URL) as ws:
        recv_task = asyncio.create_task(receive(ws))

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            pcm16 = (chunk * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
            await ws.send(pcm16)
            elapsed = i / send_rate
            print(f"  sent {elapsed:.1f}s / {duration:.1f}s", end="\r")
            await asyncio.sleep(CHUNK_DURATION)

        print(f"\n  All audio sent ({duration:.1f}s)")
        await ws.send("END")
        await asyncio.sleep(3.0)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    print("\nDone.")


async def receive(ws):
    try:
        async for msg in ws:
            data = json.loads(msg)
            if "text" in data:
                print(f"\n  [FINAL]   {data['text']}")
            elif "partial" in data:
                print(f"  [partial] {data['partial']}", end="\r")
    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
        pass


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python test_lcc.py <audio.wav> [--rate 48000]")
        sys.exit(1)

    wav_path = sys.argv[1]
    send_rate = 48000
    if "--rate" in sys.argv:
        idx = sys.argv.index("--rate")
        send_rate = int(sys.argv[idx + 1])

    if not os.path.isfile(wav_path):
        print(f"Error: {wav_path} not found")
        sys.exit(1)

    asyncio.run(stream_file(wav_path, send_rate))


if __name__ == "__main__":
    main()
