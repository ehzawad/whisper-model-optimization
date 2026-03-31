# Benchmark: serve3.py (CT2) vs serve4.py (HF + Flash Attention 2)

**Date:** 2026-03-29
**GPU:** NVIDIA GeForce RTX 2050 (4GB VRAM)
**GPU_BATCH_SIZE:** 4 (identical on both servers)
**BATCH_TIMEOUT_S:** 0.1 (100ms collection window, identical)

## Servers Under Test

| Server | Backend | Inference | Attention |
|--------|---------|-----------|-----------|
| serve3.py | CTranslate2 (faster-whisper) | `_pipeline.forward()` batched | CT2 internal |
| serve4.py | HF Transformers | `model.generate()` batched | Flash Attention 2 |

Both use the same cross-client batching architecture (100ms collection window,
asyncio.Queue, batch worker). Both have per-audio error isolation, parallel
feature extraction, and model warm-up.

## Test 1: 254 Short Meeting Clips (2-15s each)

Audio: `meeting_22_11_23/*.wav` — 254 files, mixed durations (2-15s), 48kHz WAV

| Server | Total Time | Per Item | Success | Faster? |
|--------|-----------|----------|---------|---------|
| serve3.py (CT2) | 136.55s | 0.54s | 254/254 | — |
| serve4.py (HF+FA2) | 122.82s | 0.48s | 254/254 | **10% faster** |

## Test 2: 1 Long Meeting Recording (947s / 15 min)

Audio: `meeting_22_11_23/single_audio/` — 1 file, 947.3s, 48kHz WAV
Auto-chunked into ~32 chunks of 30s each.

| Server | Total Time | RTF | Success | Faster? |
|--------|-----------|-----|---------|---------|
| serve3.py (CT2) | 40.34s | 0.043 | 1/1 | **29% faster** |
| serve4.py (HF+FA2) | 57.07s | 0.060 | 1/1 | — |

## Test 3: 53 EC Audio Clips (2.8-10.1s each)

Audio: `ec-audio/` — 53 files (13 x 2.8s + 40 x 10.1s), 16kHz WAV

| Server | Total Time | Per Item | Success | Faster? |
|--------|-----------|----------|---------|---------|
| serve3.py (CT2) | 34.74s | 0.66s | 53/53 | — |
| serve4.py (HF+FA2) | 30.92s | 0.58s | 53/53 | **11% faster** |

## Summary

| Scenario | Winner | Margin |
|----------|--------|--------|
| Many short clips (254) | **serve4.py (HF+FA2)** | 10% faster |
| Many short clips (53) | **serve4.py (HF+FA2)** | 11% faster |
| Single long audio (947s) | **serve3.py (CT2)** | 29% faster |

**Conclusion:**
- For **short clips** (the typical production workload): Flash Attention 2 with
  HF Transformers is ~10% faster than CT2 on RTX 2050.
- For **long audio** (auto-chunked): CT2 with int8_float16 quantization is
  significantly faster (~29%), likely because int8 reduces memory pressure and
  CT2's internal chunking is more efficient than our manual chunk+stack approach.
- On **T4 (16GB)** with larger batch sizes, the FA2 advantage on short clips
  should grow further due to O(n) memory scaling.

## Hardware Notes

- RTX 2050 has 4GB VRAM — limits GPU_BATCH_SIZE to 4 for both backends
- CT2 uses int8_float16 (~800MB VRAM), HF uses fp16 (~1.5GB VRAM)
- Flash Attention 2 installed via pre-built wheel:
  `pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl"`
