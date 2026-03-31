# Benchmark: INT8 Quantization Comparison (bitsandbytes vs CTranslate2)

**Date:** 2026-03-29
**GPU:** NVIDIA GeForce RTX 2050 (4GB VRAM)
**GPU_BATCH_SIZE:** 4 (identical across all servers)

## Question

serve3_resilient.py uses CT2 with int8_float16 and is 29% faster than serve4.py
(fp16) on long audio. Can we close that gap by switching serve4.py to int8
via bitsandbytes `load_in_8bit=True`?

**Answer: No. bitsandbytes int8 is 2-3x SLOWER on RTX 2050.**

## Configurations Tested

| Config | Backend | Quantization | Attention |
|--------|---------|-------------|-----------|
| serve3_resilient.py | CTranslate2 | int8_float16 (CT2 native) | CT2 internal |
| serve4.py (fp16) | HF Transformers | fp16 | Flash Attention 2 |
| serve4.py (int8) | HF Transformers | int8 (bitsandbytes) | Flash Attention 2 |

## Test 1: 254 Short Meeting Clips

| Config | Total | Per Item | vs CT2 baseline |
|--------|-------|----------|-----------------|
| serve3_resilient (CT2 int8_fp16) | **136.55s** | **0.54s** | baseline |
| serve4 fp16+FA2 | **122.82s** | **0.48s** | 10% faster |
| serve4 int8+FA2 (bitsandbytes) | **285.03s** | **1.12s** | 109% slower |

## Test 2: 1 Long Meeting Recording (947s)

| Config | Total | RTF | vs CT2 baseline |
|--------|-------|-----|-----------------|
| serve3_resilient (CT2 int8_fp16) | **40.34s** | **0.043** | baseline |
| serve4 fp16+FA2 | **57.07s** | **0.060** | 41% slower |
| serve4 int8+FA2 (bitsandbytes) | **127.05s** | **0.134** | 215% slower |

## Test 3: 53 EC Audio Clips

| Config | Total | Per Item |
|--------|-------|----------|
| serve3_resilient (CT2 int8_fp16) | 34.74s | 0.66s |
| serve4 fp16+FA2 | 30.92s | 0.58s |
| serve4 int8+FA2 (bitsandbytes) | 72.57s | 1.37s |

## Why bitsandbytes INT8 is Slow on RTX 2050

CT2's int8_float16 and bitsandbytes `load_in_8bit` are fundamentally different:

- **CTranslate2 int8_float16**: Weights stored as int8, computation in fp16.
  Dequantization happens in fused C++ CUDA kernels with minimal overhead.
  Optimized at the operator level — the entire inference graph is compiled.

- **bitsandbytes int8**: Weights stored as int8 via LLM.int8() algorithm.
  Each linear layer dequantizes weights to fp16 at runtime in Python/PyTorch.
  Dequantization overhead per layer per forward pass adds up significantly.
  On small GPUs (RTX 2050, 16 SMs), the overhead dominates the savings.

bitsandbytes int8 is designed for **fitting large models into limited VRAM**
(e.g., running a 7B model on a 24GB GPU). It trades speed for memory. On
Whisper Medium (~1.5GB fp16), there's no memory pressure to solve — the model
already fits in fp16. So int8 only adds overhead with no benefit.

## Conclusion

| Scenario | Best Config | Why |
|----------|------------|-----|
| Short clips (production) | **serve4 fp16+FA2** | FA2 attention speed wins |
| Long audio (>30s) | **serve3_resilient CT2 int8_fp16** | Native int8 + C++ chunking |
| Never use | serve4 int8 (bitsandbytes) | 2-3x slower, no benefit on this model size |

**Recommendation:** Keep serve4.py at fp16 (revert the int8 change). The fp16+FA2
config is the best HF Transformers option. For long audio on RTX 2050, CT2 wins.
On T4 (16GB) with larger batch sizes, fp16+FA2 may close the long-audio gap.
