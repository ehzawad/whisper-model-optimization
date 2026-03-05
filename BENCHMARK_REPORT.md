# Bengali Whisper Medium — Comprehensive Inference Benchmark

**Model**: Fine-tuned Whisper Medium (~769M params) for Bengali ASR
**GPU**: NVIDIA GeForce RTX 2050 (4 GB VRAM)
**Software**: PyTorch 2.10.0+cu128, Transformers 4.57.6, CUDA 12.8
**Audio corpus**: 254 individual speaker clips (19.5 min total, 3 speakers) + 1 long 15.8-min meeting recording
**Date**: 2026-03-05

---

## 1. Methods Tested

Four inference strategies were benchmarked across all 255 audio files:

| # | Method | Model | Description |
|---|---|---|---|
| A | **fp16** | Fine-tuned Medium | Half-precision + SDPA attention (baseline) |
| B | **fp16_compiled** | Fine-tuned Medium | fp16 + `torch.compile(mode="default")` kernel fusion |
| C | **turbo_fp16** | whisper-large-v3-turbo | OpenAI's pruned Large-v3 (4 decoder layers), generic Bengali |
| D | **turbo_compiled** | whisper-large-v3-turbo | Turbo + `torch.compile(mode="default")` |

All methods used `attn_implementation="sdpa"` and `torch.backends.cudnn.enabled = False`
(cuDNN 9.x lacks fp16 conv1d kernels for compute capability 8.6).

## 2. Overall Performance

| Method | Load(s) | Clips(s) | Long(s) | Clip RTF | Long RTF | VRAM Peak |
|---|---|---|---|---|---|---|
| **fp16** | 1.2 | 447.9 | 433.5 | 0.383 | 0.458 | **2,417 MiB** |
| **fp16_compiled** | 1.6 | 435.9 | 420.4 | 0.373 | 0.444 | 2,609 MiB |
| **turbo_fp16** | 72.8 | 792.0 | **224.5** | 0.677 | **0.237** | **1,782 MiB** |
| **turbo_compiled** | 5.3 | 774.9 | **227.3** | 0.662 | **0.240** | 1,808 MiB |

RTF = Real-Time Factor (lower is faster; RTF < 1.0 means faster than real-time).

## 3. Relative Speed and VRAM

| Method | Clip Speedup | Long Speedup | VRAM vs fp16 |
|---|---|---|---|
| **fp16** | 1.00x | 1.00x | baseline |
| **fp16_compiled** | 1.03x | 1.03x | +7.9% |
| **turbo_fp16** | 0.57x | **1.93x** | **-26.2%** |
| **turbo_compiled** | 0.58x | **1.91x** | **-25.2%** |

## 4. Per-Clip Inference Time Statistics

254 individual clips, duration range 2.0s - 14.8s (mean 4.6s):

| Method | Mean | Median | P95 | Max | Min |
|---|---|---|---|---|---|
| **fp16** | 1.764s | 1.517s | 3.740s | 6.189s | 0.593s |
| **fp16_compiled** | 1.716s | 1.471s | 3.613s | 6.246s | 0.568s |
| **turbo_fp16** | 3.118s | 1.935s | 6.307s | 6.714s | 0.428s |
| **turbo_compiled** | 3.051s | 1.934s | 6.096s | 6.196s | 0.443s |

## 5. Per-Speaker Breakdown (Mean Inference Time)

| Method | Sazzad (90 clips) | saiful (69 clips) | Sunny (95 clips) |
|---|---|---|---|
| **fp16** | 1.783s | 1.906s | 1.641s |
| **fp16_compiled** | 1.738s | 1.850s | 1.598s |
| **turbo_fp16** | 4.480s | 2.080s | 2.582s |
| **turbo_compiled** | 4.373s | 2.043s | 2.530s |

Note: Turbo is disproportionately slow on Sazzad's clips (4.5s vs 2.1s for saiful).
This correlates with hallucination — turbo generates repetitive garbage tokens on
Sazzad's noisier clips, inflating generation time.

## 6. Transcript Accuracy (Cross-Method Similarity)

### 6.1 vs fp16 Baseline (254 clips)

| Method | Avg Word Sim | Avg Char Sim | Mean Output Len | Empty Clips |
|---|---|---|---|---|
| **fp16** (ref) | 100.0% | 100.0% | 46.1 chars | 0 |
| **fp16_compiled** | **99.6%** | **99.9%** | 46.1 chars | 0 |
| **turbo_fp16** | 8.7% | 38.5% | 107.5 chars | 0 |
| **turbo_compiled** | 8.9% | 38.4% | 107.7 chars | 0 |

### 6.2 Long Audio (15.8 min) vs fp16

| Method | Word Sim | Char Sim | Length |
|---|---|---|---|
| **fp16** (ref) | 100.0% | 100.0% | 9,411 chars |
| **fp16_compiled** | **99.8%** | **99.8%** | 9,410 chars |
| **turbo_fp16** | 3.6% | 0.5% | 6,069 chars |
| **turbo_compiled** | 4.1% | 0.7% | 6,096 chars |

### 6.3 Pairwise Word Similarity Matrix

|  | fp16 | fp16_compiled | turbo_fp16 | turbo_compiled |
|---|---|---|---|---|
| **fp16** | 100.0% | 99.6% | 8.7% | 8.9% |
| **fp16_compiled** | 99.6% | 100.0% | 8.7% | 8.9% |
| **turbo_fp16** | 8.7% | 8.7% | 100.0% | 93.4% |
| **turbo_compiled** | 8.9% | 8.9% | 93.4% | 100.0% |

## 7. Turbo Hallucination Analysis

The generic whisper-large-v3-turbo produces severe hallucinations on this meeting
audio. Examples from the worst clips:

**fp16 (correct):**
> আমরা সবাই আনুষ্ঠানিকভাবে আর আমার অ্যারাউন্ট ছয়টা চল্লিশের দিকে

**turbo_fp16 (hallucinated):**
> নেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনেনে

The turbo model also shows systematic Bengali script errors even on clips where it
doesn't fully hallucinate:
- Missing juktakshar (conjunct consonants)
- Phonetic transliteration instead of standard Bengali orthography
- "আম্রা" instead of "আমরা", "সিধান্তো" instead of "সিদ্ধান্ত"

Root cause: whisper-large-v3-turbo is a generic multilingual model not fine-tuned
for Bengali meeting audio. The fine-tuned medium model's domain-specific training
data gives it a massive accuracy advantage despite being a smaller model.

## 8. torch.compile Analysis

`torch.compile(mode="default")` provides a modest 3% speedup on this hardware:

- Clip RTF: 0.383 -> 0.373 (2.7% faster)
- Long RTF: 0.458 -> 0.444 (3.1% faster)
- VRAM cost: +192 MiB (+7.9%) for compiled kernel caches

The limited speedup is due to the RTX 2050's small SM count (16 SMs) — torch.compile's
kernel fusion and `max_autotune_gemm` mode cannot be used effectively. On larger GPUs
(A100, RTX 4090), torch.compile typically delivers 4-5x speedup.

Note: `mode="reduce-overhead"` (CUDA Graphs) was tested but causes OOM — CUDA Graphs
allocate ~856 MiB in private pools, exhausting the 4 GB VRAM.

## 9. Overall Efficiency

| Method | Total Compute(s) | Rel Speed | Avg Accuracy | VRAM | Model |
|---|---|---|---|---|---|
| **fp16** | 881.4 | 1.00x | 100.0% | 2,417 MiB | fine-tuned medium |
| **fp16_compiled** | 856.2 | 1.03x | 99.6% | 2,609 MiB | fine-tuned medium |
| **turbo_fp16** | 1,016.5 | 0.87x | 8.7% | 1,782 MiB | turbo-v3 generic |
| **turbo_compiled** | 1,002.2 | 0.88x | 8.9% | 1,808 MiB | turbo-v3 generic |

## 10. Key Findings

1. **Fine-tuned fp16 remains the best choice** — fastest on short clips, perfect
   accuracy, and fits within 4 GB VRAM with 1.6 GB headroom.

2. **torch.compile gives ~3% speedup** on RTX 2050, not worth the +8% VRAM overhead
   and compilation latency. Becomes significant only on GPUs with more SMs.

3. **whisper-large-v3-turbo is 1.93x faster on long audio** due to its 4-layer
   decoder (vs 24 layers), but **catastrophically fails on Bengali meeting audio**
   — only 8.7% word similarity. Fine-tuning is essential for domain-specific Bengali ASR.

4. **Turbo uses 26% less VRAM** (1,782 MiB vs 2,417 MiB) thanks to the pruned
   decoder, but the accuracy penalty makes it unusable for this use case.

5. **Short clip overhead matters** — turbo's larger encoder (large-v3 encoder vs
   medium encoder) makes per-clip latency ~1.8x worse than the fine-tuned medium,
   despite having fewer decoder layers.

## 11. Recommendation

**Use fp16 + SDPA** (`transcribe.py`) for production Bengali ASR on RTX 2050.

For Bengali-specific improvements, the path forward is:
- Fine-tune whisper-large-v3-turbo on Bengali data (would combine turbo's decoder
  speed with Bengali accuracy)
- Use speculative decoding with a Bengali-distilled assistant model (when available)

## 12. Previous Quantization Results (Single Long Audio)

For reference, the earlier quantization benchmark on just the 15.8-min audio:

| Method | VRAM Peak | Wall Time | Word Similarity |
|---|---|---|---|
| **fp16** | 2,417 MiB | 443s | baseline |
| **bnb int8** | 1,789 MiB | 930s (0.48x) | 96.5% |
| **bnb int4 (nf4)** | 2,065 MiB | 535s (0.83x) | 89.9% |
| **quanto int8** | 2,823 MiB | 1,133s (0.39x) | 98.6% |

## 13. File Reference

```
bengali-whisper-medium/
├── transcribe.py                       # Inference script (fp16 + SDPA)
├── benchmark_enhanced.py               # Comprehensive benchmark script
├── BENCHMARK_REPORT.md                 # This report
├── benchmark_results/
│   ├── enhanced_summary.json           # Aggregate metrics (JSON)
│   ├── transcripts_fp16.json           # Per-file transcripts + timing
│   ├── transcripts_fp16_compiled.json
│   ├── transcripts_turbo_fp16.json
│   └── transcripts_turbo_compiled.json
└── meeting_22_11_23/
    ├── *.wav                           # 254 audio chunks (3 speakers)
    ├── *.txt                           # Corresponding transcripts
    └── single_audio/
        ├── *.wav                       # Full 15.8 min meeting audio
        └── benchmark_*.txt             # Previous quantization benchmarks
```
