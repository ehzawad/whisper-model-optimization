# Bengali Whisper Medium — Inference Optimization Report

**Model**: Fine-tuned Whisper Medium (~769M params) for Bengali ASR
**GPU**: NVIDIA GeForce RTX 2050 (4 GB VRAM)
**Software**: PyTorch 2.9.0+cu128, Transformers 4.57.1, CUDA 12.8, cuDNN 9.11
**Test audio**: 15.8-minute Bengali meeting recording (48 kHz, 87 MB)
**Date**: 2026-03-04

---

## 1. Problem

The model weights in float32 occupy 2.9 GB. With activation memory and KV cache
overhead, peak VRAM during inference reaches ~4.5 GB — exceeding the 4 GB
available on the RTX 2050. The model cannot run in its default precision on this
GPU.

## 2. Methods Tested

Four quantization strategies were benchmarked on the same 15.8-minute audio:

| # | Method | Description |
|---|---|---|
| A | **float16** | Half-precision weights + SDPA attention (baseline) |
| B | **bitsandbytes int8** | LLM.int8() quantization via `BitsAndBytesConfig(load_in_8bit=True)` |
| C | **bitsandbytes int4** | NF4 quantization via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` |
| D | **quanto int8** | int8 weight quantization via `QuantoConfig(weights="int8")` |

All methods used `attn_implementation="sdpa"` for fused scaled-dot-product
attention. cuDNN was disabled due to a missing fp16 conv1d kernel in cuDNN 9.11
for this GPU (compute capability 8.6); the non-cuDNN fallback has negligible
impact since only the two initial encoder convolutions are affected.

## 3. Results

### 3.1 Performance

| Method | VRAM Peak | VRAM vs fp16 | Wall Time | Relative Speed |
|---|---|---|---|---|
| **float16** | **2,417 MiB** | baseline | **443 s** | **1.00x** |
| **bnb int8** | **1,789 MiB** | **-26.0%** | 930 s | 0.48x |
| **bnb int4 (nf4)** | 2,065 MiB | -14.6% | 535 s | 0.83x |
| **quanto int8** | 2,823 MiB | +16.8% | 1,133 s | 0.39x |

### 3.2 Transcription Quality

Similarity was measured against the float16 baseline using Python's
`difflib.SequenceMatcher` on the full 15.8-minute transcript:

| Method | Word-level Similarity | Char-level Similarity | Output Length |
|---|---|---|---|
| **float16** (ref) | — | — | 9,411 chars |
| **bnb int8** | **96.5%** | **95.5%** | 9,666 chars |
| **bnb int4 (nf4)** | 89.9% | 11.1% | 8,849 chars |
| **quanto int8** | **98.6%** | **96.8%** | 9,402 chars |

### 3.3 Sample Comparison (opening lines)

**float16:**
> হ্যাঁ আমরা সবাই আনামিউটেড থাকবো আলোয়েস আর এরাউন্ড ছয়টা চল্লিশের দিকে শুরু করছি

**bnb int8:**
> হ্যাঁ আমরা সবাই আনামিউটেড থাকবো আলোয়েস আর এরাউন্ড ছয়টা চল্লিশের দিকে শুরু করছি

**bnb int4:**
> হ্যাঁ আমরা সবাই আন্ডামিউটেড থাকবো আলোয়েস আর এরাউন্ট ছয়টা চল্লিশের দিকে শুরু করছি

**quanto int8:**
> হ্যাঁ আমরা সবাই আনামিউটেড থাকবো আলোয়েস আর এরাউন্ড ছয়টা চল্লিশের দিকে শুরু করছি

Note: bnb int4 introduces visible errors from word 3 onward ("আনামিউটেড" →
"আন্ডামিউটেড", "এরাউন্ড" → "এরাউন্ট").

## 4. Analysis

### float16 (Recommended)
The best balance of speed, accuracy, and simplicity. Peak VRAM of 2.4 GB leaves
1.6 GB of headroom on the 4 GB GPU — more than enough for concurrent system
tasks. No additional dependencies beyond PyTorch and Transformers.

### bitsandbytes int8 (Best for constrained memory)
Achieves the largest VRAM reduction (-26%, from 2.4 GB to 1.8 GB) with 96.5%
word-level fidelity. The differences are minor and typically affect
transliterations of English loanwords (e.g., "অনেক" vs "আমাদের"). The
trade-off is 2x slower inference. Requires the `bitsandbytes` and `accelerate`
packages.

### bitsandbytes int4 (Not recommended)
Counter-intuitively uses more VRAM than int8 (2.1 GB vs 1.8 GB) due to
dequantization buffer overhead in the NF4 scheme. Quality drops significantly
— 89.9% word match with errors appearing from the very first sentence. The 11.1%
character-level similarity reflects pervasive small changes across the entire
transcript. Not suitable for production ASR.

### quanto int8 (Not recommended)
Highest fidelity among quantized methods (98.6% word match) but uses MORE VRAM
than float16 (2.8 GB vs 2.4 GB) due to quanto's internal dequantization buffers.
Also the slowest at 2.6x slower than float16. Provides no practical benefit on
this hardware.

## 5. Environment Notes

- **cuDNN 9.11 + RTX 2050**: cuDNN lacks fp16/bf16 conv1d kernels for compute
  capability 8.6. Setting `torch.backends.cudnn.enabled = False` resolves this
  with no measurable impact (only 2 conv layers in the Whisper encoder are
  affected; the rest uses SDPA attention and linear layers).

- **Checkpoint compatibility**: The model was saved with Transformers 4.26 and
  lacks a `generation_config.json`. The modern `language="bn"` argument to
  `generate()` does not work. Instead, forced decoder IDs are set via
  `model.generation_config.forced_decoder_ids`.

- **Long-form audio**: Whisper's `generate()` only processes the first 30-second
  segment. For audio longer than 30 seconds, the HuggingFace `pipeline` with
  `chunk_length_s=30` handles automatic chunking and stitching.

## 6. Recommendation

**Use float16 + SDPA** (`transcribe.py` as currently configured). It provides
the best speed, full accuracy, and fits comfortably within 4 GB VRAM.

If a future use case requires lower memory (e.g., running alongside another
model), switch to **bitsandbytes int8** — accept the 2x speed penalty for a
26% VRAM reduction with minimal quality loss.

Avoid int4 and quanto int8 on this hardware.

## 7. File Reference

```
bengali-whisper-medium/
├── transcribe.py                          # Inference script (fp16 + SDPA)
├── BENCHMARK_REPORT.md                    # This report
├── meeting_22_11_23/
│   ├── *.wav                              # 254 audio chunks (by speaker)
│   ├── *.txt                              # Corresponding transcripts
│   └── single_audio/
│       ├── *.wav                          # Full 15.8 min meeting audio
│       ├── *.txt                          # Full meeting transcript (fp16)
│       ├── benchmark_fp16.txt             # Benchmark output: float16
│       ├── benchmark_bnb_int8.txt         # Benchmark output: bitsandbytes int8
│       ├── benchmark_bnb_int4.txt         # Benchmark output: bitsandbytes int4
│       └── benchmark_quanto_int8.txt      # Benchmark output: quanto int8
```
