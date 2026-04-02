# Triton Kernel Experiments — RTX 2050 (4GB, sm86)

## What was tried

Custom Triton GPU kernels to accelerate Whisper inference by fusing
operations that make separate memory round-trips in the standard HF
Transformers pipeline.

### Kernels written (`triton_kernels.py`)

| Kernel | What it fuses | Microbenchmark |
|---|---|---|
| `fused_residual_layernorm` | residual add + LayerNorm in one pass | 1.29x (0.77ms -> 0.60ms) |
| `fused_residual_add_clamp` | residual add + fp16 overflow clamp | 1.69x (0.75ms -> 0.45ms) |
| `triton_gelu` | standalone GELU activation | 1.02x (negligible) |

Microbenchmarks on encoder-sized tensors `[4, 1500, 1024]` fp16.

### Integration

Monkey-patched all 48 WhisperEncoderLayer + WhisperDecoderLayer forward
methods to use fused kernels, skip eval-mode dropout no-ops, and use
cached weight references.

## End-to-end results (50 files, 759s audio)

Fair comparison at the same batch size, 3 runs each, best of 3:

```
bs=4:  Baseline=23.84s  Triton=23.23s  +2.5%
bs=5:  Baseline=22.25s  Triton=21.72s  +2.4%
bs=6:  Baseline=21.60s  Triton=21.79s  -0.9%
bs=7:  Baseline=20.99s  Triton=21.44s  -2.1%
bs=8:  Baseline=20.70s  Triton=21.05s  -1.7%
```

Triton kernels give ~2.5% at bs=4-5. At larger batch sizes the
monkey-patching overhead (Python function dispatch, `.contiguous()`
copies from FA2 output views) exceeds the bandwidth savings.

## Other things tried (all negative on RTX 2050)

| Approach | Result | Why |
|---|---|---|
| `torch.compile(decoder, mode="default")` | -353% | Inductor compilation overhead, "not enough SMs" |
| `torch.compile(model, mode="reduce-overhead")` | -53% | CUDA graph overhead, 16 SMs too few |
| `bitsandbytes INT8 (load_in_8bit)` | -159% | Mixed-precision dequantization overhead per matmul |
| Static KV cache (`cache_implementation="static"`) | -233% | Pre-allocation for max sequence length wastes compute |
| Custom decode loop (bypass `model.generate()`) | +1.1% | Python overhead in generate() is already tiny |

## Why the improvement ceiling is ~2.5%

1. **Decoder is 88% of wall time** — encoder (where Triton kernels shine
   on [4, 1500, 1024] tensors) is only 12%.
2. **Decoder operates on [bs, 1, 1024] tensors** — too small for memory
   bandwidth savings to matter. Kernel launch overhead dominates.
3. **Weight loading is the bottleneck** — each decoder step loads ~768MB
   of weights through 96 GB/s bus = 8ms theoretical floor. Fusing the
   cheap element-wise ops around the matmuls can't reduce this.
4. **FA2 output views require `.contiguous()` copies** — partially
   negates the bandwidth savings from fusion.

## What would actually move the needle

- Quantized matmul kernels (INT4/INT8 weight-only quantization with
  efficient dequantization, not bitsandbytes)
- Speculative decoding (draft with a smaller model)
- Smaller/distilled model (fewer decoder layers)
- Hardware with more memory bandwidth
