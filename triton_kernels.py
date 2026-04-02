"""Custom Triton GPU kernels for Whisper inference acceleration.

Fused operations that reduce global memory round-trips on bandwidth-limited
GPUs (RTX 2050: 96 GB/s). Each kernel eliminates intermediate tensor
materializations that standard PyTorch emits as separate kernel launches.

Kernels:
    fused_residual_layernorm  — residual add + LayerNorm in one pass
    fused_residual_add_clamp  — residual add + fp16 clamp (encoder epilogue)
    triton_gelu               — GELU activation (standalone fused kernel)
"""

import torch
import triton
import triton.language as tl


# ── Fused Residual + LayerNorm ──────────────────────────────────────────────
#
# Replaces:  hidden = residual + hidden; residual = hidden; hidden = LayerNorm(hidden)
# With:      residual_out, normed_out = fused_residual_layernorm(hidden, residual, w, b)
#
# Saves one full read+write of [B, S, D] tensor per invocation.
# Used ~48 times per forward pass (encoder + decoder layer boundaries).

@triton.jit
def _fused_residual_layernorm_kernel(
    X_ptr,            # attention/FFN output [N, D]
    Residual_ptr,     # residual input [N, D]
    Weight_ptr,       # LayerNorm gamma [D]
    Bias_ptr,         # LayerNorm beta [D]
    Out_ptr,          # normalized output [N, D]
    ResOut_ptr,       # new residual = X + Residual [N, D]
    stride_x: tl.constexpr,
    stride_out: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, D)

    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_x + cols).to(tl.float32)

    # Residual add
    added = x + r

    # Store new residual (for next sub-block)
    tl.store(ResOut_ptr + row * stride_out + cols, added.to(tl.float16))

    # LayerNorm
    mean = tl.sum(added, axis=0) / D
    centered = added - mean
    var = tl.sum(centered * centered, axis=0) / D
    inv_std = tl.rsqrt(var + eps)
    normed = centered * inv_std

    # Affine transform
    w = tl.load(Weight_ptr + cols).to(tl.float32)
    b = tl.load(Bias_ptr + cols).to(tl.float32)
    out = normed * w + b

    tl.store(Out_ptr + row * stride_out + cols, out.to(tl.float16))


def fused_residual_layernorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + LayerNorm.

    Returns (normed_output, new_residual) where new_residual = hidden_states + residual.
    """
    hidden_states = hidden_states.contiguous()
    residual = residual.contiguous()
    shape = hidden_states.shape
    N = hidden_states.numel() // shape[-1]
    D = shape[-1]

    out = torch.empty_like(hidden_states)
    res_out = torch.empty_like(hidden_states)

    _fused_residual_layernorm_kernel[(N,)](
        hidden_states, residual, weight, bias, out, res_out,
        stride_x=D, stride_out=D, D=D, eps=eps,
    )
    return out, res_out


# ── Fused Residual Add + Clamp (Encoder Epilogue) ──────────────────────────
#
# Replaces:  hidden = residual + hidden; hidden = clamp(hidden, -max, max)
# The fp16 clamp is needed to prevent overflow in encoder layers.

@triton.jit
def _fused_residual_add_clamp_kernel(
    X_ptr,
    Residual_ptr,
    Out_ptr,
    n_elements,
    clamp_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    r = tl.load(Residual_ptr + offsets, mask=mask).to(tl.float32)
    out = x + r
    out = tl.minimum(tl.maximum(out, -clamp_val), clamp_val)
    tl.store(Out_ptr + offsets, out.to(tl.float16), mask=mask)


def fused_residual_add_clamp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Fused residual add + fp16 overflow clamp."""
    hidden_states = hidden_states.contiguous()
    residual = residual.contiguous()
    n = hidden_states.numel()
    out = torch.empty_like(hidden_states)
    clamp_val = torch.finfo(torch.float16).max - 1000

    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fused_residual_add_clamp_kernel[grid](
        hidden_states, residual, out, n, clamp_val, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ── Fused GELU ─────────────────────────────────────────────────────────────
#
# Standalone GELU kernel. Marginal benefit over torch.compile's auto-fusion
# but useful when torch.compile is not applied (e.g., decoder path).

@triton.jit
def _gelu_kernel(
    X_ptr,
    Out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    out = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Out_ptr + offsets, out.to(tl.float16), mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation via Triton kernel."""
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
