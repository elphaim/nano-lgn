"""Minimal nanochat-style GPT with a pluggable FFN slot."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with NO learnable parameters (nanochat style)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d). Normalize so RMS over last dim == 1.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


def precompute_rope(dim: int, max_len: int, base: float = 10000.0):
    """Precompute cos/sin tables for RoPE. Returns (cos, sin) of shape (max_len, dim/2)."""
    assert dim % 2 == 0, "RoPE dim must be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # (max_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x: (..., T, dim). cos, sin: (T, dim/2)."""
    # Split into pairs: (..., T, dim) → (..., T, half, 2)
    leading = x.shape[:-1]
    x2 = x.reshape(*leading[:-1], leading[-1], -1, 2)
    x_e = x2[..., 0]
    x_o = x2[..., 1]
    # Broadcast cos/sin: (T, half) over (..., T, half).
    out_e = x_e * cos - x_o * sin
    out_o = x_e * sin + x_o * cos
    out = torch.stack((out_e, out_o), dim=-1)
    return out.reshape(*leading, -1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, no biases.

    Uses F.scaled_dot_product_attention with is_causal=True (Flash kernel
    when available).
    """

    def __init__(self, d_model: int, n_head: int, ctx_len: int, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.ctx_len = ctx_len
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        cos, sin = precompute_rope(self.head_dim, ctx_len, base=rope_base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        assert T <= self.ctx_len, f"seq len {T} exceeds ctx_len {self.ctx_len}"
        qkv = self.qkv(x)                                   # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, T, D) → (B, H, T, Dh)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)          # (B, T, D)
        return self.proj(out)


class ReLU2MLP(nn.Module):
    """Standard nanochat MLP: Linear(d, m*d) -> ReLU² -> Linear(m*d, d).

    No biases, no learnable scale.
    """

    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(d_model, mult * d_model, bias=False)
        self.down = nn.Linear(mult * d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.up(x)
        h = F.relu(h).square()
        return self.down(h)
