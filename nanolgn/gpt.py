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
