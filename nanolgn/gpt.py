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
