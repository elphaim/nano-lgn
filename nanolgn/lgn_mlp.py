# nanolgn/lgn_mlp.py
"""LGN-MLP block: ThermometerEncode -> LGNBody -> GroupSumDecode.

Drop-in replacement for the per-block ReLU² FFN.
"""
from __future__ import annotations
import math
import torch
from torch import nn, Tensor

from .lgn import LGNBody


def _inv_sigmoid(p: float) -> float:
    return math.log(p / (1.0 - p))


class ThermometerEncode(nn.Module):
    """(B, T, d) -> (B, T, K*d) in (0, 1).

    For each scalar feature x_i and level k:
        b_{i,k} = sigmoid(s_k * (x_i - theta_{i,k}))

    Params:
        theta ∈ ℝ^(d, K): per-feature, per-level threshold.
        s ∈ ℝ^K: per-level sharpness (shared across features).
    """

    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.k = k

        # Init thresholds: same K levels for every feature, spread on the
        # input axis via inverse-sigmoid of evenly-spaced probabilities.
        levels = torch.tensor(
            [_inv_sigmoid((i + 1) / (k + 1)) for i in range(k)]
        )                                              # (K,)
        theta = levels.unsqueeze(0).expand(d_model, k).clone()  # (d, K)
        self.theta = nn.Parameter(theta)

        # Sharpness shared across features, init = 1.
        self.s = nn.Parameter(torch.ones(k))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d). Broadcast to (..., d, K), then flatten last two dims.
        x_e = x.unsqueeze(-1)                          # (..., d, 1)
        # theta: (d, K) → broadcasts cleanly.
        # s: (K,) → broadcasts cleanly.
        b = torch.sigmoid(self.s * (x_e - self.theta))  # (..., d, K)
        return b.flatten(-2)                           # (..., d*K)


class GroupSumDecode(nn.Module):
    """(B, T, K*d) -> (B, T, d) via contiguous group-sum, scaled and centered.

    Reshape last dim as (d, K), sum over K, divide by tau, subtract 0.5.
    Default tau=K → outputs in [-0.5, 0.5].
    """

    def __init__(self, d_model: int, k: int, tau: float):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.tau = float(tau)

    def forward(self, z: Tensor) -> Tensor:
        # z: (..., d*K)
        leading = z.shape[:-1]
        z2 = z.reshape(*leading, self.d_model, self.k)  # (..., d, K)
        return z2.sum(dim=-1) / self.tau - 0.5
