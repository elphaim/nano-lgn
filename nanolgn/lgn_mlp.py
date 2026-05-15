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


class LGNMLPBlock(nn.Module):
    """Drop-in replacement for the per-block FFN: continuous → LGN → continuous.

    Same forward shape contract as a standard MLP: (B, T, d) -> (B, T, d).
    The caller (Block) does the residual add.
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        depth: int,
        tau: float,
        seed: int,
        residual_init_strength: float = 7.5,
    ):
        super().__init__()
        self.encode = ThermometerEncode(d_model=d_model, k=k)
        self.body = LGNBody(
            n=k * d_model,
            depth=depth,
            seed=seed,
            residual_init_strength=residual_init_strength,
        )
        self.decode = GroupSumDecode(d_model=d_model, k=k, tau=tau)

    def forward(self, x: Tensor) -> Tensor:
        b = self.encode(x)         # (B, T, K*d) in (0,1)
        z = self.body(b)           # (B, T, K*d) in (0,1)
        y = self.decode(z)         # (B, T, d)   in [-0.5, 0.5]
        return y


@torch.no_grad()
def lgn_block_stats(block: "LGNMLPBlock", last_ffn_out: Tensor) -> dict:
    """Collect early-warning diagnostics for one LGN-MLP block.

    Args:
        block: the LGNMLPBlock instance.
        last_ffn_out: the (B, T, d) tensor returned by the most recent forward.

    Returns dict with keys:
        ffn_out_norm_mean : float — mean L2 norm per token
        ffn_out_max       : float — max abs value
        gate_entropy_mean : float — mean entropy of softmax(W) across all neurons (nats)
        threshold_in_range_frac : float — fraction of (theta_{i,k}) that lie within
            ±3 of zero (i.e. plausibly active for typical RMS-normed inputs).
    """
    out = last_ffn_out
    norm = out.flatten(0, -2).norm(dim=-1).mean().item()
    omax = out.abs().max().item()

    entropies = []
    for layer in block.body.layers:
        p = torch.softmax(layer.W, dim=-1)
        h = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)   # (n,)
        entropies.append(h.mean().item())
    gate_entropy_mean = sum(entropies) / max(1, len(entropies))

    theta = block.encode.theta
    in_range = ((theta.abs() <= 3.0).float()).mean().item()

    return {
        "ffn_out_norm_mean": norm,
        "ffn_out_max": omax,
        "gate_entropy_mean": gate_entropy_mean,
        "threshold_in_range_frac": in_range,
    }
