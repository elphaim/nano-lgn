# nanolgn/lgn.py
"""LogicLayer: one layer of a Differentiable Logic Gate Network.

Each of N output neurons is a softmax-mixture over the 16 binary gates,
applied to two fixed-random inputs (pi_a[j], pi_b[j]) drawn from the previous
layer. Residual init makes the layer ≈ identity-on-pi_a at t=0 so deep stacks
train.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor

from .gates import all_gates_stack, GATE_A_INDEX


class LogicLayer(nn.Module):
    """N -> N width-preserving differentiable logic-gate layer.

    Args:
        n: input/output width.
        seed: deterministic seed for the random connection table.
        residual_init_strength: logit applied to gate "A" (passthrough on a)
            at init. softmax([s, 0, ..., 0])[A] = e^s / (e^s + 15);
            s=7.5 → ≈ 0.9918 (leakage ≈ 0.0082 across the other 15 gates).
    """

    def __init__(self, n: int, seed: int, residual_init_strength: float = 7.5):
        super().__init__()
        self.n = n
        self.residual_init_strength = float(residual_init_strength)

        # Fixed random connections, deterministic in seed.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pi_a = torch.randint(0, n, (n,), generator=gen, dtype=torch.long)
        pi_b = torch.randint(0, n, (n,), generator=gen, dtype=torch.long)
        self.register_buffer("pi_a", pi_a)
        self.register_buffer("pi_b", pi_b)

        # Gate-mixture logits, one row per output neuron. Residual init: gate
        # "A" gets a strong logit; the rest stay at 0.
        W = torch.zeros(n, 16)
        W[:, GATE_A_INDEX] = self.residual_init_strength
        self.W = nn.Parameter(W)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., n) in [0,1]. Returns (..., n) in [0,1]."""
        a = x.index_select(-1, self.pi_a)        # (..., n)
        b = x.index_select(-1, self.pi_b)        # (..., n)
        gates = all_gates_stack(a, b)            # (..., n, 16)
        p = torch.softmax(self.W, dim=-1)        # (n, 16)
        # Broadcast p across leading dims; sum over the gate dim.
        out = (gates * p).sum(dim=-1)            # (..., n)
        return out
