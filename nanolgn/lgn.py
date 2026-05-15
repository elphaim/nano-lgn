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

from .gates import GATE_A_INDEX, GATE_COEFFS


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

        # Multilinear coefficients per gate (α, β, γ, δ); see gates.GATE_COEFFS.
        self.register_buffer("gate_coeffs", torch.tensor(GATE_COEFFS, dtype=torch.float32))

        # Gate-mixture logits, one row per output neuron. Residual init: gate
        # "A" gets a strong logit; the rest stay at 0.
        W = torch.zeros(n, 16)
        W[:, GATE_A_INDEX] = self.residual_init_strength
        self.W = nn.Parameter(W)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., n) in [0,1]. Returns (..., n) in [0,1].

        Polynomial form: Σ_g p_g · gate_g(a, b) = α + β·a + γ·b + δ·a·b, where
        (α, β, γ, δ) = p @ GATE_COEFFS. Mathematically identical to
        (all_gates_stack(a, b) * p).sum(-1), but never materializes the
        (..., n, 16) stack — the dominant activation cost on the old path.
        """
        a = x.index_select(-1, self.pi_a)                    # (..., n)
        b = x.index_select(-1, self.pi_b)                    # (..., n)
        p = torch.softmax(self.W, dim=-1)                    # (n, 16)
        coeffs = p @ self.gate_coeffs                        # (n, 4)
        alpha, beta, gamma, delta = coeffs.unbind(dim=-1)    # each (n,)
        return alpha + beta * a + gamma * b + delta * (a * b)


class LGNBody(nn.Module):
    """Stack of L width-preserving LogicLayers.

    Each layer has its own connection table, deterministically derived from
    the body's seed (layer i uses seed = base_seed * 1_000_003 + i).
    """

    def __init__(
        self,
        n: int,
        depth: int,
        seed: int,
        residual_init_strength: float = 7.5,
    ):
        super().__init__()
        self.n = n
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                LogicLayer(
                    n=n,
                    seed=seed * 1_000_003 + i,
                    residual_init_strength=residual_init_strength,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
