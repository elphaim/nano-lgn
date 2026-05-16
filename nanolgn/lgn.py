# nanolgn/lgn.py
"""LogicLayer: one layer of a Differentiable Logic Gate Network.

Each of N output neurons is a softmax-mixture over the 16 binary gates,
applied to a pair of inputs (a, b) chosen by the layer's interconnect:
fixed-random buffers (pi_a[j], pi_b[j]) by default, or a learnable Top-K
softmax router when interconnect="topk". Residual init makes the layer
≈ identity-on-a at t=0 so deep stacks train.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor

from .gates import GATE_A_INDEX, GATE_COEFFS


class LearnableTopKInterconnect(nn.Module):
    """Learnable Top-K softmax router producing 2n pair-slots from n inputs.

    Each output slot has K candidate input indices (drawn deterministically
    from seed at init) and a learnable logit per candidate. In train mode the
    output is the K-way softmax-mixture; in eval mode it's the argmax gather.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        topk: int,
        c_sparsity: float,
        seed: int,
    ):
        super().__init__()
        if topk > n_in:
            raise ValueError(f"topk={topk} cannot exceed n_in={n_in}")
        self.n_in = n_in
        self.n_out = n_out
        self.topk = topk
        self.c_sparsity = float(c_sparsity)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Dirac-style init: top-K of N(0,1) gives one dominant candidate per
        # output, mimicking fixed random routing at t=0.
        cc = torch.randn(n_in, 2 * n_out, generator=gen)
        top_c, top_indices = torch.topk(cc, topk, dim=0, largest=True, sorted=True)

        self.top_c = nn.Parameter(top_c)
        self.register_buffer("top_indices", top_indices)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., n_in) → (..., 2*n_out)."""
        if self.training:
            gathered = x[..., self.top_indices]                       # (..., K, 2n_out)
            w = torch.softmax(self.top_c * self.c_sparsity, dim=0)    # (K, 2n_out)
            return (gathered * w).sum(dim=-2)                         # (..., 2n_out)
        top1 = torch.argmax(self.top_c, dim=0)                        # (2n_out,)
        idx = self.top_indices[top1, torch.arange(2 * self.n_out, device=self.top_indices.device)]
        return x.index_select(-1, idx)


class LogicLayer(nn.Module):
    """N -> N width-preserving differentiable logic-gate layer.

    Args:
        n: input/output width.
        seed: deterministic seed for the random connection table.
        residual_init_strength: logit applied to gate "A" (passthrough on a)
            at init. softmax([s, 0, ..., 0])[A] = e^s / (e^s + 15);
            s=7.5 → ≈ 0.9918 (leakage ≈ 0.0082 across the other 15 gates).
        interconnect: "fixed" (default) uses random pi_a/pi_b buffers;
            "topk" uses a learnable Top-K softmax router.
        topk: K candidates per pair-slot when interconnect="topk".
        c_sparsity: softmax temperature on the router logits.
    """

    def __init__(
        self,
        n: int,
        seed: int,
        residual_init_strength: float = 7.5,
        interconnect: str = "fixed",
        topk: int = 8,
        c_sparsity: float = 1.0,
    ):
        super().__init__()
        if interconnect not in ("fixed", "topk"):
            raise ValueError(f"interconnect must be 'fixed' or 'topk', got {interconnect!r}")
        self.n = n
        self.residual_init_strength = float(residual_init_strength)
        self.interconnect_kind = interconnect

        if interconnect == "fixed":
            gen = torch.Generator(device="cpu").manual_seed(seed)
            pi_a = torch.randint(0, n, (n,), generator=gen, dtype=torch.long)
            pi_b = torch.randint(0, n, (n,), generator=gen, dtype=torch.long)
            self.register_buffer("pi_a", pi_a)
            self.register_buffer("pi_b", pi_b)
        else:
            self.interconnect = LearnableTopKInterconnect(
                n_in=n, n_out=n, topk=topk, c_sparsity=c_sparsity, seed=seed,
            )

        self.register_buffer("gate_coeffs", torch.tensor(GATE_COEFFS, dtype=torch.float32))

        W = torch.zeros(n, 16)
        W[:, GATE_A_INDEX] = self.residual_init_strength
        self.W = nn.Parameter(W)

    def _route(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.interconnect_kind == "fixed":
            a = x.index_select(-1, self.pi_a)
            b = x.index_select(-1, self.pi_b)
            return a, b
        pair = self.interconnect(x)                          # (..., 2n)
        a, b = pair[..., : self.n], pair[..., self.n :]
        return a, b

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., n) in [0,1]. Returns (..., n) in [0,1].

        Polynomial form: Σ_g p_g · gate_g(a, b) = α + β·a + γ·b + δ·a·b, where
        (α, β, γ, δ) = p @ GATE_COEFFS. Mathematically identical to
        (all_gates_stack(a, b) * p).sum(-1), but never materializes the
        (..., n, 16) stack — the dominant activation cost on the old path.
        """
        a, b = self._route(x)
        p = torch.softmax(self.W, dim=-1)                    # (n, 16)
        coeffs = p @ self.gate_coeffs                        # (n, 4)
        alpha, beta, gamma, delta = coeffs.unbind(dim=-1)
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
