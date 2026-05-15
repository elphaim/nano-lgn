# nanolgn/gates.py
"""Soft (product-t-norm) relaxations of the 16 binary logic gates.

Each gate is a function (a, b) -> out, where a and b are tensors with values
in [0, 1]. Outputs are in [0, 1]. Gates are listed in the same order as the
softmax weight columns in LogicLayer (gate index = column index).

Ordering is the standard 4-bit truth-table ordering on (a, b) ∈ {0,1}² read as
(b, a) → bit:
    index 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
which yields gate index 3 = "A" (passthrough on a) — used by residual init.
"""
from __future__ import annotations
import torch
from torch import Tensor

GATE_NAMES = (
    "FALSE",     # 0
    "AND",       # 1
    "A_AND_NB",  # 2
    "A",         # 3   <-- residual-init target (passthrough on a)
    "NA_AND_B",  # 4
    "B",         # 5
    "XOR",       # 6
    "OR",        # 7
    "NOR",       # 8
    "XNOR",      # 9
    "NB",        # 10
    "A_OR_NB",   # 11
    "NA",        # 12
    "NA_OR_B",   # 13
    "NAND",      # 14
    "TRUE",      # 15
)

def _g_false(a: Tensor, b: Tensor) -> Tensor:    return torch.zeros_like(a)
def _g_and(a, b):                                return a * b
def _g_a_and_nb(a, b):                           return a - a * b
def _g_a(a, b):                                  return a
def _g_na_and_b(a, b):                           return b - a * b
def _g_b(a, b):                                  return b
def _g_xor(a, b):                                return a + b - 2.0 * a * b
def _g_or(a, b):                                 return a + b - a * b
def _g_nor(a, b):                                return 1.0 - (a + b - a * b)
def _g_xnor(a, b):                               return 1.0 - (a + b - 2.0 * a * b)
def _g_nb(a, b):                                 return 1.0 - b
def _g_a_or_nb(a, b):                            return 1.0 - b + a * b
def _g_na(a, b):                                 return 1.0 - a
def _g_na_or_b(a, b):                            return 1.0 - a + a * b
def _g_nand(a, b):                               return 1.0 - a * b
def _g_true(a, b):                               return torch.ones_like(a)

GATE_FNS = (
    _g_false, _g_and, _g_a_and_nb, _g_a,
    _g_na_and_b, _g_b, _g_xor, _g_or,
    _g_nor, _g_xnor, _g_nb, _g_a_or_nb,
    _g_na, _g_na_or_b, _g_nand, _g_true,
)

GATE_A_INDEX = 3  # passthrough on input a — used by residual-init in LogicLayer.

# Coefficients (α, β, γ, δ) such that gate_g(a, b) = α + β·a + γ·b + δ·a·b.
# Every binary gate is multilinear of degree ≤ 2 on (a, b), so the softmax
# mixture Σ_g p_g · gate_g collapses to (Σ p_g α_g) + (Σ p_g β_g)·a + …,
# letting LogicLayer skip the (..., n, 16) stack. Same ordering as GATE_FNS.
GATE_COEFFS: tuple[tuple[float, float, float, float], ...] = (
    (0.0,  0.0,  0.0,  0.0),   # FALSE
    (0.0,  0.0,  0.0,  1.0),   # AND
    (0.0,  1.0,  0.0, -1.0),   # A_AND_NB
    (0.0,  1.0,  0.0,  0.0),   # A
    (0.0,  0.0,  1.0, -1.0),   # NA_AND_B
    (0.0,  0.0,  1.0,  0.0),   # B
    (0.0,  1.0,  1.0, -2.0),   # XOR
    (0.0,  1.0,  1.0, -1.0),   # OR
    (1.0, -1.0, -1.0,  1.0),   # NOR
    (1.0, -1.0, -1.0,  2.0),   # XNOR
    (1.0,  0.0, -1.0,  0.0),   # NB
    (1.0,  0.0, -1.0,  1.0),   # A_OR_NB
    (1.0, -1.0,  0.0,  0.0),   # NA
    (1.0, -1.0,  0.0,  1.0),   # NA_OR_B
    (1.0,  0.0,  0.0, -1.0),   # NAND
    (1.0,  0.0,  0.0,  0.0),   # TRUE
)

def gate(idx: int, a: Tensor, b: Tensor) -> Tensor:
    """Dispatch wrapper: gate(idx, a, b) = GATE_FNS[idx](a, b)."""
    return GATE_FNS[idx](a, b)


def all_gates_stack(a: Tensor, b: Tensor) -> Tensor:
    """Stack all 16 gate outputs along a new last dim. Shape: (..., 16)."""
    return torch.stack([fn(a, b) for fn in GATE_FNS], dim=-1)
