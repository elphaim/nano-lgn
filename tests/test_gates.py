# tests/test_gates.py
import torch
import pytest
from nanolgn.gates import GATE_NAMES, gate, GATE_FNS, GATE_COEFFS

CORNERS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

# Truth tables indexed by (a, b) ∈ {0,1}².
# Order MUST match GATE_NAMES below.
TRUTH = {
    "FALSE":   [0, 0, 0, 0],
    "AND":     [0, 0, 0, 1],
    "A_AND_NB":[0, 0, 1, 0],
    "A":       [0, 0, 1, 1],
    "NA_AND_B":[0, 1, 0, 0],
    "B":       [0, 1, 0, 1],
    "XOR":     [0, 1, 1, 0],
    "OR":      [0, 1, 1, 1],
    "NOR":     [1, 0, 0, 0],
    "XNOR":    [1, 0, 0, 1],
    "NB":      [1, 0, 1, 0],
    "A_OR_NB": [1, 0, 1, 1],
    "NA":      [1, 1, 0, 0],
    "NA_OR_B": [1, 1, 0, 1],
    "NAND":    [1, 1, 1, 0],
    "TRUE":    [1, 1, 1, 1],
}

def test_gate_names_are_16_unique():
    assert len(GATE_NAMES) == 16
    assert len(set(GATE_NAMES)) == 16

def test_truth_table_matches_at_corners():
    for g_idx, name in enumerate(GATE_NAMES):
        expected = TRUTH[name]
        for (a_val, b_val), exp in zip(CORNERS, expected):
            a = torch.tensor(a_val)
            b = torch.tensor(b_val)
            got = GATE_FNS[g_idx](a, b).item()
            assert abs(got - exp) < 1e-6, f"{name}({a_val},{b_val}) = {got}, expected {exp}"

def test_gate_dispatch_function_matches_index():
    a = torch.rand(4)
    b = torch.rand(4)
    for g_idx in range(16):
        got = gate(g_idx, a, b)
        expected = GATE_FNS[g_idx](a, b)
        assert torch.allclose(got, expected)

def test_gates_are_vectorized():
    a = torch.rand(3, 5)
    b = torch.rand(3, 5)
    for g_idx in range(16):
        out = GATE_FNS[g_idx](a, b)
        assert out.shape == (3, 5)
        assert torch.all(out >= 0.0 - 1e-6)
        assert torch.all(out <= 1.0 + 1e-6)


def test_gate_coeffs_reproduce_each_gate():
    """For every gate g, α + β·a + γ·b + δ·a·b must equal GATE_FNS[g](a, b)."""
    coeffs = torch.tensor(GATE_COEFFS)
    assert coeffs.shape == (16, 4)
    torch.manual_seed(0)
    a = torch.rand(64)
    b = torch.rand(64)
    for g_idx in range(16):
        alpha, beta, gamma, delta = coeffs[g_idx].tolist()
        got = alpha + beta * a + gamma * b + delta * a * b
        expected = GATE_FNS[g_idx](a, b)
        assert torch.allclose(got, expected, atol=1e-6), f"{GATE_NAMES[g_idx]}"
