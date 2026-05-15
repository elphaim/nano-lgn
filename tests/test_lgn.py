# tests/test_lgn.py
import torch
import pytest
from nanolgn.lgn import LogicLayer

def _make(n=64, seed=0, residual_init_strength=7.5):
    return LogicLayer(n=n, seed=seed, residual_init_strength=residual_init_strength)

def test_shape_preserves_width_and_batch():
    layer = _make(n=64)
    x = torch.rand(2, 7, 64)
    y = layer(x)
    assert y.shape == (2, 7, 64)

def test_output_in_unit_interval():
    layer = _make(n=128)
    x = torch.rand(4, 5, 128)
    y = layer(x)
    assert torch.all(y >= -1e-5)
    assert torch.all(y <= 1.0 + 1e-5)

def test_finite_forward_and_backward():
    layer = _make(n=64)
    x = torch.rand(2, 3, 64, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert torch.isfinite(y)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.W.grad).all()

def test_residual_init_is_approximately_identity_on_pi_a():
    n = 256
    layer = _make(n=n, residual_init_strength=7.5)
    x = torch.rand(8, n)
    y = layer(x)
    expected = x[..., layer.pi_a]  # passthrough on input slot a
    # softmax([7.5, 0, ..., 0])[A] = e^7.5 / (e^7.5 + 15)
    #                              ≈ 1808.04 / 1823.04 ≈ 0.9918
    # leakage ≈ 0.0082 spread across 15 gates; max elementwise error ≲ 0.0082.
    assert torch.allclose(y, expected, atol=2e-2)

def test_determinism_same_seed_same_connections():
    a = _make(n=64, seed=42)
    b = _make(n=64, seed=42)
    assert torch.equal(a.pi_a, b.pi_a)
    assert torch.equal(a.pi_b, b.pi_b)

def test_different_seeds_give_different_connections():
    a = _make(n=64, seed=0)
    b = _make(n=64, seed=1)
    assert not torch.equal(a.pi_a, b.pi_a)

def test_W_is_a_parameter_pi_are_buffers():
    layer = _make(n=64)
    param_names = {name for name, _ in layer.named_parameters()}
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert "W" in param_names
    assert "pi_a" in buffer_names
    assert "pi_b" in buffer_names

from nanolgn.lgn import LGNBody

def test_lgn_body_shape():
    body = LGNBody(n=128, depth=4, seed=0)
    x = torch.rand(2, 3, 128)
    assert body(x).shape == (2, 3, 128)

def test_lgn_body_residual_init_is_approx_identity_chain():
    n, L = 128, 6
    body = LGNBody(n=n, depth=L, seed=0, residual_init_strength=7.5)
    x = torch.rand(4, n)
    y = body(x)
    # Each layer is ≈ passthrough on its own pi_a (different per layer).
    # The composition is the chained gather. Walk it explicitly.
    z = x
    for layer in body.layers:
        z = z[..., layer.pi_a]
    # At s=7.5 per-layer leakage ≲ 0.0082; 6 layers compound it. atol=5e-2
    # leaves headroom while still meaningfully constraining drift.
    assert torch.allclose(y, z, atol=5e-2)

def test_lgn_body_layer_seeds_differ():
    body = LGNBody(n=64, depth=3, seed=0)
    pis = [layer.pi_a for layer in body.layers]
    assert not torch.equal(pis[0], pis[1])
    assert not torch.equal(pis[1], pis[2])

def test_lgn_body_gradients_flow_to_all_W():
    body = LGNBody(n=64, depth=3, seed=0)
    x = torch.rand(2, 64)
    body(x).sum().backward()
    for layer in body.layers:
        assert layer.W.grad is not None
        assert torch.isfinite(layer.W.grad).all()


def test_logiclayer_polynomial_matches_explicit_stack_form():
    """Polynomial form must equal (all_gates_stack(a,b) * softmax(W)).sum(-1)."""
    from nanolgn.gates import all_gates_stack
    n = 64
    layer = _make(n=n, seed=42)
    # Perturb W so we exercise a non-degenerate softmax.
    with torch.no_grad():
        layer.W.add_(torch.randn_like(layer.W))
    x = torch.rand(2, 5, n)
    y_poly = layer(x)
    a = x.index_select(-1, layer.pi_a)
    b = x.index_select(-1, layer.pi_b)
    p = torch.softmax(layer.W, dim=-1)
    y_ref = (all_gates_stack(a, b) * p).sum(dim=-1)
    assert torch.allclose(y_poly, y_ref, atol=1e-6)
