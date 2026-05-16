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


from nanolgn.lgn import LearnableTopKInterconnect


def _make_topk(n_in=64, n_out=64, topk=4, c_sparsity=1.0, seed=0):
    return LearnableTopKInterconnect(
        n_in=n_in, n_out=n_out, topk=topk, c_sparsity=c_sparsity, seed=seed,
    )


def test_topk_interconnect_train_shape():
    ic = _make_topk(n_in=64, n_out=64, topk=4)
    ic.train()
    x = torch.rand(2, 7, 64)
    out = ic(x)
    assert out.shape == (2, 7, 128)  # 2 * n_out


def test_topk_interconnect_eval_shape():
    ic = _make_topk(n_in=64, n_out=64, topk=4)
    ic.eval()
    x = torch.rand(2, 7, 64)
    out = ic(x)
    assert out.shape == (2, 7, 128)


def test_topk_interconnect_param_and_buffer_layout():
    ic = _make_topk(n_in=32, n_out=16, topk=5)
    param_names = {name for name, _ in ic.named_parameters()}
    buffer_names = {name for name, _ in ic.named_buffers()}
    assert "top_c" in param_names
    assert "top_indices" in buffer_names
    assert ic.top_c.shape == (5, 32)         # (topk, 2 * n_out)
    assert ic.top_indices.shape == (5, 32)


def test_topk_interconnect_indices_are_valid():
    ic = _make_topk(n_in=64, n_out=64, topk=4, seed=123)
    assert ic.top_indices.min().item() >= 0
    assert ic.top_indices.max().item() < 64


def test_topk_interconnect_determinism_on_seed():
    a = _make_topk(seed=42)
    b = _make_topk(seed=42)
    assert torch.equal(a.top_indices, b.top_indices)
    assert torch.equal(a.top_c, b.top_c)


def test_topk_interconnect_different_seeds_differ():
    a = _make_topk(seed=0)
    b = _make_topk(seed=1)
    assert not torch.equal(a.top_indices, b.top_indices)


def test_topk_interconnect_finite_backward():
    ic = _make_topk(n_in=64, n_out=64, topk=4)
    ic.train()
    x = torch.rand(2, 64, requires_grad=True)
    out = ic(x).sum()
    out.backward()
    assert torch.isfinite(out)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(ic.top_c.grad).all()


def test_topk_interconnect_eval_matches_train_when_top_c_is_one_hot():
    """If top_c is one-hot, softmax → argmax, so train output == eval output."""
    ic = _make_topk(n_in=64, n_out=64, topk=4, c_sparsity=1.0, seed=7)
    with torch.no_grad():
        ic.top_c.zero_()
        ic.top_c[0, :] = 100.0  # K=0 saturates the softmax
    x = torch.rand(3, 64)
    ic.train()
    y_train = ic(x)
    ic.eval()
    y_eval = ic(x)
    assert torch.allclose(y_train, y_eval, atol=1e-5)


def test_topk_interconnect_output_in_unit_interval_for_unit_input():
    ic = _make_topk(n_in=64, n_out=64, topk=4)
    ic.train()
    x = torch.rand(4, 64)  # in [0, 1]
    out = ic(x)
    assert torch.all(out >= -1e-5)
    assert torch.all(out <= 1.0 + 1e-5)


def test_topk_interconnect_c_sparsity_changes_forward_output():
    """Regression guard: c_sparsity must reach the forward path. Same seed
    fixes top_c and top_indices, so the only thing that can change the
    train output is the c_sparsity multiplier on the softmax logits."""
    ic_lo = _make_topk(n_in=64, n_out=64, topk=4, c_sparsity=1.0, seed=0)
    ic_hi = _make_topk(n_in=64, n_out=64, topk=4, c_sparsity=5.0, seed=0)
    assert torch.equal(ic_lo.top_c, ic_hi.top_c)
    assert torch.equal(ic_lo.top_indices, ic_hi.top_indices)
    ic_lo.train()
    ic_hi.train()
    x = torch.rand(2, 64)
    y_lo = ic_lo(x)
    y_hi = ic_hi(x)
    assert not torch.allclose(y_lo, y_hi, atol=1e-4)


def test_logic_layer_default_is_fixed_routing():
    layer = LogicLayer(n=64, seed=0)
    # Fixed routing keeps pi_a, pi_b buffers and does not register top_c.
    buffer_names = {name for name, _ in layer.named_buffers()}
    param_names = {name for name, _ in layer.named_parameters()}
    assert "pi_a" in buffer_names
    assert "pi_b" in buffer_names
    assert "interconnect.top_c" not in param_names


def test_logic_layer_topk_routing_shape():
    layer = LogicLayer(n=64, seed=0, interconnect="topk", topk=4, c_sparsity=1.0)
    x = torch.rand(2, 7, 64)
    y = layer(x)
    assert y.shape == (2, 7, 64)


def test_logic_layer_topk_routing_output_in_unit_interval():
    layer = LogicLayer(n=128, seed=0, interconnect="topk", topk=4, c_sparsity=1.0)
    x = torch.rand(4, 5, 128)
    y = layer(x)
    assert torch.all(y >= -1e-5)
    assert torch.all(y <= 1.0 + 1e-5)


def test_logic_layer_topk_finite_backward_to_both_W_and_top_c():
    layer = LogicLayer(n=64, seed=0, interconnect="topk", topk=4, c_sparsity=1.0)
    x = torch.rand(2, 3, 64, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert torch.isfinite(y)
    assert torch.isfinite(layer.W.grad).all()
    assert torch.isfinite(layer.interconnect.top_c.grad).all()


def test_logic_layer_topk_param_set_includes_top_c():
    layer = LogicLayer(n=64, seed=0, interconnect="topk", topk=4)
    param_names = {name for name, _ in layer.named_parameters()}
    assert "W" in param_names
    assert "interconnect.top_c" in param_names


def test_logic_layer_topk_rejects_invalid_interconnect_value():
    with pytest.raises(ValueError):
        LogicLayer(n=64, seed=0, interconnect="bogus")


def test_lgn_body_topk_shape():
    body = LGNBody(n=128, depth=3, seed=0, interconnect="topk", topk=4, c_sparsity=1.0)
    x = torch.rand(2, 5, 128)
    assert body(x).shape == (2, 5, 128)


def test_lgn_body_topk_gradients_flow_to_all_top_c():
    body = LGNBody(n=64, depth=3, seed=0, interconnect="topk", topk=4, c_sparsity=1.0)
    x = torch.rand(2, 64)
    body(x).sum().backward()
    for layer in body.layers:
        assert layer.W.grad is not None
        assert layer.interconnect.top_c.grad is not None
        assert torch.isfinite(layer.interconnect.top_c.grad).all()


def test_lgn_body_topk_per_layer_seeds_differ():
    body = LGNBody(n=64, depth=3, seed=0, interconnect="topk", topk=4)
    indices = [layer.interconnect.top_indices for layer in body.layers]
    assert not torch.equal(indices[0], indices[1])
    assert not torch.equal(indices[1], indices[2])


def test_lgn_body_default_is_fixed():
    body = LGNBody(n=64, depth=2, seed=0)
    for layer in body.layers:
        assert layer.interconnect_kind == "fixed"
