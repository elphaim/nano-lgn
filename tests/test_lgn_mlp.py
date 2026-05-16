# tests/test_lgn_mlp.py
import torch
import pytest
from nanolgn.lgn_mlp import ThermometerEncode, GroupSumDecode, LGNMLPBlock

def test_thermo_shape():
    enc = ThermometerEncode(d_model=128, k=16)
    x = torch.randn(2, 7, 128)
    y = enc(x)
    assert y.shape == (2, 7, 16 * 128)

def test_thermo_output_in_unit_interval():
    enc = ThermometerEncode(d_model=64, k=8)
    x = torch.randn(4, 3, 64) * 5.0  # large-ish range
    y = enc(x)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)

def test_thermo_thresholds_initialized_spread():
    enc = ThermometerEncode(d_model=4, k=8)
    # All features share the same K thresholds at init, sorted ascending
    # via inverse_sigmoid(k/(K+1)) for k=1..K.
    theta_per_feature = enc.theta[0]                   # (K,)
    sorted_theta, _ = torch.sort(theta_per_feature)
    assert torch.allclose(theta_per_feature, sorted_theta)
    # Sigmoid of the thresholds should be evenly spread between 0 and 1.
    sig = torch.sigmoid(theta_per_feature)
    diffs = sig[1:] - sig[:-1]
    assert torch.all(diffs > 0)
    assert torch.allclose(diffs, diffs[0] * torch.ones_like(diffs), atol=1e-5)

def test_thermo_finite_backward():
    enc = ThermometerEncode(d_model=32, k=8)
    x = torch.randn(2, 32, requires_grad=True)
    enc(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(enc.theta.grad).all()
    assert torch.isfinite(enc.s.grad).all()

def test_thermo_param_count_is_dK_plus_K():
    d, k = 128, 16
    enc = ThermometerEncode(d_model=d, k=k)
    n = sum(p.numel() for p in enc.parameters())
    assert n == d * k + k


def test_thermo_output_dtype_follows_input():
    """Under autocast(bf16) the input arrives as bf16 but params stay fp32; the
    encoder must cast back to the input dtype or the entire LGN body runs in
    fp32 and quadruples activation memory."""
    enc = ThermometerEncode(d_model=8, k=4)
    x_bf16 = torch.randn(2, 8, dtype=torch.bfloat16)
    assert enc(x_bf16).dtype == torch.bfloat16
    x_fp32 = torch.randn(2, 8, dtype=torch.float32)
    assert enc(x_fp32).dtype == torch.float32


def test_decode_shape_and_grouping():
    dec = GroupSumDecode(d_model=4, k=3, tau=3.0)
    # Build z so each group sums to a known value:
    # group i gets values [i, i, i] → sum = 3i → /tau=3 → i → −0.5.
    z = torch.tensor([
        [0., 0., 0.,  1., 1., 1.,  2., 2., 2.,  3., 3., 3.],
    ])  # (1, 12) = (1, d*K)
    y = dec(z)
    assert y.shape == (1, 4)
    expected = torch.tensor([[-0.5, 0.5, 1.5, 2.5]])
    assert torch.allclose(y, expected)

def test_decode_default_tau_centers_at_zero_for_uniform_half():
    dec = GroupSumDecode(d_model=8, k=16, tau=16.0)
    # Inputs uniformly = 0.5 → group sum = 8 → /tau = 0.5 → −0.5 → 0.
    z = torch.full((2, 5, 8 * 16), 0.5)
    y = dec(z)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

def test_decode_no_learnable_params():
    dec = GroupSumDecode(d_model=8, k=16, tau=16.0)
    assert sum(p.numel() for p in dec.parameters()) == 0


def test_block_shape_contract_matches_mlp_slot():
    block = LGNMLPBlock(d_model=128, k=16, depth=4, tau=16.0, seed=0)
    x = torch.randn(2, 7, 128)
    y = block(x)
    assert y.shape == (2, 7, 128)

def test_block_finite_forward_backward():
    block = LGNMLPBlock(d_model=64, k=8, depth=3, tau=8.0, seed=0)
    x = torch.randn(2, 5, 64, requires_grad=True)
    block(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in block.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

def test_block_output_is_centered_at_init_for_zero_input():
    # With x=0, sigmoid output ≈ sigmoid(-theta) ≈ uniform spread; under
    # residual init each LGNlayer ≈ passthrough; GroupSum then /K and shifts
    # by -0.5. Just check finiteness and bounded range, not exact zero.
    block = LGNMLPBlock(d_model=32, k=8, depth=2, tau=8.0, seed=0)
    x = torch.zeros(1, 4, 32)
    y = block(x)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 1.0


def test_lgnmlpblock_topk_shape_contract_matches_mlp_slot():
    from nanolgn.lgn_mlp import LGNMLPBlock
    block = LGNMLPBlock(
        d_model=32, k=4, depth=2, tau=4.0, seed=0,
        interconnect="topk", topk=3, c_sparsity=1.0,
    )
    x = torch.randn(2, 5, 32)
    y = block(x)
    assert y.shape == (2, 5, 32)


def test_lgnmlpblock_topk_finite_backward():
    from nanolgn.lgn_mlp import LGNMLPBlock
    block = LGNMLPBlock(
        d_model=32, k=4, depth=2, tau=4.0, seed=0,
        interconnect="topk", topk=3, c_sparsity=1.0,
    )
    x = torch.randn(2, 5, 32, requires_grad=True)
    loss = block(x).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    for layer in block.body.layers:
        assert torch.isfinite(layer.interconnect.top_c.grad).all()
