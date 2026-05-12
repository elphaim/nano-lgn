# tests/test_lgn_mlp.py
import torch
import pytest
from nanolgn.lgn_mlp import ThermometerEncode

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
