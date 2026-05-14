import torch
import pytest
from nanolgn.gpt import RMSNorm
from nanolgn.gpt import precompute_rope, apply_rope

def test_rmsnorm_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(2, 5, 64)
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

def test_rmsnorm_no_learnable_params():
    norm = RMSNorm(64)
    assert sum(p.numel() for p in norm.parameters()) == 0

def test_rope_shapes():
    cos, sin = precompute_rope(dim=32, max_len=128, base=10000.0)
    assert cos.shape == (128, 32 // 2)
    assert sin.shape == (128, 32 // 2)

def test_rope_application_preserves_norm():
    head_dim = 32
    cos, sin = precompute_rope(dim=head_dim, max_len=64, base=10000.0)
    x = torch.randn(2, 4, 7, head_dim)  # (B, H, T, D)
    y = apply_rope(x, cos[:7], sin[:7])  # cos/sin sliced to T (apply_rope contract)
    # Rotation preserves L2 norm in each (even, odd) pair.
    assert torch.allclose(y.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-5)

def test_rope_zero_position_is_identity():
    head_dim = 16
    cos, sin = precompute_rope(dim=head_dim, max_len=4, base=10000.0)
    x = torch.randn(1, 1, 1, head_dim)   # one token at position 0
    y = apply_rope(x, cos[:1], sin[:1])
    assert torch.allclose(y, x, atol=1e-6)
