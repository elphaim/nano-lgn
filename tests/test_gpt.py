import torch
import pytest
from nanolgn.gpt import RMSNorm
from nanolgn.gpt import precompute_rope, apply_rope
from nanolgn.gpt import CausalSelfAttention
from nanolgn.gpt import ReLU2MLP

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

def test_attention_shape():
    attn = CausalSelfAttention(d_model=64, n_head=4, ctx_len=32)
    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert y.shape == (2, 16, 64)

def test_attention_is_causal_changing_future_does_not_change_past():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_head=2, ctx_len=8)
    x = torch.randn(1, 8, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 32)   # change positions 5..7
    y2 = attn(x2)
    # Positions 0..4 must be unchanged.
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-5)

def test_relu2_mlp_shape():
    mlp = ReLU2MLP(d_model=128, mult=4)
    x = torch.randn(2, 7, 128)
    assert mlp(x).shape == (2, 7, 128)

def test_relu2_mlp_param_count():
    mlp = ReLU2MLP(d_model=128, mult=4)
    expected = 128 * 512 + 512 * 128   # 2 linear, no bias
    assert sum(p.numel() for p in mlp.parameters()) == expected
