import torch
import pytest
from nanolgn.gpt import RMSNorm

def test_rmsnorm_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(2, 5, 64)
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

def test_rmsnorm_no_learnable_params():
    norm = RMSNorm(64)
    assert sum(p.numel() for p in norm.parameters()) == 0
