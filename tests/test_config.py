import torch
import pytest
from nanolgn.config import TransformerCfg, LGNCfg, make_ffn_factory
from nanolgn.gpt import GPT

def test_make_factory_mlp():
    cfg = TransformerCfg(
        d_model=32, n_layer=2, n_head=2, ctx_len=16,
        vocab_size=257, ffn="mlp", seed=0,
    )
    factory = make_ffn_factory(cfg, lgn=None)
    gpt = GPT(cfg, ffn_factory=factory)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    assert gpt(idx).shape == (1, 4, cfg.vocab_size)

def test_make_factory_lgn():
    cfg = TransformerCfg(
        d_model=32, n_layer=2, n_head=2, ctx_len=16,
        vocab_size=257, ffn="lgn", seed=0,
    )
    lgn = LGNCfg(K=8, L=2, tau=8.0, residual_init_strength=7.5)
    factory = make_ffn_factory(cfg, lgn=lgn)
    gpt = GPT(cfg, ffn_factory=factory)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    assert gpt(idx).shape == (1, 4, cfg.vocab_size)

def test_lgn_factory_raises_when_lgn_cfg_missing():
    cfg = TransformerCfg(
        d_model=32, n_layer=2, n_head=2, ctx_len=16,
        vocab_size=257, ffn="lgn", seed=0,
    )
    with pytest.raises(ValueError, match="lgn"):
        make_ffn_factory(cfg, lgn=None)

def test_lgn_factory_uses_distinct_seed_per_block():
    cfg = TransformerCfg(
        d_model=32, n_layer=3, n_head=2, ctx_len=16,
        vocab_size=257, ffn="lgn", seed=42,
    )
    lgn = LGNCfg(K=4, L=1, tau=4.0, residual_init_strength=7.5)
    factory = make_ffn_factory(cfg, lgn=lgn)
    gpt = GPT(cfg, ffn_factory=factory)
    pi0 = gpt.blocks[0].ffn.body.layers[0].pi_a
    pi1 = gpt.blocks[1].ffn.body.layers[0].pi_a
    pi2 = gpt.blocks[2].ffn.body.layers[0].pi_a
    assert not torch.equal(pi0, pi1)
    assert not torch.equal(pi1, pi2)
