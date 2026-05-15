import torch
from nanolgn.config import TransformerCfg, LGNCfg, make_ffn_factory
from nanolgn.gpt import GPT
from nanolgn.lgn_mlp import LGNMLPBlock, lgn_block_stats

def test_lgn_block_stats_returns_expected_keys():
    cfg = TransformerCfg(
        d_model=32, n_layer=1, n_head=2, ctx_len=8,
        vocab_size=257, ffn="lgn", seed=0,
    )
    lgn = LGNCfg(K=4, L=2, tau=4.0)  # residual_init_strength defaults to s=7.5
    model = GPT(cfg, ffn_factory=make_ffn_factory(cfg, lgn=lgn))
    block = model.blocks[0]
    block.cache_ffn_out = True
    idx = torch.randint(0, 257, (1, 4))
    model(idx)
    assert isinstance(block.ffn, LGNMLPBlock)
    stats = lgn_block_stats(block.ffn, block.last_ffn_out)
    assert set(stats.keys()) == {
        "ffn_out_norm_mean", "ffn_out_max",
        "gate_entropy_mean", "threshold_in_range_frac",
    }
    for v in stats.values():
        assert isinstance(v, float)
