"""End-to-end smoke training test.

Trains a tiny LGN model for ~30 steps on synthetic in-memory data and asserts
that val loss at step 30 is strictly less than at step 5. Catches "does it
train at all" without paying for a full run.

Marked as `slow` so it can be skipped in fast pytest runs.
"""
import math
import numpy as np
import torch
import pytest
from nanolgn.config import TransformerCfg, LGNCfg, make_ffn_factory
from nanolgn.gpt import GPT


@pytest.mark.slow
def test_lgn_model_trains_on_synthetic_data():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = TransformerCfg(
        d_model=32, n_layer=2, n_head=2, ctx_len=16,
        vocab_size=64, ffn="lgn", seed=0,
    )
    lgn = LGNCfg(K=4, L=2, tau=4.0)  # residual_init_strength defaults to s=7.5
    model = GPT(cfg, ffn_factory=make_ffn_factory(cfg, lgn=lgn))

    # Synthetic data: a simple repeating pattern the model should learn fast.
    # Token at position t is t % 7 → next-token target is (t+1) % 7.
    seq = torch.tensor([(t % 7) for t in range(2000)], dtype=torch.long)

    def batch():
        starts = torch.randint(0, len(seq) - 17, (8,))
        xs = torch.stack([seq[s:s+16] for s in starts])
        ys = torch.stack([seq[s+1:s+17] for s in starts])
        return xs, ys

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))

    losses = []
    for step in range(30):
        x, y = batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early = sum(losses[:5]) / 5
    late  = sum(losses[-5:]) / 5
    assert math.isfinite(late)
    assert late < early - 0.1, f"loss did not decrease: early={early:.3f} late={late:.3f}"
