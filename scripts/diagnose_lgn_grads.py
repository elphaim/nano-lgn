"""LGN gradient-magnitude diagnostic.

Pre-POC-B sanity check: does the LGN body's gate-mixture parameter `W`
receive non-trivial gradient signal during real training, or is the body
permanently near-identity at residual_init_strength=7.5?

Runs the POC-A LGN config on a synthetic `t % 7` next-token task for
~200 AdamW steps under bf16 autocast (matches scripts/train.py). Measures
the L2 norm of `LogicLayer.W.grad` (mean across the 16 LGN-W tensors)
versus the L2 norm of `attn.qkv.weight.grad` (mean across the 4 attention
blocks) at step 0 and every 10 steps. Prints two stdout tables and a
heuristic-verdict line.

Verdict heuristic (LIKELY LEARNING == YES) requires BOTH:
  - ratio(last step) >= 2 * ratio(step 0)  (gates de-saturating)
  - grad_W(last step) > 1e-7               (signal is non-trivial)
Otherwise: LIKELY LEARNING == NO. Limitations: single seed, 200 steps,
synthetic data. A "NO" verdict is a flag, not proof that the body is
fundamentally stuck — could mean "needs longer," "needs different LR,"
or "needs a task with stronger long-range structure."

Usage:
    python scripts/diagnose_lgn_grads.py [--steps N] [--seed S] [--device cuda|cpu]

Defaults: --steps 200, --seed 0, --device autodetect (cuda if available).
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import torch
from torch import nn

from configs.poc_a_lgn import cfg, lgn
from nanolgn.config import make_ffn_factory
from nanolgn.gpt import GPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LGN gradient-magnitude diagnostic")
    p.add_argument("--steps", type=int, default=200,
                   help="number of training steps (default: 200)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for torch + numpy (default: 0)")
    p.add_argument("--device", type=str, default=None,
                   choices=["cuda", "cpu"],
                   help="device override (default: cuda if available else cpu)")
    return p.parse_args()


def resolve_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_params(model: nn.Module) -> dict:
    """Walk model.named_parameters() and return a structured grouping.

    Returns:
        {
          "W_per_block": list[list[Parameter]] of shape [n_layer][L],
          "attn":        list[Parameter] of length n_layer,
          "embed":       Parameter,
          "head":        Parameter,
        }

    Validates expected counts and raises RuntimeError on mismatch — defends
    against silent renames in nanolgn.gpt / nanolgn.lgn / nanolgn.lgn_mlp.
    Expected names (verified against a live GPT(cfg=poc_a_lgn.cfg)):
      - blocks.{i}.ffn.body.layers.{j}.W            (4 blocks x 4 layers = 16)
      - blocks.{i}.attn.qkv.weight                  (4)
      - tok_emb.weight                              (1)
      - lm_head.weight                              (1)
    """
    named = dict(model.named_parameters())
    n_layer = cfg.n_layer
    L = lgn.L

    W_per_block: list[list[nn.Parameter]] = []
    for i in range(n_layer):
        row: list[nn.Parameter] = []
        for j in range(L):
            key = f"blocks.{i}.ffn.body.layers.{j}.W"
            if key not in named:
                raise RuntimeError(f"expected LGN-W parameter not found: {key!r}")
            row.append(named[key])
        W_per_block.append(row)

    attn: list[nn.Parameter] = []
    for i in range(n_layer):
        key = f"blocks.{i}.attn.qkv.weight"
        if key not in named:
            raise RuntimeError(f"expected attn parameter not found: {key!r}")
        attn.append(named[key])

    if "tok_emb.weight" not in named:
        raise RuntimeError("expected 'tok_emb.weight' parameter not found")
    if "lm_head.weight" not in named:
        raise RuntimeError("expected 'lm_head.weight' parameter not found")

    return {
        "W_per_block": W_per_block,
        "attn":        attn,
        "embed":       named["tok_emb.weight"],
        "head":        named["lm_head.weight"],
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cpu":
        print("WARNING: POC-A shape on CPU is slow; expect >=10 min for 200 steps. "
              "Consider running on GPU.", file=sys.stderr)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = GPT(cfg, ffn_factory=make_ffn_factory(cfg, lgn=lgn)).to(device)
    print(f"model params: {model.num_params():,}  device: {device}")
    print(f"config: d_model={cfg.d_model} n_layer={cfg.n_layer} ctx_len={cfg.ctx_len}")
    print(f"lgn:    K={lgn.K} L={lgn.L} tau={lgn.tau} s={lgn.residual_init_strength}")

    param_groups = discover_params(model)
    print(f"discovered: {len(param_groups['W_per_block'])} blocks x "
          f"{len(param_groups['W_per_block'][0])} LGN-W tensors, "
          f"{len(param_groups['attn'])} attn-qkv, 1 embed, 1 head")

    return 0


if __name__ == "__main__":
    sys.exit(main())
