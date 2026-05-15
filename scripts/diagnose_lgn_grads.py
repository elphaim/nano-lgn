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


def make_synthetic_data(vocab_size: int, n_tokens: int, device: torch.device) -> torch.Tensor:
    """Return a 1-D LongTensor of length n_tokens with seq[t] = t % 7.

    The `vocab_size` argument is accepted for API symmetry but unused — the
    `t % 7` sequence only uses tokens 0..6, well below any realistic vocab.
    """
    del vocab_size  # accepted for API symmetry, not used
    return torch.arange(n_tokens, dtype=torch.long, device=device) % 7


def _grad_norm(p: nn.Parameter) -> float:
    """L2 norm of p.grad in fp32. Returns 0.0 if grad is None."""
    if p.grad is None:
        return 0.0
    return torch.linalg.vector_norm(p.grad.detach().float()).item()


def compute_grad_metrics(groups: dict) -> dict:
    """Compute the per-measurement dict of gradient norms.

    Mean-of-norms aggregation (apples-to-apples between LGN-W and attn):
      - grad_W:           mean of |W.grad|_2 across all 16 LGN-W tensors
      - grad_W_per_block: list of 4 floats, mean within each block
      - grad_attn:        mean of |qkv.weight.grad|_2 across 4 attn blocks
      - grad_embed, grad_lm_head: single-tensor L2 norms
      - ratio:            grad_W / grad_attn (avoiding div-by-zero)
    """
    W_per_block = groups["W_per_block"]
    grad_W_per_block = [
        float(np.mean([_grad_norm(w) for w in row]))
        for row in W_per_block
    ]
    grad_W = float(np.mean(grad_W_per_block))
    grad_attn = float(np.mean([_grad_norm(a) for a in groups["attn"]]))
    grad_embed = _grad_norm(groups["embed"])
    grad_lm_head = _grad_norm(groups["head"])
    ratio = grad_W / grad_attn if grad_attn > 0 else float("inf")

    return {
        "grad_W":           grad_W,
        "grad_W_per_block": grad_W_per_block,
        "grad_attn":        grad_attn,
        "grad_embed":       grad_embed,
        "grad_lm_head":     grad_lm_head,
        "ratio":            ratio,
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

    groups = discover_params(model)
    print(f"discovered: {len(groups['W_per_block'])} blocks x "
          f"{len(groups['W_per_block'][0])} LGN-W tensors, "
          f"{len(groups['attn'])} attn-qkv, 1 embed, 1 head")

    if args.steps <= 0:
        return 0

    # Synthetic data: enough tokens for B*T random windows over `steps` iterations.
    B = 8
    T = cfg.ctx_len
    n_tokens = max(200_000, B * T * 20)
    seq = make_synthetic_data(cfg.vocab_size, n_tokens, device)
    rng = np.random.default_rng(args.seed)

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = rng.integers(0, n_tokens - T - 1, size=B)
        xs = torch.stack([seq[s:s + T] for s in starts])
        ys = torch.stack([seq[s + 1:s + T + 1] for s in starts])
        return xs, ys

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    use_autocast = device.type == "cuda"

    measurements: list[dict] = []
    log_steps = {0} | set(range(10, args.steps + 1, 10))
    if len(log_steps) == 1:
        print(f"warning: --steps {args.steps} < 10; only step 0 will be measured "
              f"(verdict will be degenerate)", file=sys.stderr)

    for step in range(args.steps + 1):
        x, y = sample_batch()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if step in log_steps:
            m = compute_grad_metrics(groups)
            m["step"] = step
            m["loss"] = float(loss.item())
            measurements.append(m)

        opt.step()

    # Temporary diagnostic print (replaced by formatted tables in Task 3):
    for m in measurements:
        print(m)

    return 0


if __name__ == "__main__":
    sys.exit(main())
