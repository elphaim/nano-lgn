# LGN Gradient-Magnitude Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/diagnose_lgn_grads.py`, a one-shot pre-POC-B diagnostic that determines whether `LogicLayer.W` parameters receive non-trivial gradient signal during real training, or whether the LGN body is permanently near-identity at `s=7.5` init.

**Architecture:** Single ~150-LOC standalone Python script. Imports the POC-A LGN config verbatim, builds the same `GPT`, runs ~200 AdamW steps on a synthetic `t % 7` next-token task under bf16 autocast on CUDA (fp32 on CPU), measures per-block gradient L2 norms at step 0 + every 10 steps, prints two stdout tables and a heuristic-verdict line. No tests, no CI, no W&B, no checkpointing.

**Tech Stack:** Python 3.11+, PyTorch 2.x, NumPy, existing `nanolgn.*` and `configs.poc_a_lgn` modules. Uses `torch.amp.autocast("cuda", dtype=torch.bfloat16, ...)` to mirror `scripts/train.py`'s numerical conditions.

**Spec:** `docs/superpowers/specs/2026-05-15-lgn-grad-diagnostic-design.md` (read this first if you have not).

**Review guidance:** Task 1 (parameter discovery + validation asserts) and Task 2 (training loop + measurement) have real design surface — dispatch both review subagents (spec compliance, then code quality). Task 3 (output formatting + verdict) is mostly mechanical from the spec; one review pass (code quality) is sufficient.

---

## Task 1: Scaffolding, CLI, model construction, parameter discovery

**Files:**
- Create: `scripts/diagnose_lgn_grads.py`

- [ ] **Step 1: Create the script with module docstring, imports, and CLI**

```python
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
from typing import Iterable

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
```

- [ ] **Step 2: Add `discover_params` function with validation asserts**

Insert this function immediately after `resolve_device` and before `main`:

```python
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
```

- [ ] **Step 3: Verify the script imports cleanly and discovers params**

Run:
```bash
.venv/bin/python -m scripts.diagnose_lgn_grads --steps 0 --device cpu
```

Expected stdout (param count and total may shift if model changes; the key
shape `4 blocks x 4 LGN-W tensors, 4 attn-qkv, 1 embed, 1 head` must appear):
```
WARNING: POC-A shape on CPU is slow; ...
model params: 13,660,480  device: cpu
config: d_model=128 n_layer=4 ctx_len=256
lgn:    K=16 L=4 tau=16.0 s=7.5
discovered: 4 blocks x 4 LGN-W tensors, 4 attn-qkv, 1 embed, 1 head
```

If any expected parameter name is missing, `discover_params` raises a
clean `RuntimeError` instead of `KeyError` on a downstream line.

- [ ] **Step 4: Commit**

```bash
git add scripts/diagnose_lgn_grads.py
git commit -m "feat(diagnostic): scaffold LGN grad-magnitude diagnostic with param discovery"
```

---

## Task 2: Synthetic data, training loop, and per-step measurement

**Files:**
- Modify: `scripts/diagnose_lgn_grads.py`

- [ ] **Step 1: Add `make_synthetic_data` helper**

Insert immediately after `discover_params` and before `main`:

```python
def make_synthetic_data(vocab_size: int, n_tokens: int, device: torch.device) -> torch.Tensor:
    """Return a 1-D LongTensor of length n_tokens with seq[t] = t % 7.

    Modulated by vocab_size for safety (cfg.vocab_size=50257 >> 7, so this
    is a no-op in practice, but it makes the function robust to vocab swaps).
    """
    raw = [(t % 7) % vocab_size for t in range(n_tokens)]
    return torch.tensor(raw, dtype=torch.long, device=device)
```

- [ ] **Step 2: Add `compute_grad_metrics` helper**

Insert immediately after `make_synthetic_data`:

```python
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
```

- [ ] **Step 3: Extend `main` with training loop + measurement schedule**

Replace the existing `main` body (after the existing `discover_params` /
`print(f"discovered: ...")` line) with the full training loop. Show the
complete new `main`:

```python
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
```

Note: the loop runs `args.steps + 1` iterations so we measure at step 0
*and* the nominal last step (e.g. with `--steps 200` we measure at steps
0, 10, 20, ..., 200 — 21 measurements).

- [ ] **Step 4: Verify the training loop runs and measurements have the right shape**

Run a short sanity loop (CPU is fine for shape verification; this is slow
but only 3 iterations):

```bash
.venv/bin/python -m scripts.diagnose_lgn_grads --steps 0 --device cpu
```

Expected: prints the param header lines then exits cleanly with no
measurements (since `args.steps <= 0` returns early).

Then run a tiny-but-measuring sanity:

```bash
.venv/bin/python -m scripts.diagnose_lgn_grads --steps 10 --device cpu
```

Expected stdout to include 2 measurement dicts (step 0 and step 10), each
containing keys `step, loss, grad_W, grad_W_per_block, grad_attn,
grad_embed, grad_lm_head, ratio`. `grad_W_per_block` is a 4-element list.
All numeric values must be finite and non-negative; `grad_attn` and
`grad_embed` should be strictly positive (these are normal Linear/Embedding
weights and see gradient on step 0).

The CPU run for `--steps 10` is slow (~2-3 min) — that's OK for a one-time
verification; production runs will use GPU.

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_lgn_grads.py
git commit -m "feat(diagnostic): training loop + per-step gradient metrics"
```

---

## Task 3: Formatted output tables, verdict heuristic, polish

**Files:**
- Modify: `scripts/diagnose_lgn_grads.py`

- [ ] **Step 1: Add `format_trajectory_table` and `format_per_block_table` helpers**

Insert immediately after `compute_grad_metrics`:

```python
def format_trajectory_table(measurements: list[dict]) -> str:
    """Table 1: per-step trajectory of gradient norms and loss."""
    header = (
        " step | loss   | |∇W|    | |∇attn| | |∇embed| | |∇head| | W/attn\n"
        "------+--------+----------+----------+-----------+----------+----------"
    )
    rows = []
    for m in measurements:
        rows.append(
            f" {m['step']:>4} | {m['loss']:>6.3f} | "
            f"{m['grad_W']:.2e} | {m['grad_attn']:.2e} | "
            f"{m['grad_embed']:.2e}  | {m['grad_lm_head']:.2e} | "
            f"{m['ratio']:.2e}"
        )
    return header + "\n" + "\n".join(rows)


def format_per_block_table(measurements: list[dict]) -> str:
    """Table 2: per-block |grad W| at first and last measurement."""
    first, last = measurements[0], measurements[-1]
    header = (
        f" block | |∇W| @ step {first['step']:<4} | |∇W| @ step {last['step']:<4} | ratio (last/first)\n"
        "-------+----------------------+----------------------+--------------------"
    )
    rows = []
    for i, (g0, g1) in enumerate(zip(first["grad_W_per_block"], last["grad_W_per_block"])):
        r = (g1 / g0) if g0 > 0 else float("inf")
        rows.append(
            f"   {i:>3} | {g0:.2e}             | {g1:.2e}             | {r:.2e}"
        )
    return header + "\n" + "\n".join(rows)
```

- [ ] **Step 2: Add `format_verdict` helper**

Insert immediately after `format_per_block_table`:

```python
def format_verdict(measurements: list[dict]) -> str:
    """Heuristic verdict: LIKELY LEARNING == YES iff BOTH conditions hold.

    - ratio(last step) >= 2 * ratio(step 0):     gap closing (gates de-saturating)
    - grad_W(last step) > 1e-7:                  signal is non-trivial

    Otherwise: LIKELY LEARNING == NO. This is a flag, not proof of failure;
    a NO verdict could mean "needs longer," "needs different LR," or
    "needs a task with stronger long-range structure."
    """
    first, last = measurements[0], measurements[-1]
    r0, r1 = first["ratio"], last["ratio"]
    grad_W_last = last["grad_W"]
    ratio_ok = r1 >= 2.0 * r0
    grad_ok = grad_W_last > 1e-7
    likely_learning = "YES" if (ratio_ok and grad_ok) else "NO"
    return (
        "=== VERDICT ===\n"
        f"ratio(step {first['step']:<4}):  {r0:.2e}\n"
        f"ratio(step {last['step']:<4}):  {r1:.2e}  (>= 2x first? {ratio_ok})\n"
        f"|∇W|(step {last['step']:<4}): {grad_W_last:.2e}  (> 1e-7? {grad_ok})\n"
        f"LIKELY LEARNING:    {likely_learning}"
    )
```

- [ ] **Step 3: Replace the temporary print loop with formatted output**

In `main`, replace these lines:
```python
    # Temporary diagnostic print (replaced by formatted tables in Task 3):
    for m in measurements:
        print(m)
```

With:
```python
    if not measurements:
        print("no measurements (steps=0)")
        return 0

    print()
    print(format_trajectory_table(measurements))
    print()
    print(format_per_block_table(measurements))
    print()
    print(format_verdict(measurements))
    return 0
```

- [ ] **Step 4: Verify the output is complete and formatted correctly**

Run a short end-to-end on CPU to confirm formatting (still slow but only
3 measurements):

```bash
.venv/bin/python -m scripts.diagnose_lgn_grads --steps 20 --device cpu
```

Expected stdout (numeric values will vary; structure must match):
```
WARNING: POC-A shape on CPU is slow; ...
model params: 13,660,480  device: cpu
config: ...
lgn:    ...
discovered: 4 blocks x 4 LGN-W tensors, 4 attn-qkv, 1 embed, 1 head

 step | loss   | |∇W|    | |∇attn| | |∇embed| | |∇head| | W/attn
------+--------+----------+----------+-----------+----------+----------
    0 | ...    | ...      | ...      | ...       | ...      | ...
   10 | ...    | ...      | ...      | ...       | ...      | ...
   20 | ...    | ...      | ...      | ...       | ...      | ...

 block | |∇W| @ step 0    | |∇W| @ step 20   | ratio (last/first)
-------+----------------------+----------------------+--------------------
     0 | ...                  | ...                  | ...
     1 | ...                  | ...                  | ...
     2 | ...                  | ...                  | ...
     3 | ...                  | ...                  | ...

=== VERDICT ===
ratio(step 0   ):  ...
ratio(step 20  ):  ...  (>= 2x first? ...)
|∇W|(step 20  ): ...  (> 1e-7? ...)
LIKELY LEARNING:    ...
```

Both tables must render with aligned columns, the verdict block must
appear with all four lines, and `LIKELY LEARNING` must be exactly `YES`
or `NO`.

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_lgn_grads.py
git commit -m "feat(diagnostic): formatted output tables + heuristic verdict"
```

---

## Out of scope (deferred follow-ups, not part of this plan)

- Pytest coverage for the script. The spec explicitly says "No tests, no CI."
- Real-TinyStories variant. Synthetic data is sufficient for this diagnostic;
  a real-data variant could be added later if the synthetic verdict is "NO"
  and we want to rule out task-specificity.
- Multi-seed averaging. Single seed; user can re-run with different `--seed`
  if they want a second data point.
- Comparison against the MLP path. The MLP path has no `LogicLayer.W`, so
  there is nothing to diagnose.
- Stronger LGN-actually-learning test (e.g., longer-horizon XOR-style task
  where attention alone cannot solve). Listed as a separate post-POC
  follow-up in `nano-lgn-poc-state.md`.

## Definition of done

After Task 3 commits cleanly:

1. `.venv/bin/python -m scripts.diagnose_lgn_grads --steps 200` runs on a T4 in under
   3 minutes and prints both tables + a verdict line.
2. `--device cpu` runs (slowly but correctly) on a machine without a GPU.
3. Renaming any of `blocks.{i}.ffn.body.layers.{j}.W`, `blocks.{i}.attn.qkv.weight`,
   `tok_emb.weight`, or `lm_head.weight` in the model causes `discover_params`
   to raise a clean `RuntimeError` with the missing key in the message —
   not a silent miscount or downstream `KeyError`.
4. The verdict line says either `LIKELY LEARNING: YES` or `LIKELY LEARNING: NO`,
   never anything else.
