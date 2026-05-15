# LGN gradient-magnitude diagnostic — design

**Date:** 2026-05-15
**Status:** approved, ready for implementation plan
**Scope:** one-shot pre-POC-B diagnostic script

## Motivation

POC-A finished with LGN val_loss = 2.7874 vs MLP val_loss = 2.5404 (+9.72 %, past
the Stretch criterion). However, the Task 20 smoke-test review found that at
`residual_init_strength = 7.5`, `|grad LogicLayer.W|` is ~10³–10⁴× smaller than
`|grad attn.c_attn.weight|` at step 0 — the expected consequence of softmax
saturation (`p_A ≈ 0.9918`, so `∂p / ∂W` has a factor of `p_A·(1-p_A) ≈ 0.008`,
~125× intrinsic shrinkage on top of upstream-gradient differences).

This is consistent with two scenarios that POC-A val_loss alone cannot
distinguish:

1. **LGN is learning a little.** Gates de-saturate over training, body picks up
   non-identity logical structure, contributes to the loss curve.
2. **LGN body is permanently near-identity.** Gates stay saturated at `p_A ≈
   0.99…`, body is effectively a no-op, all the loss decrease is being done by
   the surrounding attention / embedding / LM head.

The diagnostic determines which of these is closer to reality, before we
commit GPU-hours to POC-B (3–6 h/run on FineWeb-edu).

## Non-goals

- **Not** proving the LGN body is learning *useful* logical features. That
  requires a longer-horizon task where attention alone is insufficient (e.g.,
  XOR-style problems). Listed as a separate post-POC follow-up.
- **Not** a regression test. Lives in `scripts/`, not `tests/`. One-shot tool.
- **Not** a production training script. No checkpointing, no W&B, no eval loop.

## Architecture

### File

`scripts/diagnose_lgn_grads.py` — single Python file, ~150 lines.

### Configuration

Imports the existing POC-A LGN config verbatim:

```python
from configs.poc_a_lgn import cfg, lgn
```

This gives `d_model=128, n_layer=4, n_head=4, ctx_len=256, vocab_size=50257`,
`K=8, L=4, tau=4.0, residual_init_strength=7.5`. No shape overrides — fidelity
to the configuration we're judging matters more than diagnostic speed.

### Data

In-memory synthetic stream, identical pattern to `tests/test_smoke_train.py`:

```python
seq = torch.tensor([(t % 7) for t in range(N)], dtype=torch.long)
```

`N` is large enough to draw `--steps` random `(B=8, T=256)` batches without
repetition pressure (`N = 200_000` is plenty for the default 200 steps).
No `data/*.bin` dependency. The synthetic task produces non-trivial gradients
through the whole network (smoke test confirms loss decrease in 30 steps);
that's all we need to study gradient flow into LGN-W.

### Optimizer

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
```

Same as `tests/test_smoke_train.py` for consistency with prior art.

### Step count and measurement schedule

200 steps default, CLI-overridable. Measured **at step 0** (after the first
`loss.backward()`, before the first `opt.step()`) **and then every 10 steps**
through step 200. Rationale: step 0 already shows the ~10³–10⁴× gap
(smoke-test finding); we want to observe whether the ratio trajectory bends
upward (gates de-saturating) or stays flat (stuck).

### Measurement

At each scheduled step, after `loss.backward()` and before `opt.step()`,
under `torch.no_grad()`:

| Quantity | Definition |
|---|---|
| `grad_W` | **Mean across the 16 `LogicLayer.W` tensors** (4 blocks × 4 LGN layers) of `‖W.grad‖₂`. Mean-of-norms, not norm-of-concatenation — keeps it apples-to-apples with `grad_attn`. |
| `grad_W_per_block[i]` | Mean across the 4 `LogicLayer.W` tensors within block `i` of `‖W.grad‖₂`. List of 4 floats. |
| `grad_attn` | Mean across the 4 blocks of `‖block.attn.c_attn.weight.grad‖₂`. The canonical "normal Linear weight" comparator. |
| `grad_embed` | `‖wte.weight.grad‖₂` — single tensor. |
| `grad_lm_head` | `‖lm_head.weight.grad‖₂` — single tensor. |
| `ratio` | `grad_W / grad_attn` — the headline number. |
| `loss` | Current training loss (sanity check). |

All norms are computed in fp32 regardless of autocast dtype (cast `.grad` to
fp32 before `.norm()`). Use `torch.linalg.vector_norm` on the flattened tensor
to be explicit about the operation.

**Note on units:** `W` has shape `(N, 16)` where `N = K·d_model = 1024`, while
`attn.c_attn.weight` has shape `(3·d_model, d_model) = (384, 128)`. The two
tensors are not the same size, so the absolute value of the ratio is not by
itself meaningful — what matters is whether `ratio(last)/ratio(first) ≥ 2`,
which is invariant to constant scaling and captures gate de-saturation.

Gradients are extracted by walking `model.named_parameters()` and matching
parameter names. The expected name patterns are:

- LGN-W: `blocks.{i}.ffn.body.layers.{j}.W` (where the body is the
  `LGNBody` inside `LGNMLPBlock`).
- Attn: `blocks.{i}.attn.c_attn.weight`.
- Embed: `wte.weight`.
- Head: `lm_head.weight`.

The script asserts at startup that it finds the expected number of each
(16 W tensors, 4 attn tensors, 1 embed, 1 head). Failure to find any is a
fatal error with a clear message — defends against silent renames in
`nanolgn.gpt` / `nanolgn.lgn` / `nanolgn.lgn_mlp` in the future.

### Output

Two stdout tables, plain ASCII (no rich formatting).

**Table 1 — per-step trajectory:**

```
 step | loss  | |∇W|    | |∇attn| | |∇embed| | |∇head| | W/attn
------+-------+---------+---------+----------+---------+---------
    0 | 4.14  | 2.30e-6 | 1.51e-3 |  4.12e-4 |  3.21e-3 | 1.52e-3
   10 | 3.87  | ...     | ...     |  ...     |  ...     | ...
  ...
  200 | ...   | ...     | ...     |  ...     |  ...     | ...
```

**Table 2 — per-block `|∇W|` at first and last measurement (step 0 and the
last measured step, e.g. step 200 with defaults):**

```
 block | |∇W| @ step 0 | |∇W| @ step 200 | ratio (last/first)
-------+----------------+------------------+--------------------
     0 | 2.10e-6        | ...              | ...
     1 | 2.35e-6        | ...              | ...
     2 | 2.41e-6        | ...              | ...
     3 | 2.34e-6        | ...              | ...
```

This surfaces block-asymmetry (e.g., "only block 0 sees signal").

### Verdict heuristic

Printed after both tables:

```
=== VERDICT ===
ratio(step 0):       X.XXe-YY
ratio(last step):    X.XXe-YY
|∇W|(last step):     X.XXe-YY
LIKELY LEARNING:     <YES|NO>
```

Rule:

- **YES** iff *both*:
  - `ratio(last step) >= 2 * ratio(step 0)` (gap is closing as gates de-saturate)
  - `grad_W(last step) > 1e-7` (signal is non-trivial in absolute terms)
- **NO** otherwise.

Header comment in the script documents the heuristic's limitations
explicitly: 200 steps is short; "NO" here does not prove "fundamentally
stuck" — it could mean "needs longer," "needs a different LR," or "needs a
task with stronger long-range structure." It is a flag, not a verdict.

### CLI

```
python scripts/diagnose_lgn_grads.py [--steps N] [--seed S] [--device {cuda,cpu}]
```

Defaults:

- `--steps 200`
- `--seed 0`
- `--device` — auto-detect, prefers cuda if `torch.cuda.is_available()`.

Prints a warning if running on CPU ("POC-A shape on CPU is slow; expect ≥10
min for 200 steps. Consider running on GPU.").

### Dependencies

None new. Uses `torch`, `numpy`, and `nanolgn.*` already in `.venv`.

### Data download

None. Synthetic in-memory data only.

## What we will and won't learn

**Will learn:**

- Whether gradients reach LGN-W at all (vs. zero/NaN — sanity).
- Whether the W/attn ratio improves over the run (de-saturation) or stays
  flat (stuck near-identity).
- Whether all 4 blocks see roughly symmetric W-gradient flow, or whether
  one block dominates.

**Won't learn:**

- Whether the body learns *useful* logical features (separate follow-up).
- Whether real-TinyStories dynamics match synthetic (very likely qualitatively
  the same; the smoke-test finding holds across both).
- The right value of `residual_init_strength` for production. We only test
  the s=7.5 default.

## Success criteria for the diagnostic itself

The script is "done" when:

1. It runs to completion on a T4 in under 3 minutes.
2. Both tables are produced and human-readable.
3. The verdict line is unambiguous (one of two strings).
4. The script aborts cleanly with a clear error if the parameter-name
   walk finds the wrong number of LGN-W / attn / embed / head tensors —
   defends against silent renames.

## Out of scope

- Pytest coverage of the script. One-shot tool.
- W&B / tensorboard logging.
- Checkpoint loading. Always runs from `_init_weights` initialization.
- Comparison against MLP. The MLP path has no LGN-W, so there's nothing to
  diagnose.
- Multi-seed averaging. Single seed; user can re-run with different `--seed`
  if they want a second data point.
