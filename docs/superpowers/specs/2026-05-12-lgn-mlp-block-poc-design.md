# nano-lgn POC: replacing the nanochat MLP block with a Logic Gate Network

**Date:** 2026-05-12
**Status:** Approved, ready for implementation planning
**Scope:** Proof-of-concept only. See "Out of scope" at the bottom.

## Goal

Replace the per-block FFN ("MLP") inside a nanochat-style GPT transformer with
a **Differentiable Logic Gate Network** (LGN, Petersen et al., NeurIPS 2022 /
2024) and show that the resulting language model trains — i.e. val loss
decreases monotonically and lands within a stated tolerance of an
apples-to-apples MLP baseline.

The two reference systems are:
- **nanochat** (Karpathy, 2025): GPT-style transformer, RMSNorm, RoPE, ReLU²
  MLP block, Muon optimizer. See `https://github.com/karpathy/nanochat`.
- **Differentiable LGN**: each "neuron" is a softmax mixture over the 16
  two-input binary logic gates, parameterized so it is differentiable in
  training and reduces to a discrete logic circuit at inference. See
  `https://github.com/Felix-Petersen/difflogic` and arXiv 2210.08277 / 2411.04732.

## Approach: two-stage POC (variant "C")

| | POC-A (conservative) | POC-B (mid) |
|---|---|---|
| `d_model` | 128 | 256 |
| `n_layer` | 4 | 6 |
| `n_head` | 4 | 4 |
| `ctx_len` | 256 | 512 |
| Data | TinyStories (~470M tokens) | FineWeb-edu sample, ~500M–1B tokens |
| LGN body | K=16, L=4, N=K·d=2,048 | K=32, L=6, N=K·d=8,192 |
| Hardware | single GPU, ~30 min/run | single GPU, ~3–6 h/run |
| Steps | ~5,000 | ~20,000 |

Implement A first, drive bugs out, then promote to B by switching the config.
Same code path, no rewrite.

## Architecture

### Repository layout

```
nano-lgn/
├── nanolgn/
│   ├── __init__.py
│   ├── lgn.py            # LogicLayer + LGN body
│   ├── lgn_mlp.py        # ThermometerEncode + body + GroupSumDecode
│   ├── gates.py          # 16 binary-gate soft relaxations
│   ├── gpt.py            # minimal nanochat-style transformer, FFN-pluggable
│   └── config.py         # @dataclass TransformerCfg, LGNCfg
├── scripts/
│   ├── train.py          # single training entrypoint, takes a config name
│   └── eval.py           # val loss; optional hard-discretized eval (future)
├── configs/
│   ├── poc_a_mlp.py      # baseline:  ReLU² MLP
│   ├── poc_a_lgn.py      # LGN-MLP, conservative
│   ├── poc_b_mlp.py
│   └── poc_b_lgn.py
├── tests/
│   ├── test_gates.py
│   ├── test_lgn.py
│   ├── test_lgn_mlp.py
│   └── test_smoke_train.py
└── data/                 # tokenized shards
```

**Hard rules:**
1. The LGN-MLP module is a plain `nn.Module` with
   `forward(x: (B, T, d_model)) -> (B, T, d_model)`. Same shape contract as the
   ReLU² MLP. Fully isolable from the transformer.
2. The transformer's FFN slot is selected by config (`"mlp"` or `"lgn"`). MLP
   and LGN variants share one `Block` and one `GPT` class. No parallel forks of
   `gpt.py`.

### The LGN-MLP block (three stages)

#### Stage 1 — ThermometerEncode: `(B, T, d) → (B, T, N=K·d)`

For each scalar feature `x_i` (i=1..d) and each level k=1..K:
```
b_{i,k} = sigmoid( s_k · (x_i − θ_{i,k}) )
```

Learnable parameters per block:
- `θ ∈ ℝ^(d, K)` — per-feature, per-level threshold
- `s ∈ ℝ^K` — per-level sharpness, init = 1.0, **shared across features**

Threshold init: `θ_{i,k} = inverse_sigmoid(k / (K+1))` — spread across the input
range so all levels are active.

Param count: `d·K + K` (≈ `d·K`). For POC-A this is 2,064, dominated by `θ`.

Output is in (0,1) elementwise, packed into a per-token vector of `N = K·d`
"binary-ish" values.

#### Stage 2 — LGN body: `(B, T, N) → (B, T, N)` (width-preserving)

L stacked `LogicLayer`s. Each layer:

```
Params:   W ∈ ℝ^(N, 16)             # learnable gate-mixture logits per neuron
Buffers:  π_a, π_b ∈ ℤ^N           # frozen random connection indices in [0, N)

Forward (per token):
    a = z[..., π_a]                # gather, (B, T, N)
    b = z[..., π_b]                # gather, (B, T, N)
    p = softmax(W, dim=-1)         # (N, 16)
    out_j = Σ_g  p_{j,g} · f_g(a_j, b_j)
```

The 16 gate functions `f_g(a,b)` are the product-t-norm soft relaxations:

| g | gate | f(a,b) |
|---|---|---|
| 0 | FALSE | 0 |
| 1 | AND | a·b |
| 2 | A∧¬B | a − a·b |
| 3 | A (passthrough) | a |
| 4 | ¬A∧B | b − a·b |
| 5 | B | b |
| 6 | XOR | a + b − 2ab |
| 7 | OR | a + b − ab |
| 8 | NOR | 1 − (a + b − ab) |
| 9 | XNOR | 1 − (a + b − 2ab) |
| 10 | ¬B | 1 − b |
| 11 | A∨¬B | 1 − b + ab |
| 12 | ¬A | 1 − a |
| 13 | ¬A∨B | 1 − a + ab |
| 14 | NAND | 1 − ab |
| 15 | TRUE | 1 |

**Residual init** (2024-paper trick): init `W[:, 3] = +5.0` (gate "A" =
passthrough on slot a), all other columns = 0. Then `softmax(W)[:, 3] ≈ 0.993`,
so at init each layer is ≈ identity-on-π_a. This makes L stacked layers
trainable from scratch — without it, the 2022-paper variant struggles past
depth 4.

`π_a`, `π_b` are sampled once at module init from a seed derived from the
global config seed, and frozen. No learning on the connection table.

Body param count: `L · N · 16 = 16·L·K·d` learnable.
- POC-A (d=128, K=16, L=4): ≈ 131k params (vs ≈ 131k for ReLU² MLP at same d).
- POC-B (d=256, K=32, L=6): ≈ 786k params (vs ≈ 524k for ReLU² MLP at same d).

Within ~50% — clean enough comparison for the POC.

#### Stage 3 — GroupSumDecode: `(B, T, N) → (B, T, d)`

```
z' = z.reshape(B, T, d, K)     # contiguous groups of K
y_i = (Σ_k z'_{i,k}) / τ        # τ scalar, default τ = K → y ∈ [0,1]
y   = y − 0.5                  # zero-mean shift; no learnable params
```

The shift to zero mean matters: the residual stream is approximately zero-mean
(RMSNorm before the block scrubs scale, but the residual add still benefits
from zero-centered increments). `τ` is a config scalar; promoting it to a
learnable per-channel scalar is a one-line change if loss curves demand it.

#### Whole block

```python
def forward(self, x):              # x: (B, T, d)
    b = self.thermo_encode(x)     # (B, T, K·d)   in (0,1)
    z = self.lgn_body(b)          # (B, T, K·d)   in (0,1)
    y = self.group_sum_decode(z)  # (B, T, d)     in [-0.5, 0.5]
    return y                      # caller does residual add
```

### Transformer scaffold (`nanolgn/gpt.py`)

Minimal nanochat-style GPT.

**Kept from nanochat:**
- RMSNorm without learnable parameters (pre-norm; before attention and FFN)
- RoPE positional encoding, base θ = 10,000
- ReLU² in the **baseline** MLP block (so the baseline is honest)
- Standard causal self-attention
- Tied input embeddings / LM head

**Skipped for POC** (clear extension points, not on critical path):
- Sliding-window attention
- Value embeddings
- Per-layer learnable residual scalars
- Logit softcapping
- Q/K normalization
- Muon optimizer — plain AdamW for everything

**Pluggable FFN slot:**

```python
class Block(nn.Module):
    def __init__(self, cfg, ffn_factory):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = ffn_factory(cfg)        # MLP or LGNMLPBlock

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

`ffn_factory` is chosen by `cfg.ffn ∈ {"mlp", "lgn"}`.

### Configs

```python
# configs/poc_a_lgn.py
cfg = TransformerCfg(
    d_model=128, n_layer=4, n_head=4, ctx_len=256,
    vocab_size=50257,            # GPT-2 BPE
    ffn="lgn",
)
lgn = LGNCfg(K=16, L=4, tau=16, residual_init_strength=5.0)
```

```python
# configs/poc_b_lgn.py
cfg = TransformerCfg(
    d_model=256, n_layer=6, n_head=4, ctx_len=512,
    vocab_size=50257,
    ffn="lgn",
)
lgn = LGNCfg(K=32, L=6, tau=32, residual_init_strength=5.0)
```

`poc_a_mlp.py` / `poc_b_mlp.py` are identical to their `_lgn` siblings except
`ffn="mlp"` and `LGNCfg` is unused — they are the apples-to-apples baselines.

## Data and training

### Tokenizer and data

- **Tokenizer:** GPT-2 BPE via `tiktoken` (`gpt2`, 50,257 vocab). No tokenizer
  training step in the POC.
- **POC-A data:** TinyStories, pre-tokenized to a single binary `uint16` shard
  on disk, memmap-loaded.
- **POC-B data:** FineWeb-edu sample-10BT, take ~500M–1B tokens. Same loader,
  different shard.

Loader: a small `IterableDataset` (~30 LOC) that memmap-reads `uint16` tokens
and yields `(B, T+1)` windows at random offsets (input/target shift). No DDP,
no fancy packing.

### Optimizer and schedule

- **AdamW**, β=(0.9, 0.95), weight_decay=0.1, all parameters in one group to
  start. (A `W`-specific lr group is a documented fallback — see Risks.)
- **Cosine LR** with 200-step linear warmup.
- POC-A: peak lr = 3e-4, min lr = 3e-5, ~5,000 steps.
- POC-B: peak lr = 1e-4, min lr = 1e-5, ~20,000 steps.
- Gradient clip at 1.0.
- bf16 autocast on CUDA, fp32 weights.

### Logging (only first ~200 steps to keep overhead trivial)

Per LGN-MLP block, per step:
- `||ffn_out||` mean and max
- Gate-mixture entropy `H(softmax(W))` averaged across neurons
- Fraction of thermometer thresholds `θ` in the active range of their feature

These are the early-warning signals for the LGN block specifically.

### Reproducibility

Single global seed in the config. The connection tables `π_a`, `π_b` are
derived deterministically from that seed, so two runs of POC-A-LGN with seed=0
are bit-identical on CPU.

## Evaluation

Eval every 250 training steps on a held-out 1M-token slice. Log val loss and
val perplexity.

Two head-to-head comparisons:
1. **POC-A:** `poc_a_mlp` vs `poc_a_lgn`, same seed, same data, same steps.
2. **POC-B:** `poc_b_mlp` vs `poc_b_lgn`, same.

### Success criteria (stated upfront)

- **Must:** the LGN variant trains without diverging — val loss decreases
  monotonically over the first 1,000 steps; no NaN/Inf in forward, backward, or
  weights.
- **Strong:** LGN-A reaches val loss within **30%** of MLP-A at matched steps.
- **Stretch:** LGN-B reaches val loss within 15% of MLP-B.

The 30% bar is deliberately not 10%: replacing the FFN with an LGN is a real
architectural mismatch, not a tweak, and we want an honest first signal rather
than an aspirational one.

## Testing

Three layers.

### 1. Math unit tests (`tests/test_gates.py`)

- All 16 gate functions return the truth-table value at the four corners of
  `{0,1}²`. Catches sign errors immediately.
- Each gate is monotone in `a` and `b` where it should be.

### 2. Module invariant tests (`tests/test_lgn.py`, `tests/test_lgn_mlp.py`)

- **Shape:** `LGNMLPBlock(d=128, K=16, L=4)(torch.randn(2, 7, 128)).shape ==
  (2, 7, 128)`.
- **Finiteness:** forward + backward have no NaN/Inf on random inputs.
- **Residual init ≈ identity-on-π_a:** with `residual_init_strength=5.0` and
  one layer, the body's output is close to `gather(z, π_a)` (tol ~1e-2).
- **Gradient flow:** gradient of loss wrt `θ` (encoder) and `W` (gate mixture)
  is nonzero on a random batch.
- **Determinism:** same seed → identical `π_a`, `π_b`, identical forward.

### 3. Smoke training test (`tests/test_smoke_train.py`, on demand)

Train POC-A for 100 steps on 100k tokens; assert val loss at step 100 is
strictly less than at step 10. Catches "does it train at all" without paying
for a full run.

## Risks

Ranked by how much they could hurt.

1. **The continuous↔binary boundary may simply not pass enough information per
   token.** A standard MLP carries `d_model` reals through; we squeeze that
   into `K·d_model` near-binary values. If thermometer resolution is too low,
   every block degrades the residual stream and val loss plateaus high. K is
   a config knob; the per-block `||ffn_out||` and gate-entropy logs surface
   this early.

2. **Random pairwise connections in a fixed table.** Each gate sees only 2 of
   N inputs, randomly chosen at init. Many neurons may end up correlated /
   redundant. The user has a specific alternative LGN body in mind — that's
   the swap target after the POC works in baseline form.

3. **Soft-mixture training, never hard at inference.** The whole pitch of LGNs
   is fast discrete logic at inference; the POC sidesteps that entirely. "POC
   trains" does not imply "discretized network would work." Followup project.

4. **GroupSum centering by `−0.5` is a guess.** If residual stream drift shows
   up in logs, switching to a learned per-channel bias is a one-line change.

5. **AdamW for a softmax-parameterized gate distribution may be the wrong
   optimizer.** difflogic recommends Adam at lr=1e-2 (10× normal). If the
   gate-mixture entropy is flat in the first 1,000 steps, try a higher lr on
   `W` specifically via a param-group split before assuming the architecture
   is broken.

6. **Implementation-bug risk.** The connection-table gather and the contiguous
   reshape `(B, T, N) → (B, T, d, K)` in GroupSum are the kind of code that's
   easy to get silently wrong. The shape and residual-init tests are
   specifically there to catch this.

## Out of scope

Explicit, so we don't drift:
- Multi-GPU / DDP training
- Hard-discretized inference (the actual fast-logic-circuit win)
- Muon optimizer
- Convolutional / tree-based LGN variants
- Tokenizer training
- Sliding-window attention, value embeddings, Q/K norm, logit softcapping,
  per-layer residual scalars
- Anything past POC-B (full nanochat speedrun parity, RL, SFT, web UI)

## References

- Karpathy, *nanochat*. `https://github.com/karpathy/nanochat`
- Karpathy, *nanochat discussion 481* (architecture writeup).
  `https://github.com/karpathy/nanochat/discussions/481`
- Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022.
  arXiv `2210.08277`. `https://github.com/Felix-Petersen/difflogic`
- Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS
  2024. arXiv `2411.04732`. (Source of the residual-init trick.)
