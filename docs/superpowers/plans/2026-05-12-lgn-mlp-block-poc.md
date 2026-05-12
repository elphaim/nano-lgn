# nano-lgn POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-block FFN inside a nanochat-style GPT transformer with a Differentiable Logic Gate Network (LGN) and show it trains. Two-stage POC (POC-A: d=128, 4 layers; POC-B: d=256, 6 layers).

**Architecture:** Each transformer block's FFN slot is a pluggable factory (`"mlp"` or `"lgn"`). The LGN-MLP block is `ThermometerEncode → stacked LogicLayers → GroupSumDecode`, all in `[0,1]` internally, returning a centered `d_model` vector for the residual add. LGN body uses fixed random pairwise connections, softmax mixture over 16 binary gates, residual-init (gate "A" passthrough at start). Training is soft-mixture only.

**Tech Stack:** Python 3.11+, PyTorch 2.x (uses `F.scaled_dot_product_attention`), tiktoken (GPT-2 BPE), HuggingFace `datasets` (TinyStories download), pytest. Single GPU, plain AdamW.

**Spec:** `docs/superpowers/specs/2026-05-12-lgn-mlp-block-poc-design.md` (read this first if you have not).

---

## Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `nanolgn/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "nanolgn"
version = "0.0.1"
description = "Replacing nanochat's MLP block with a Differentiable Logic Gate Network."
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "tiktoken>=0.7",
    "numpy>=1.26",
    "datasets>=2.18",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-xdist>=3.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nanolgn"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
*.egg-info/
build/
dist/
data/*.bin
data/*.npy
runs/
.venv/
.vscode/
.idea/
```

- [ ] **Step 3: Write `README.md` skeleton**

```markdown
# nano-lgn

Replacing the MLP block in [nanochat](https://github.com/karpathy/nanochat) with a
Differentiable Logic Gate Network ([Petersen et al.](https://arxiv.org/abs/2210.08277)).

## Status

Proof-of-concept. See `docs/superpowers/specs/2026-05-12-lgn-mlp-block-poc-design.md`
for the spec and `docs/superpowers/plans/2026-05-12-lgn-mlp-block-poc.md` for the
plan.

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

Full data prep, training, and eval instructions are added at the end of the
implementation plan.
```

- [ ] **Step 4: Create empty package init files**

```python
# nanolgn/__init__.py
"""nano-lgn: a Logic Gate Network in place of nanochat's MLP block."""
```

```python
# tests/__init__.py
```

- [ ] **Step 5: Verify the package installs and pytest runs**

Run: `pip install -e ".[dev]" && pytest -q`
Expected: `no tests ran` exit code 5 (acceptable; means the harness works).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md nanolgn/__init__.py tests/__init__.py
git commit -m "chore: project bootstrap (pyproject, gitignore, package skeleton)"
```

---

## Task 2: 16 binary-gate soft relaxations (`gates.py`)

**Files:**
- Create: `nanolgn/gates.py`
- Test: `tests/test_gates.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gates.py
import torch
import pytest
from nanolgn.gates import GATE_NAMES, gate, GATE_FNS

CORNERS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

# Truth tables indexed by (a, b) ∈ {0,1}².
# Order MUST match GATE_NAMES below.
TRUTH = {
    "FALSE":   [0, 0, 0, 0],
    "AND":     [0, 0, 0, 1],
    "A_AND_NB":[0, 0, 1, 0],
    "A":       [0, 0, 1, 1],
    "NA_AND_B":[0, 1, 0, 0],
    "B":       [0, 1, 0, 1],
    "XOR":     [0, 1, 1, 0],
    "OR":      [0, 1, 1, 1],
    "NOR":     [1, 0, 0, 0],
    "XNOR":    [1, 0, 0, 1],
    "NB":      [1, 0, 1, 0],
    "A_OR_NB": [1, 0, 1, 1],
    "NA":      [1, 1, 0, 0],
    "NA_OR_B": [1, 1, 0, 1],
    "NAND":    [1, 1, 1, 0],
    "TRUE":    [1, 1, 1, 1],
}

def test_gate_names_are_16_unique():
    assert len(GATE_NAMES) == 16
    assert len(set(GATE_NAMES)) == 16

def test_truth_table_matches_at_corners():
    for g_idx, name in enumerate(GATE_NAMES):
        expected = TRUTH[name]
        for (a_val, b_val), exp in zip(CORNERS, expected):
            a = torch.tensor(a_val)
            b = torch.tensor(b_val)
            got = GATE_FNS[g_idx](a, b).item()
            assert abs(got - exp) < 1e-6, f"{name}({a_val},{b_val}) = {got}, expected {exp}"

def test_gate_dispatch_function_matches_index():
    a = torch.rand(4)
    b = torch.rand(4)
    for g_idx in range(16):
        got = gate(g_idx, a, b)
        expected = GATE_FNS[g_idx](a, b)
        assert torch.allclose(got, expected)

def test_gates_are_vectorized():
    a = torch.rand(3, 5)
    b = torch.rand(3, 5)
    for g_idx in range(16):
        out = GATE_FNS[g_idx](a, b)
        assert out.shape == (3, 5)
        assert torch.all(out >= 0.0 - 1e-6)
        assert torch.all(out <= 1.0 + 1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gates.py -v`
Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement `gates.py`**

```python
# nanolgn/gates.py
"""Soft (product-t-norm) relaxations of the 16 binary logic gates.

Each gate is a function (a, b) -> out, where a and b are tensors with values
in [0, 1]. Outputs are in [0, 1]. Gates are listed in the same order as the
softmax weight columns in LogicLayer (gate index = column index).

Ordering is the standard 4-bit truth-table ordering on (a, b) ∈ {0,1}² read as
(b, a) → bit:
    index 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
which yields gate index 3 = "A" (passthrough on a) — used by residual init.
"""
from __future__ import annotations
import torch
from torch import Tensor

GATE_NAMES = (
    "FALSE",     # 0
    "AND",       # 1
    "A_AND_NB",  # 2
    "A",         # 3   <-- residual-init target (passthrough on a)
    "NA_AND_B",  # 4
    "B",         # 5
    "XOR",       # 6
    "OR",        # 7
    "NOR",       # 8
    "XNOR",      # 9
    "NB",        # 10
    "A_OR_NB",   # 11
    "NA",        # 12
    "NA_OR_B",   # 13
    "NAND",      # 14
    "TRUE",      # 15
)

def _g_false(a: Tensor, b: Tensor) -> Tensor:    return torch.zeros_like(a)
def _g_and(a, b):                                return a * b
def _g_a_and_nb(a, b):                           return a - a * b
def _g_a(a, b):                                  return a
def _g_na_and_b(a, b):                           return b - a * b
def _g_b(a, b):                                  return b
def _g_xor(a, b):                                return a + b - 2.0 * a * b
def _g_or(a, b):                                 return a + b - a * b
def _g_nor(a, b):                                return 1.0 - (a + b - a * b)
def _g_xnor(a, b):                               return 1.0 - (a + b - 2.0 * a * b)
def _g_nb(a, b):                                 return 1.0 - b
def _g_a_or_nb(a, b):                            return 1.0 - b + a * b
def _g_na(a, b):                                 return 1.0 - a
def _g_na_or_b(a, b):                            return 1.0 - a + a * b
def _g_nand(a, b):                               return 1.0 - a * b
def _g_true(a, b):                               return torch.ones_like(a)

GATE_FNS = (
    _g_false, _g_and, _g_a_and_nb, _g_a,
    _g_na_and_b, _g_b, _g_xor, _g_or,
    _g_nor, _g_xnor, _g_nb, _g_a_or_nb,
    _g_na, _g_na_or_b, _g_nand, _g_true,
)

GATE_A_INDEX = 3  # passthrough on input a — used by residual-init in LogicLayer.

def gate(idx: int, a: Tensor, b: Tensor) -> Tensor:
    """Dispatch wrapper: gate(idx, a, b) = GATE_FNS[idx](a, b)."""
    return GATE_FNS[idx](a, b)


def all_gates_stack(a: Tensor, b: Tensor) -> Tensor:
    """Stack all 16 gate outputs along a new last dim. Shape: (..., 16)."""
    return torch.stack([fn(a, b) for fn in GATE_FNS], dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gates.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gates.py tests/test_gates.py
git commit -m "feat(gates): 16 soft binary-gate relaxations + truth-table tests"
```

---

## Task 3: `LogicLayer` (one LGN layer, `lgn.py`)

**Files:**
- Create: `nanolgn/lgn.py`
- Test: `tests/test_lgn.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lgn.py
import torch
import pytest
from nanolgn.lgn import LogicLayer

def _make(n=64, seed=0, residual_init_strength=7.5):
    return LogicLayer(n=n, seed=seed, residual_init_strength=residual_init_strength)

def test_shape_preserves_width_and_batch():
    layer = _make(n=64)
    x = torch.rand(2, 7, 64)
    y = layer(x)
    assert y.shape == (2, 7, 64)

def test_output_in_unit_interval():
    layer = _make(n=128)
    x = torch.rand(4, 5, 128)
    y = layer(x)
    assert torch.all(y >= -1e-5)
    assert torch.all(y <= 1.0 + 1e-5)

def test_finite_forward_and_backward():
    layer = _make(n=64)
    x = torch.rand(2, 3, 64, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert torch.isfinite(y)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.W.grad).all()

def test_residual_init_is_approximately_identity_on_pi_a():
    n = 256
    layer = _make(n=n, residual_init_strength=7.5)
    x = torch.rand(8, n)
    y = layer(x)
    expected = x[..., layer.pi_a]  # passthrough on input slot a
    # softmax([7.5, 0, ..., 0])[A] = e^7.5 / (e^7.5 + 15)
    #                              ≈ 1808.04 / 1823.04 ≈ 0.9918
    # leakage ≈ 0.0082 spread across 15 gates; max elementwise error ≲ 0.0082.
    assert torch.allclose(y, expected, atol=2e-2)

def test_determinism_same_seed_same_connections():
    a = _make(n=64, seed=42)
    b = _make(n=64, seed=42)
    assert torch.equal(a.pi_a, b.pi_a)
    assert torch.equal(a.pi_b, b.pi_b)

def test_different_seeds_give_different_connections():
    a = _make(n=64, seed=0)
    b = _make(n=64, seed=1)
    assert not torch.equal(a.pi_a, b.pi_a)

def test_W_is_a_parameter_pi_are_buffers():
    layer = _make(n=64)
    param_names = {name for name, _ in layer.named_parameters()}
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert "W" in param_names
    assert "pi_a" in buffer_names
    assert "pi_b" in buffer_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lgn.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `LogicLayer`**

```python
# nanolgn/lgn.py
"""LogicLayer: one layer of a Differentiable Logic Gate Network.

Each of N output neurons is a softmax-mixture over the 16 binary gates,
applied to two fixed-random inputs (pi_a[j], pi_b[j]) drawn from the previous
layer. Residual init makes the layer ≈ identity-on-pi_a at t=0 so deep stacks
train.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor

from .gates import all_gates_stack, GATE_A_INDEX


class LogicLayer(nn.Module):
    """N -> N width-preserving differentiable logic-gate layer.

    Args:
        n: input/output width.
        seed: deterministic seed for the random connection table.
        residual_init_strength: logit applied to gate "A" (passthrough on a)
            at init. softmax([s, 0, ..., 0])[A] = e^s / (e^s + 15);
            s=7.5 → ≈ 0.9918 (leakage ≈ 0.0082 across the other 15 gates).
    """

    def __init__(self, n: int, seed: int, residual_init_strength: float = 7.5):
        super().__init__()
        self.n = n
        self.residual_init_strength = float(residual_init_strength)

        # Fixed random connections, deterministic in seed.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pi_a = torch.randint(0, n, (n,), generator=gen, dtype=torch.long)
        pi_b = torch.randint(0, n, (n,), generator=gen, dtype=torch.long)
        self.register_buffer("pi_a", pi_a)
        self.register_buffer("pi_b", pi_b)

        # Gate-mixture logits, one row per output neuron. Residual init: gate
        # "A" gets a strong logit; the rest stay at 0.
        W = torch.zeros(n, 16)
        W[:, GATE_A_INDEX] = self.residual_init_strength
        self.W = nn.Parameter(W)

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., n) in [0,1]. Returns (..., n) in [0,1]."""
        a = x.index_select(-1, self.pi_a)        # (..., n)
        b = x.index_select(-1, self.pi_b)        # (..., n)
        gates = all_gates_stack(a, b)            # (..., n, 16)
        p = torch.softmax(self.W, dim=-1)        # (n, 16)
        # Broadcast p across leading dims; sum over the gate dim.
        out = (gates * p).sum(dim=-1)            # (..., n)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lgn.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/lgn.py tests/test_lgn.py
git commit -m "feat(lgn): LogicLayer with residual init and frozen random connections"
```

---

## Task 4: `LGNBody` (stack of `LogicLayer`s)

**Files:**
- Modify: `nanolgn/lgn.py` (append)
- Modify: `tests/test_lgn.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lgn.py`:

```python
from nanolgn.lgn import LGNBody

def test_lgn_body_shape():
    body = LGNBody(n=128, depth=4, seed=0)
    x = torch.rand(2, 3, 128)
    assert body(x).shape == (2, 3, 128)

def test_lgn_body_residual_init_is_approx_identity_chain():
    n, L = 128, 6
    body = LGNBody(n=n, depth=L, seed=0, residual_init_strength=7.5)
    x = torch.rand(4, n)
    y = body(x)
    # Each layer is ≈ passthrough on its own pi_a (different per layer).
    # The composition is the chained gather. Walk it explicitly.
    z = x
    for layer in body.layers:
        z = z[..., layer.pi_a]
    # At s=7.5 per-layer leakage ≲ 0.0082; 6 layers compound it. atol=5e-2
    # leaves headroom while still meaningfully constraining drift.
    assert torch.allclose(y, z, atol=5e-2)

def test_lgn_body_layer_seeds_differ():
    body = LGNBody(n=64, depth=3, seed=0)
    pis = [layer.pi_a for layer in body.layers]
    assert not torch.equal(pis[0], pis[1])
    assert not torch.equal(pis[1], pis[2])

def test_lgn_body_gradients_flow_to_all_W():
    body = LGNBody(n=64, depth=3, seed=0)
    x = torch.rand(2, 64)
    body(x).sum().backward()
    for layer in body.layers:
        assert layer.W.grad is not None
        assert torch.isfinite(layer.W.grad).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lgn.py::test_lgn_body_shape -v`
Expected: ImportError on `LGNBody`.

- [ ] **Step 3: Implement `LGNBody`**

Append to `nanolgn/lgn.py`:

```python
class LGNBody(nn.Module):
    """Stack of L width-preserving LogicLayers.

    Each layer has its own connection table, deterministically derived from
    the body's seed (layer i uses seed = base_seed * 1_000_003 + i).
    """

    def __init__(
        self,
        n: int,
        depth: int,
        seed: int,
        residual_init_strength: float = 7.5,
    ):
        super().__init__()
        self.n = n
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                LogicLayer(
                    n=n,
                    seed=seed * 1_000_003 + i,
                    residual_init_strength=residual_init_strength,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lgn.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/lgn.py tests/test_lgn.py
git commit -m "feat(lgn): LGNBody stacks L LogicLayers with per-layer seeds"
```

---

## Task 5: `ThermometerEncode` (`lgn_mlp.py`)

**Files:**
- Create: `nanolgn/lgn_mlp.py`
- Create: `tests/test_lgn_mlp.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lgn_mlp.py
import torch
import pytest
from nanolgn.lgn_mlp import ThermometerEncode

def test_thermo_shape():
    enc = ThermometerEncode(d_model=128, k=16)
    x = torch.randn(2, 7, 128)
    y = enc(x)
    assert y.shape == (2, 7, 16 * 128)

def test_thermo_output_in_unit_interval():
    enc = ThermometerEncode(d_model=64, k=8)
    x = torch.randn(4, 3, 64) * 5.0  # large-ish range
    y = enc(x)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)

def test_thermo_thresholds_initialized_spread():
    enc = ThermometerEncode(d_model=4, k=8)
    # All features share the same K thresholds at init, sorted ascending
    # via inverse_sigmoid(k/(K+1)) for k=1..K.
    theta_per_feature = enc.theta[0]                   # (K,)
    sorted_theta, _ = torch.sort(theta_per_feature)
    assert torch.allclose(theta_per_feature, sorted_theta)
    # Sigmoid of the thresholds should be evenly spread between 0 and 1.
    sig = torch.sigmoid(theta_per_feature)
    diffs = sig[1:] - sig[:-1]
    assert torch.all(diffs > 0)
    assert torch.allclose(diffs, diffs[0] * torch.ones_like(diffs), atol=1e-5)

def test_thermo_finite_backward():
    enc = ThermometerEncode(d_model=32, k=8)
    x = torch.randn(2, 32, requires_grad=True)
    enc(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(enc.theta.grad).all()
    assert torch.isfinite(enc.s.grad).all()

def test_thermo_param_count_is_dK_plus_K():
    d, k = 128, 16
    enc = ThermometerEncode(d_model=d, k=k)
    n = sum(p.numel() for p in enc.parameters())
    assert n == d * k + k
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lgn_mlp.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `ThermometerEncode`**

```python
# nanolgn/lgn_mlp.py
"""LGN-MLP block: ThermometerEncode -> LGNBody -> GroupSumDecode.

Drop-in replacement for the per-block ReLU² FFN.
"""
from __future__ import annotations
import math
import torch
from torch import nn, Tensor

from .lgn import LGNBody


def _inv_sigmoid(p: float) -> float:
    return math.log(p / (1.0 - p))


class ThermometerEncode(nn.Module):
    """(B, T, d) -> (B, T, K*d) in (0, 1).

    For each scalar feature x_i and level k:
        b_{i,k} = sigmoid(s_k * (x_i - theta_{i,k}))

    Params:
        theta ∈ ℝ^(d, K): per-feature, per-level threshold.
        s ∈ ℝ^K: per-level sharpness (shared across features).
    """

    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.k = k

        # Init thresholds: same K levels for every feature, spread on the
        # input axis via inverse-sigmoid of evenly-spaced probabilities.
        levels = torch.tensor(
            [_inv_sigmoid((i + 1) / (k + 1)) for i in range(k)]
        )                                              # (K,)
        theta = levels.unsqueeze(0).expand(d_model, k).clone()  # (d, K)
        self.theta = nn.Parameter(theta)

        # Sharpness shared across features, init = 1.
        self.s = nn.Parameter(torch.ones(k))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d). Broadcast to (..., d, K), then flatten last two dims.
        x_e = x.unsqueeze(-1)                          # (..., d, 1)
        # theta: (d, K) → broadcasts cleanly.
        # s: (K,) → broadcasts cleanly.
        b = torch.sigmoid(self.s * (x_e - self.theta))  # (..., d, K)
        return b.flatten(-2)                           # (..., d*K)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lgn_mlp.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/lgn_mlp.py tests/test_lgn_mlp.py
git commit -m "feat(lgn_mlp): ThermometerEncode (continuous → [0,1] expansion)"
```

---

## Task 6: `GroupSumDecode`

**Files:**
- Modify: `nanolgn/lgn_mlp.py` (append)
- Modify: `tests/test_lgn_mlp.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lgn_mlp.py`:

```python
from nanolgn.lgn_mlp import GroupSumDecode

def test_decode_shape_and_grouping():
    dec = GroupSumDecode(d_model=4, k=3, tau=3.0)
    # Build z so each group sums to a known value:
    # group i gets values [i, i, i] → sum = 3i → /tau=3 → i → −0.5.
    z = torch.tensor([
        [0., 0., 0.,  1., 1., 1.,  2., 2., 2.,  3., 3., 3.],
    ])  # (1, 12) = (1, d*K)
    y = dec(z)
    assert y.shape == (1, 4)
    expected = torch.tensor([[-0.5, 0.5, 1.5, 2.5]])
    assert torch.allclose(y, expected)

def test_decode_default_tau_centers_at_zero_for_uniform_half():
    dec = GroupSumDecode(d_model=8, k=16, tau=16.0)
    # Inputs uniformly = 0.5 → group sum = 8 → /tau = 0.5 → −0.5 → 0.
    z = torch.full((2, 5, 8 * 16), 0.5)
    y = dec(z)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

def test_decode_no_learnable_params():
    dec = GroupSumDecode(d_model=8, k=16, tau=16.0)
    assert sum(p.numel() for p in dec.parameters()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lgn_mlp.py::test_decode_shape_and_grouping -v`
Expected: ImportError on `GroupSumDecode`.

- [ ] **Step 3: Implement `GroupSumDecode`**

Append to `nanolgn/lgn_mlp.py`:

```python
class GroupSumDecode(nn.Module):
    """(B, T, K*d) -> (B, T, d) via contiguous group-sum, scaled and centered.

    Reshape last dim as (d, K), sum over K, divide by tau, subtract 0.5.
    Default tau=K → outputs in [-0.5, 0.5].
    """

    def __init__(self, d_model: int, k: int, tau: float):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.tau = float(tau)

    def forward(self, z: Tensor) -> Tensor:
        # z: (..., d*K)
        leading = z.shape[:-1]
        z2 = z.reshape(*leading, self.d_model, self.k)  # (..., d, K)
        return z2.sum(dim=-1) / self.tau - 0.5
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lgn_mlp.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/lgn_mlp.py tests/test_lgn_mlp.py
git commit -m "feat(lgn_mlp): GroupSumDecode (LGN [0,1] outputs → centered d_model)"
```

---

## Task 7: `LGNMLPBlock`

**Files:**
- Modify: `nanolgn/lgn_mlp.py` (append)
- Modify: `tests/test_lgn_mlp.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lgn_mlp.py`:

```python
from nanolgn.lgn_mlp import LGNMLPBlock

def test_block_shape_contract_matches_mlp_slot():
    block = LGNMLPBlock(d_model=128, k=16, depth=4, tau=16.0, seed=0)
    x = torch.randn(2, 7, 128)
    y = block(x)
    assert y.shape == (2, 7, 128)

def test_block_finite_forward_backward():
    block = LGNMLPBlock(d_model=64, k=8, depth=3, tau=8.0, seed=0)
    x = torch.randn(2, 5, 64, requires_grad=True)
    block(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in block.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

def test_block_output_is_centered_at_init_for_zero_input():
    # With x=0, sigmoid output ≈ sigmoid(-theta) ≈ uniform spread; under
    # residual init each LGNlayer ≈ passthrough; GroupSum then /K and shifts
    # by -0.5. Just check finiteness and bounded range, not exact zero.
    block = LGNMLPBlock(d_model=32, k=8, depth=2, tau=8.0, seed=0)
    x = torch.zeros(1, 4, 32)
    y = block(x)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lgn_mlp.py::test_block_shape_contract_matches_mlp_slot -v`
Expected: ImportError on `LGNMLPBlock`.

- [ ] **Step 3: Implement `LGNMLPBlock`**

Append to `nanolgn/lgn_mlp.py`:

```python
class LGNMLPBlock(nn.Module):
    """Drop-in replacement for the per-block FFN: continuous → LGN → continuous.

    Same forward shape contract as a standard MLP: (B, T, d) -> (B, T, d).
    The caller (Block) does the residual add.
    """

    def __init__(
        self,
        d_model: int,
        k: int,
        depth: int,
        tau: float,
        seed: int,
        residual_init_strength: float = 5.0,
    ):
        super().__init__()
        self.encode = ThermometerEncode(d_model=d_model, k=k)
        self.body = LGNBody(
            n=k * d_model,
            depth=depth,
            seed=seed,
            residual_init_strength=residual_init_strength,
        )
        self.decode = GroupSumDecode(d_model=d_model, k=k, tau=tau)

    def forward(self, x: Tensor) -> Tensor:
        b = self.encode(x)         # (B, T, K*d) in (0,1)
        z = self.body(b)           # (B, T, K*d) in (0,1)
        y = self.decode(z)         # (B, T, d)   in [-0.5, 0.5]
        return y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lgn_mlp.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/lgn_mlp.py tests/test_lgn_mlp.py
git commit -m "feat(lgn_mlp): LGNMLPBlock end-to-end FFN replacement"
```

---

## Task 8: `RMSNorm` (in `gpt.py`)

**Files:**
- Create: `nanolgn/gpt.py`
- Create: `tests/test_gpt.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gpt.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpt.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `RMSNorm`**

```python
# nanolgn/gpt.py
"""Minimal nanochat-style GPT with a pluggable FFN slot."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with NO learnable parameters (nanochat style)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.d = d
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d). Normalize so RMS over last dim == 1.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpt.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gpt.py tests/test_gpt.py
git commit -m "feat(gpt): RMSNorm (no learnable params, nanochat style)"
```

---

## Task 9: RoPE (rotary position embeddings)

**Files:**
- Modify: `nanolgn/gpt.py` (append)
- Modify: `tests/test_gpt.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpt.py`:

```python
from nanolgn.gpt import precompute_rope, apply_rope

def test_rope_shapes():
    cos, sin = precompute_rope(dim=32, max_len=128, base=10000.0)
    assert cos.shape == (128, 32 // 2)
    assert sin.shape == (128, 32 // 2)

def test_rope_application_preserves_norm():
    head_dim = 32
    cos, sin = precompute_rope(dim=head_dim, max_len=64, base=10000.0)
    x = torch.randn(2, 4, 7, head_dim)  # (B, H, T, D)
    y = apply_rope(x, cos, sin)
    # Rotation preserves L2 norm in each (even, odd) pair.
    assert torch.allclose(y.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-5)

def test_rope_zero_position_is_identity():
    head_dim = 16
    cos, sin = precompute_rope(dim=head_dim, max_len=4, base=10000.0)
    x = torch.randn(1, 1, 1, head_dim)   # one token at position 0
    y = apply_rope(x, cos[:1], sin[:1])
    assert torch.allclose(y, x, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpt.py -v`
Expected: ImportError on `precompute_rope`.

- [ ] **Step 3: Implement RoPE**

Append to `nanolgn/gpt.py`:

```python
def precompute_rope(dim: int, max_len: int, base: float = 10000.0):
    """Precompute cos/sin tables for RoPE. Returns (cos, sin) of shape (max_len, dim/2)."""
    assert dim % 2 == 0, "RoPE dim must be even"
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # (max_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x: (..., T, dim). cos, sin: (T, dim/2)."""
    # Split into pairs: (..., T, dim) → (..., T, half, 2)
    leading = x.shape[:-1]
    x2 = x.reshape(*leading[:-1], leading[-1], -1, 2)
    x_e = x2[..., 0]
    x_o = x2[..., 1]
    # Broadcast cos/sin: (T, half) over (..., T, half).
    out_e = x_e * cos - x_o * sin
    out_o = x_e * sin + x_o * cos
    out = torch.stack((out_e, out_o), dim=-1)
    return out.reshape(*leading, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpt.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gpt.py tests/test_gpt.py
git commit -m "feat(gpt): RoPE precompute + apply"
```

---

## Task 10: `CausalSelfAttention`

**Files:**
- Modify: `nanolgn/gpt.py` (append)
- Modify: `tests/test_gpt.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpt.py`:

```python
from nanolgn.gpt import CausalSelfAttention

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpt.py::test_attention_shape -v`
Expected: ImportError.

- [ ] **Step 3: Implement `CausalSelfAttention`**

Append to `nanolgn/gpt.py`:

```python
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, no biases.

    Uses F.scaled_dot_product_attention with is_causal=True (Flash kernel
    when available).
    """

    def __init__(self, d_model: int, n_head: int, ctx_len: int, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.ctx_len = ctx_len
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        cos, sin = precompute_rope(self.head_dim, ctx_len, base=rope_base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        assert T <= self.ctx_len, f"seq len {T} exceeds ctx_len {self.ctx_len}"
        qkv = self.qkv(x)                                   # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, T, D) → (B, H, T, Dh)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)          # (B, T, D)
        return self.proj(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpt.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gpt.py tests/test_gpt.py
git commit -m "feat(gpt): causal self-attention with RoPE (no biases, SDPA)"
```

---

## Task 11: ReLU² MLP (the baseline FFN)

**Files:**
- Modify: `nanolgn/gpt.py` (append)
- Modify: `tests/test_gpt.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpt.py`:

```python
from nanolgn.gpt import ReLU2MLP

def test_relu2_mlp_shape():
    mlp = ReLU2MLP(d_model=128, mult=4)
    x = torch.randn(2, 7, 128)
    assert mlp(x).shape == (2, 7, 128)

def test_relu2_mlp_param_count():
    mlp = ReLU2MLP(d_model=128, mult=4)
    expected = 128 * 512 + 512 * 128   # 2 linear, no bias
    assert sum(p.numel() for p in mlp.parameters()) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpt.py::test_relu2_mlp_shape -v`
Expected: ImportError on `ReLU2MLP`.

- [ ] **Step 3: Implement `ReLU2MLP`**

Append to `nanolgn/gpt.py`:

```python
class ReLU2MLP(nn.Module):
    """Standard nanochat MLP: Linear(d, m*d) -> ReLU² -> Linear(m*d, d).

    No biases, no learnable scale.
    """

    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.up = nn.Linear(d_model, mult * d_model, bias=False)
        self.down = nn.Linear(mult * d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.up(x)
        h = F.relu(h).square()
        return self.down(h)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpt.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gpt.py tests/test_gpt.py
git commit -m "feat(gpt): ReLU² MLP baseline FFN"
```

---

## Task 12: `Block` (pre-norm with pluggable FFN)

**Files:**
- Modify: `nanolgn/gpt.py` (append)
- Modify: `tests/test_gpt.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpt.py`:

```python
from nanolgn.gpt import Block

def test_block_shape_with_mlp_factory():
    cfg = type("C", (), dict(d_model=64, n_head=4, ctx_len=16))
    factory = lambda c: ReLU2MLP(c.d_model, mult=4)
    block = Block(cfg=cfg, ffn_factory=factory)
    x = torch.randn(2, 8, 64)
    assert block(x).shape == (2, 8, 64)

def test_block_residual_add_with_lgn_factory():
    from nanolgn.lgn_mlp import LGNMLPBlock
    cfg = type("C", (), dict(d_model=32, n_head=2, ctx_len=16))
    factory = lambda c: LGNMLPBlock(d_model=c.d_model, k=8, depth=2, tau=8.0, seed=0)
    block = Block(cfg=cfg, ffn_factory=factory)
    x = torch.randn(2, 4, 32)
    y = block(x)
    assert y.shape == (2, 4, 32)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpt.py::test_block_shape_with_mlp_factory -v`
Expected: ImportError on `Block`.

- [ ] **Step 3: Implement `Block`**

Append to `nanolgn/gpt.py`:

```python
class Block(nn.Module):
    """Pre-norm transformer block with a pluggable FFN factory."""

    def __init__(self, cfg, ffn_factory: Callable):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model, n_head=cfg.n_head, ctx_len=cfg.ctx_len
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = ffn_factory(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpt.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gpt.py tests/test_gpt.py
git commit -m "feat(gpt): pre-norm Block with pluggable FFN factory"
```

---

## Task 13: `GPT` model

**Files:**
- Modify: `nanolgn/gpt.py` (append)
- Modify: `tests/test_gpt.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpt.py`:

```python
from nanolgn.gpt import GPT

class _CfgMLP:
    d_model = 32
    n_layer = 2
    n_head = 2
    ctx_len = 16
    vocab_size = 257

def test_gpt_forward_shape():
    cfg = _CfgMLP()
    factory = lambda c: ReLU2MLP(c.d_model, mult=4)
    gpt = GPT(cfg, ffn_factory=factory)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = gpt(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)

def test_gpt_loss_when_targets_provided():
    cfg = _CfgMLP()
    factory = lambda c: ReLU2MLP(c.d_model, mult=4)
    gpt = GPT(cfg, ffn_factory=factory)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = gpt(idx, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_gpt_is_causal_end_to_end():
    cfg = _CfgMLP()
    torch.manual_seed(0)
    factory = lambda c: ReLU2MLP(c.d_model, mult=4)
    gpt = GPT(cfg, ffn_factory=factory)
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    logits1 = gpt(idx)
    idx2 = idx.clone()
    idx2[:, 5:] = torch.randint(0, cfg.vocab_size, (1, 3))
    logits2 = gpt(idx2)
    assert torch.allclose(logits1[:, :5], logits2[:, :5], atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gpt.py::test_gpt_forward_shape -v`
Expected: ImportError on `GPT`.

- [ ] **Step 3: Implement `GPT`**

Append to `nanolgn/gpt.py`:

```python
class GPT(nn.Module):
    """Minimal nanochat-style GPT.

    - Token embedding (no learned positional; RoPE inside attention).
    - N pre-norm Blocks with pluggable FFN.
    - RMSNorm before LM head.
    - LM head tied to embedding weights.
    """

    def __init__(self, cfg, ffn_factory: Callable):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg, ffn_factory=ffn_factory) for _ in range(cfg.n_layer)]
        )
        self.norm_out = RMSNorm(cfg.d_model)
        # Untied LM head; weight-tying optional. nanochat uses untied.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        x = self.tok_emb(idx)                              # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        logits = self.lm_head(x)                           # (B, T, V)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpt.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/gpt.py tests/test_gpt.py
git commit -m "feat(gpt): GPT model with pluggable FFN, end-to-end causal"
```

---

## Task 14: Config dataclasses + factory dispatch

**Files:**
- Create: `nanolgn/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
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
    lgn = LGNCfg(K=8, L=2, tau=8.0, residual_init_strength=5.0)
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
    lgn = LGNCfg(K=4, L=1, tau=4.0, residual_init_strength=5.0)
    factory = make_ffn_factory(cfg, lgn=lgn)
    gpt = GPT(cfg, ffn_factory=factory)
    pi0 = gpt.blocks[0].ffn.body.layers[0].pi_a
    pi1 = gpt.blocks[1].ffn.body.layers[0].pi_a
    pi2 = gpt.blocks[2].ffn.body.layers[0].pi_a
    assert not torch.equal(pi0, pi1)
    assert not torch.equal(pi1, pi2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `config.py`**

```python
# nanolgn/config.py
"""Config dataclasses + FFN factory dispatch.

A config file under `configs/` defines a module-level `cfg: TransformerCfg`
and (for LGN runs) `lgn: LGNCfg`. The training script imports these by name.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class TransformerCfg:
    d_model: int
    n_layer: int
    n_head: int
    ctx_len: int
    vocab_size: int
    ffn: str                    # "mlp" or "lgn"
    seed: int = 0


@dataclass
class LGNCfg:
    K: int
    L: int
    tau: float
    residual_init_strength: float = 5.0


def make_ffn_factory(
    cfg: TransformerCfg,
    lgn: Optional[LGNCfg],
) -> Callable:
    """Return a callable (block_idx, cfg) -> nn.Module FFN.

    To keep the factory signature shape-compatible with `Block.__init__`'s
    `ffn_factory(cfg)` call, we close over `block_idx` via a counter.
    """
    if cfg.ffn == "mlp":
        from .gpt import ReLU2MLP
        def factory(_cfg) -> "torch.nn.Module":
            return ReLU2MLP(d_model=_cfg.d_model, mult=4)
        return factory

    if cfg.ffn == "lgn":
        if lgn is None:
            raise ValueError("ffn='lgn' requires an LGNCfg argument")
        from .lgn_mlp import LGNMLPBlock
        # Each Block call gets a distinct seed via a closure-captured counter.
        counter = {"i": 0}
        base_seed = cfg.seed
        def factory(_cfg) -> "torch.nn.Module":
            i = counter["i"]
            counter["i"] += 1
            return LGNMLPBlock(
                d_model=_cfg.d_model,
                k=lgn.K,
                depth=lgn.L,
                tau=lgn.tau,
                seed=base_seed * 1_000_003 + i,
                residual_init_strength=lgn.residual_init_strength,
            )
        return factory

    raise ValueError(f"unknown ffn type: {cfg.ffn!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/config.py tests/test_config.py
git commit -m "feat(config): TransformerCfg, LGNCfg, FFN factory dispatch"
```

---

## Task 15: Concrete config files

**Files:**
- Create: `configs/__init__.py` (empty)
- Create: `configs/poc_a_mlp.py`
- Create: `configs/poc_a_lgn.py`
- Create: `configs/poc_b_mlp.py`
- Create: `configs/poc_b_lgn.py`

- [ ] **Step 1: Create `configs/__init__.py`**

```python
# configs/__init__.py
```

- [ ] **Step 2: Write `configs/poc_a_mlp.py`**

```python
# configs/poc_a_mlp.py
"""POC-A baseline: standard ReLU² MLP, d_model=128, 4 layers."""
from nanolgn.config import TransformerCfg

cfg = TransformerCfg(
    d_model=128, n_layer=4, n_head=4, ctx_len=256,
    vocab_size=50257, ffn="mlp", seed=0,
)
lgn = None

# Training schedule.
batch_size = 32
max_steps = 5000
warmup_steps = 200
peak_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
grad_clip = 1.0
eval_every = 250
eval_tokens = 1_000_000
log_every = 50
log_block_stats_until = 200
data_train = "data/tinystories_train.bin"
data_val   = "data/tinystories_val.bin"
```

- [ ] **Step 3: Write `configs/poc_a_lgn.py`**

```python
# configs/poc_a_lgn.py
"""POC-A LGN: same config as poc_a_mlp, ffn='lgn'."""
from nanolgn.config import TransformerCfg, LGNCfg

cfg = TransformerCfg(
    d_model=128, n_layer=4, n_head=4, ctx_len=256,
    vocab_size=50257, ffn="lgn", seed=0,
)
lgn = LGNCfg(K=16, L=4, tau=16.0, residual_init_strength=5.0)

batch_size = 32
max_steps = 5000
warmup_steps = 200
peak_lr = 3e-4
min_lr = 3e-5
weight_decay = 0.1
grad_clip = 1.0
eval_every = 250
eval_tokens = 1_000_000
log_every = 50
log_block_stats_until = 200
data_train = "data/tinystories_train.bin"
data_val   = "data/tinystories_val.bin"
```

- [ ] **Step 4: Write `configs/poc_b_mlp.py` and `configs/poc_b_lgn.py`**

```python
# configs/poc_b_mlp.py
from nanolgn.config import TransformerCfg

cfg = TransformerCfg(
    d_model=256, n_layer=6, n_head=4, ctx_len=512,
    vocab_size=50257, ffn="mlp", seed=0,
)
lgn = None

batch_size = 16
max_steps = 20000
warmup_steps = 200
peak_lr = 1e-4
min_lr = 1e-5
weight_decay = 0.1
grad_clip = 1.0
eval_every = 250
eval_tokens = 1_000_000
log_every = 50
log_block_stats_until = 200
data_train = "data/fineweb_train.bin"
data_val   = "data/fineweb_val.bin"
```

```python
# configs/poc_b_lgn.py
from nanolgn.config import TransformerCfg, LGNCfg

cfg = TransformerCfg(
    d_model=256, n_layer=6, n_head=4, ctx_len=512,
    vocab_size=50257, ffn="lgn", seed=0,
)
lgn = LGNCfg(K=32, L=6, tau=32.0, residual_init_strength=5.0)

batch_size = 16
max_steps = 20000
warmup_steps = 200
peak_lr = 1e-4
min_lr = 1e-5
weight_decay = 0.1
grad_clip = 1.0
eval_every = 250
eval_tokens = 1_000_000
log_every = 50
log_block_stats_until = 200
data_train = "data/fineweb_train.bin"
data_val   = "data/fineweb_val.bin"
```

- [ ] **Step 5: Verify all configs import cleanly**

Run: `python -c "import importlib; [importlib.import_module(f'configs.{n}') for n in ['poc_a_mlp','poc_a_lgn','poc_b_mlp','poc_b_lgn']]; print('OK')"`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add configs/
git commit -m "feat(configs): POC-A and POC-B MLP+LGN training configs"
```

---

## Task 16: Data preparation script for TinyStories

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/prepare_tinystories.py`

- [ ] **Step 1: Create `scripts/__init__.py`** (empty)

- [ ] **Step 2: Write `scripts/prepare_tinystories.py`**

```python
# scripts/prepare_tinystories.py
"""Download TinyStories, tokenize with GPT-2 BPE, write uint16 binary shards.

Outputs:
    data/tinystories_train.bin
    data/tinystories_val.bin

Each file is a flat sequence of uint16 tokens (GPT-2 vocab fits in 16 bits).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data"
TRAIN_PATH = os.path.join(OUT_DIR, "tinystories_train.bin")
VAL_PATH = os.path.join(OUT_DIR, "tinystories_val.bin")


def encode_split(split, encoder, out_path: str) -> int:
    """Tokenize one split, append BOS-style separator between docs, write."""
    eot = encoder.eot_token
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    with open(out_path, "wb") as f:
        for ex in tqdm(split, desc=f"tokenize → {os.path.basename(out_path)}"):
            ids = encoder.encode_ordinary(ex["text"])
            ids.append(eot)
            arr = np.asarray(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total += arr.size
    return total


def main() -> int:
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories")
    n_train = encode_split(ds["train"], enc, TRAIN_PATH)
    n_val   = encode_split(ds["validation"], enc, VAL_PATH)
    print(f"train tokens: {n_train:,}")
    print(f"val tokens:   {n_val:,}")
    print(f"wrote {TRAIN_PATH} and {VAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke-test the script imports cleanly (no run yet)**

Run: `python -c "import scripts.prepare_tinystories as p; print('OK')"`
Expected: `OK` (no actual download in tests).

- [ ] **Step 4: Document how to run it in the README**

Append to `README.md`:

```markdown

## Data preparation

POC-A uses TinyStories. To download and tokenize:

```bash
python scripts/prepare_tinystories.py
```

This writes `data/tinystories_train.bin` and `data/tinystories_val.bin`,
each a flat uint16 sequence of GPT-2 BPE tokens. ~30 minutes on a fast disk.

POC-B uses FineWeb-edu — script analogous, not in scope for the POC.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/prepare_tinystories.py README.md
git commit -m "feat(scripts): TinyStories tokenization script (uint16 shards)"
```

---

## Task 17: Memmap data loader

**Files:**
- Create: `nanolgn/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import os
import numpy as np
import torch
import pytest
from nanolgn.data import MemmapTokenLoader

def _write_dummy_shard(tmp_path, n_tokens=10000):
    p = tmp_path / "tokens.bin"
    arr = np.arange(n_tokens, dtype=np.uint16) % 257
    p.write_bytes(arr.tobytes())
    return str(p)

def test_loader_yields_correct_shapes(tmp_path):
    path = _write_dummy_shard(tmp_path)
    loader = MemmapTokenLoader(path=path, batch_size=4, ctx_len=16, seed=0)
    x, y = next(iter(loader))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_loader_targets_are_inputs_shifted_by_one(tmp_path):
    path = _write_dummy_shard(tmp_path)
    loader = MemmapTokenLoader(path=path, batch_size=2, ctx_len=8, seed=0)
    x, y = next(iter(loader))
    # The loader reads ctx_len+1 raw tokens and splits into (x, y).
    for b in range(2):
        # Find this row in the underlying file by matching x[b,0] then verify shift.
        # Simpler: just check elementwise shift property for this batch.
        pass  # see next test
    # Build from arr again to verify directly
    arr = np.frombuffer(open(path, "rb").read(), dtype=np.uint16).astype(np.int64)
    arr_t = torch.from_numpy(arr)
    # All x rows must satisfy y == arr_t shifted by 1 from x's start.
    for b in range(x.size(0)):
        # find any starting offset s where arr[s:s+8] == x[b]
        for s in range(arr.size - 9):
            if torch.equal(arr_t[s:s+8], x[b]):
                assert torch.equal(arr_t[s+1:s+9], y[b])
                break
        else:
            pytest.fail(f"could not align row {b}")

def test_loader_determinism_same_seed_same_first_batch(tmp_path):
    path = _write_dummy_shard(tmp_path)
    a = MemmapTokenLoader(path=path, batch_size=4, ctx_len=16, seed=42)
    b = MemmapTokenLoader(path=path, batch_size=4, ctx_len=16, seed=42)
    xa, ya = next(iter(a))
    xb, yb = next(iter(b))
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `MemmapTokenLoader`**

```python
# nanolgn/data.py
"""Memmap-backed iterable token loader.

Reads a flat uint16 file, yields (x, y) batches where y = x shifted by one.
Sampling is i.i.d. uniform over valid offsets — no epoch boundaries.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


class MemmapTokenLoader:
    """Random-offset iterable over a uint16 token shard.

    Args:
        path: file written by `scripts/prepare_tinystories.py`.
        batch_size: B.
        ctx_len: T (sequence length per row).
        seed: RNG seed for offset sampling.
    """

    def __init__(self, path: str, batch_size: int, ctx_len: int, seed: int):
        self.path = path
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self._mm = np.memmap(path, dtype=np.uint16, mode="r")
        self._gen = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        n = self._mm.shape[0]
        max_start = n - self.ctx_len - 1
        starts = self._gen.integers(0, max_start, size=self.batch_size)
        xs = np.empty((self.batch_size, self.ctx_len), dtype=np.int64)
        ys = np.empty((self.batch_size, self.ctx_len), dtype=np.int64)
        for i, s in enumerate(starts):
            chunk = self._mm[s : s + self.ctx_len + 1].astype(np.int64)
            xs[i] = chunk[:-1]
            ys[i] = chunk[1:]
        return torch.from_numpy(xs), torch.from_numpy(ys)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nanolgn/data.py tests/test_data.py
git commit -m "feat(data): memmap token loader with seeded random offsets"
```

---

## Task 18: Training script (basic)

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Write `scripts/train.py`**

```python
# scripts/train.py
"""Train a nano-lgn config end-to-end.

Usage:
    python scripts/train.py poc_a_mlp
    python scripts/train.py poc_a_lgn

The argument is the module name under `configs/` (without .py).
"""
from __future__ import annotations
import argparse
import importlib
import math
import os
import sys
import time
import torch
from torch import nn

from nanolgn.config import TransformerCfg, LGNCfg, make_ffn_factory
from nanolgn.gpt import GPT
from nanolgn.data import MemmapTokenLoader


def cosine_lr(step: int, warmup: int, max_steps: int, peak: float, min_lr: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak - min_lr) * cos


def evaluate(model, loader, device, max_tokens: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                _, loss = model(x, y)
            n = x.numel()
            total_loss += loss.item() * n
            total_tokens += n
            if total_tokens >= max_tokens:
                break
    model.train()
    return total_loss / max(1, total_tokens)


def build_model(cfg_module) -> tuple[GPT, torch.device]:
    cfg: TransformerCfg = cfg_module.cfg
    lgn: LGNCfg | None = cfg_module.lgn
    factory = make_ffn_factory(cfg, lgn=lgn)
    torch.manual_seed(cfg.seed)
    model = GPT(cfg, ffn_factory=factory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config module under configs/, e.g. poc_a_lgn")
    args = parser.parse_args()

    cfg_mod = importlib.import_module(f"configs.{args.config}")
    cfg = cfg_mod.cfg

    model, device = build_model(cfg_mod)
    print(f"model params: {model.num_params():,}  device: {device}")

    train_loader = MemmapTokenLoader(
        path=cfg_mod.data_train,
        batch_size=cfg_mod.batch_size,
        ctx_len=cfg.ctx_len,
        seed=cfg.seed,
    )
    val_loader_factory = lambda: MemmapTokenLoader(
        path=cfg_mod.data_val,
        batch_size=cfg_mod.batch_size,
        ctx_len=cfg.ctx_len,
        seed=cfg.seed + 1,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_mod.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg_mod.weight_decay,
    )

    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(cfg_mod.max_steps):
        lr = cosine_lr(step, cfg_mod.warmup_steps, cfg_mod.max_steps,
                       cfg_mod.peak_lr, cfg_mod.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_mod.grad_clip)
        opt.step()

        if step % cfg_mod.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:>6d}  lr {lr:.2e}  train_loss {loss.item():.4f}  "
                  f"elapsed {elapsed:.1f}s")

        if step > 0 and step % cfg_mod.eval_every == 0:
            val_loss = evaluate(model, val_loader_factory(), device,
                                max_tokens=cfg_mod.eval_tokens)
            ppl = math.exp(min(val_loss, 20.0))
            print(f"  >>> step {step}  val_loss {val_loss:.4f}  val_ppl {ppl:.2f}")

    val_loss = evaluate(model, val_loader_factory(), device,
                        max_tokens=cfg_mod.eval_tokens)
    print(f"FINAL  val_loss {val_loss:.4f}  val_ppl {math.exp(min(val_loss, 20.0)):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-import the script**

Run: `python -c "import scripts.train as t; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat(scripts): training loop (AdamW, cosine LR, bf16, eval)"
```

---

## Task 19: Per-block diagnostic logging

**Files:**
- Modify: `nanolgn/lgn_mlp.py` (append helpers)
- Modify: `scripts/train.py` (wire up logging)

- [ ] **Step 1: Add diagnostic helpers to `lgn_mlp.py`**

Append to `nanolgn/lgn_mlp.py`:

```python
@torch.no_grad()
def lgn_block_stats(block: "LGNMLPBlock", last_ffn_out: Tensor) -> dict:
    """Collect early-warning diagnostics for one LGN-MLP block.

    Args:
        block: the LGNMLPBlock instance.
        last_ffn_out: the (B, T, d) tensor returned by the most recent forward.

    Returns dict with keys:
        ffn_out_norm_mean : float — mean L2 norm per token
        ffn_out_max       : float — max abs value
        gate_entropy_mean : float — mean entropy of softmax(W) across all neurons (nats)
        threshold_in_range_frac : float — fraction of (theta_{i,k}) that lie within
            ±3 of zero (i.e. plausibly active for typical RMS-normed inputs).
    """
    out = last_ffn_out
    norm = out.flatten(0, -2).norm(dim=-1).mean().item()
    omax = out.abs().max().item()

    entropies = []
    for layer in block.body.layers:
        p = torch.softmax(layer.W, dim=-1)
        h = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)   # (n,)
        entropies.append(h.mean().item())
    gate_entropy_mean = sum(entropies) / max(1, len(entropies))

    theta = block.encode.theta
    in_range = ((theta.abs() <= 3.0).float()).mean().item()

    return {
        "ffn_out_norm_mean": norm,
        "ffn_out_max": omax,
        "gate_entropy_mean": gate_entropy_mean,
        "threshold_in_range_frac": in_range,
    }
```

- [ ] **Step 2: Modify `Block.forward` to optionally cache its FFN output**

Edit `nanolgn/gpt.py`'s `Block` class:

Replace:

```python
class Block(nn.Module):
    """Pre-norm transformer block with a pluggable FFN factory."""

    def __init__(self, cfg, ffn_factory: Callable):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model, n_head=cfg.n_head, ctx_len=cfg.ctx_len
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = ffn_factory(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

With:

```python
class Block(nn.Module):
    """Pre-norm transformer block with a pluggable FFN factory.

    If `cache_ffn_out` is True, `last_ffn_out` holds the most recent FFN output
    (detached, on-device). Used for early-training diagnostic logging.
    """

    def __init__(self, cfg, ffn_factory: Callable):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model, n_head=cfg.n_head, ctx_len=cfg.ctx_len
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = ffn_factory(cfg)
        self.cache_ffn_out: bool = False
        self.last_ffn_out: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        ffn_out = self.ffn(self.norm2(x))
        if self.cache_ffn_out:
            self.last_ffn_out = ffn_out.detach()
        x = x + ffn_out
        return x
```

- [ ] **Step 3: Wire up logging in `scripts/train.py`**

In `scripts/train.py`, just before the training loop (after `opt = ...`), insert:

```python
    # Enable per-block FFN-out caching only while we're collecting stats.
    from nanolgn.lgn_mlp import LGNMLPBlock, lgn_block_stats
    is_lgn = cfg_mod.lgn is not None
    if is_lgn:
        for block in model.blocks:
            block.cache_ffn_out = True
```

In the same script, inside the training loop, just after the `opt.step()` line, insert:

```python
        if is_lgn and step < cfg_mod.log_block_stats_until and step % cfg_mod.log_every == 0:
            for i, block in enumerate(model.blocks):
                if isinstance(block.ffn, LGNMLPBlock) and block.last_ffn_out is not None:
                    s = lgn_block_stats(block.ffn, block.last_ffn_out)
                    print(f"    block {i}: norm={s['ffn_out_norm_mean']:.3f} "
                          f"max={s['ffn_out_max']:.3f} "
                          f"H(p)={s['gate_entropy_mean']:.3f} "
                          f"theta_in_range={s['threshold_in_range_frac']:.2f}")
        if is_lgn and step == cfg_mod.log_block_stats_until:
            for block in model.blocks:
                block.cache_ffn_out = False
                block.last_ffn_out = None
```

- [ ] **Step 4: Smoke-test the wiring with a tiny in-memory run**

Add `tests/test_logging.py`:

```python
# tests/test_logging.py
import torch
from nanolgn.config import TransformerCfg, LGNCfg, make_ffn_factory
from nanolgn.gpt import GPT
from nanolgn.lgn_mlp import LGNMLPBlock, lgn_block_stats

def test_lgn_block_stats_returns_expected_keys():
    cfg = TransformerCfg(
        d_model=32, n_layer=1, n_head=2, ctx_len=8,
        vocab_size=257, ffn="lgn", seed=0,
    )
    lgn = LGNCfg(K=4, L=2, tau=4.0, residual_init_strength=5.0)
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
```

- [ ] **Step 5: Run the new test**

Run: `pytest tests/test_logging.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add nanolgn/lgn_mlp.py nanolgn/gpt.py scripts/train.py tests/test_logging.py
git commit -m "feat(logging): per-block LGN diagnostics (||ffn_out||, H(p), thresholds)"
```

---

## Task 20: Smoke training test

**Files:**
- Create: `tests/test_smoke_train.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_smoke_train.py
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
    lgn = LGNCfg(K=4, L=2, tau=4.0, residual_init_strength=5.0)
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
```

- [ ] **Step 2: Register the `slow` marker and run the test**

Edit `pyproject.toml`'s `[tool.pytest.ini_options]` block to add:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
markers = [
    "slow: end-to-end smoke training; opt in with -m slow",
]
```

Run: `pytest -m slow tests/test_smoke_train.py -v`
Expected: 1 passed (takes ~10–60s on CPU).

- [ ] **Step 3: Verify it is skipped by default**

Run: `pytest -q`
Expected: previous test count + `1 deselected`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_train.py pyproject.toml
git commit -m "test: smoke training test (LGN learns synthetic next-token task)"
```

---

## Task 21: README finalization (running the POC)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append usage instructions to `README.md`**

```markdown

## Running the POC

### POC-A (TinyStories, single GPU, ~30 min/run)

Prepare data once:

```bash
python scripts/prepare_tinystories.py
```

Train the MLP baseline and the LGN variant back-to-back:

```bash
python scripts/train.py poc_a_mlp
python scripts/train.py poc_a_lgn
```

Both runs use the same data, same seed, same step count. Compare the printed
`FINAL val_loss` lines. Success criteria:

- **Must:** LGN variant trains without diverging.
- **Strong:** LGN val loss within 30% of MLP val loss.
- **Stretch:** within 15%.

### POC-B (FineWeb-edu, single GPU, ~3–6 h/run)

Out of scope for the initial POC pass. Add a `prepare_fineweb.py` modeled on
the TinyStories script before running `python scripts/train.py poc_b_lgn`.

## Running the tests

Fast suite (skips smoke training):

```bash
pytest -q
```

Including the smoke training test:

```bash
pytest -q -m slow
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: finalize README with POC run instructions"
```

---

## Self-review

Spec coverage check (against `docs/superpowers/specs/2026-05-12-lgn-mlp-block-poc-design.md`):

- §"Approach: two-stage POC" — covered by configs in Task 15 and run instructions in Task 21.
- §"Architecture / Repository layout" — Tasks 1–18 produce every file listed.
- §"Stage 1 ThermometerEncode" — Task 5.
- §"Stage 2 LGN body" with residual init and per-layer seeds — Tasks 3–4.
- §"Stage 3 GroupSumDecode" — Task 6.
- §"Whole block" `LGNMLPBlock` — Task 7.
- §"Transformer scaffold" with pluggable FFN factory — Tasks 8–14.
- §"Configs" — Task 15.
- §"Tokenizer and data" / §"Data preparation" — Task 16.
- §"Loader" — Task 17.
- §"Optimizer and schedule" — Task 18.
- §"Logging" diagnostics (`||ffn_out||`, gate entropy, threshold range) — Task 19.
- §"Reproducibility" (seeded connections, deterministic factory) — Task 14 test, Task 17 test.
- §"Testing" three layers — gate truth tables (Task 2), module invariants incl. residual-init-≈-identity (Tasks 3, 4, 5, 6, 7), smoke training (Task 20).
- §"Risks #1 K too low" — config knob `K` exposed (Task 15), `lgn_block_stats` reports `gate_entropy_mean` for early warning (Task 19).
- §"Risks #5 wrong optimizer" — single AdamW group is the documented default (Task 18); future param-group split is a one-line change in `scripts/train.py`.

No spec section is uncovered.

Placeholder scan: no `TBD`, no "implement later", no "similar to Task N" without code. Every test step has a concrete test; every implementation step has the actual code; every shell step has the actual command and expected behavior.

Type/name consistency check:

- `LogicLayer(n=, seed=, residual_init_strength=)` — same signature in Tasks 3, 4, 14.
- `LGNBody(n=, depth=, seed=, residual_init_strength=)` — same in Tasks 4, 7.
- `ThermometerEncode(d_model=, k=)` — same in Tasks 5, 7.
- `GroupSumDecode(d_model=, k=, tau=)` — same in Tasks 6, 7.
- `LGNMLPBlock(d_model=, k=, depth=, tau=, seed=, residual_init_strength=)` — same in Tasks 7, 12, 14, 19, 20.
- `LGNCfg(K=, L=, tau=, residual_init_strength=)` — capitalised `K`, `L`. Used consistently in Tasks 14, 15, 19, 20.
- `make_ffn_factory(cfg, lgn=)` — same call sites everywhere.
- `Block(cfg, ffn_factory=)` — same in Tasks 12, 13.
- `GPT(cfg, ffn_factory=)` — same in Tasks 13, 14, 19, 20.
- `MemmapTokenLoader(path=, batch_size=, ctx_len=, seed=)` — same in Tasks 17, 18.
- `lgn_block_stats(block, last_ffn_out)` — single signature, used in Tasks 19 testing and the train-loop wiring.
- `Block.cache_ffn_out` / `Block.last_ffn_out` introduced in Task 19 and used in the same task's logging block; not referenced before introduction.

All checks pass.
