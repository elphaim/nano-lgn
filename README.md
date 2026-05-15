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

## Data preparation

POC-A uses TinyStories. To download and tokenize:

```bash
python scripts/prepare_tinystories.py
```

This writes `data/tinystories_train.bin` and `data/tinystories_val.bin`,
each a flat uint16 sequence of GPT-2 BPE tokens. ~30 minutes on a fast disk.

POC-B uses FineWeb-edu — script analogous, not in scope for the POC.

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

Full suite (includes the smoke training test, ~8s on CPU):

```bash
pytest -q
```

Smoke training test only:

```bash
pytest -q -m slow
```

Note: the `slow` marker is registered but is **not** auto-deselected from
the default run. To skip slow tests by default, add
`addopts = "-m 'not slow'"` to `[tool.pytest.ini_options]` in `pyproject.toml`.
