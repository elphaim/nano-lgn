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
