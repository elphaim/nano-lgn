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
    residual_init_strength: float = 7.5
    interconnect: str = "fixed"
    topk: int = 8
    c_sparsity: float = 1.0


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
                interconnect=lgn.interconnect,
                topk=lgn.topk,
                c_sparsity=lgn.c_sparsity,
            )
        return factory

    raise ValueError(f"unknown ffn type: {cfg.ffn!r}")
