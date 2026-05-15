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
