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
