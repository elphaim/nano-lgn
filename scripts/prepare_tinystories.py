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
