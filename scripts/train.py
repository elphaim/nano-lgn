"""Train a nano-lgn config end-to-end.

Usage:
    python scripts/train.py poc_a_mlp
    python scripts/train.py poc_a_lgn

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 -m scripts.train poc_a_lgn_topk

`cfg_mod.batch_size` is the per-rank batch. Effective global batch = batch_size * world_size.
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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


def _has_native_bf16(device: torch.device) -> bool:
    """True only when the GPU has native bf16 Tensor Cores (Ampere sm_80+).

    `torch.cuda.is_bf16_supported()` is too permissive — it returns True on
    Turing (T4, sm_75) where bf16 runs emulated on the fp32 path with no
    Tensor Core acceleration. Gate on compute capability instead.
    """
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def setup_ddp() -> tuple[int, int, int, bool]:
    """Returns (rank, world_size, local_rank, is_ddp). No-op if env vars absent."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def evaluate(model, loader, device, max_tokens: int, amp_dtype: torch.dtype) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            n = x.numel()
            total_loss += loss.item() * n
            total_tokens += n
            if total_tokens >= max_tokens:
                break
    model.train()
    return total_loss / max(1, total_tokens)


def build_model(cfg_module, device: torch.device) -> GPT:
    cfg: TransformerCfg = cfg_module.cfg
    lgn: LGNCfg | None = cfg_module.lgn
    factory = make_ffn_factory(cfg, lgn=lgn)
    torch.manual_seed(cfg.seed)
    model = GPT(cfg, ffn_factory=factory)
    model.to(device)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config module under configs/, e.g. poc_a_lgn")
    args = parser.parse_args()

    rank, world_size, local_rank, is_ddp = setup_ddp()
    is_main = rank == 0

    cfg_mod = importlib.import_module(f"configs.{args.config}")
    cfg = cfg_mod.cfg

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_ddp else "cuda")
    else:
        device = torch.device("cpu")

    amp_dtype = torch.bfloat16 if _has_native_bf16(device) else torch.float16

    model = build_model(cfg_mod, device)
    if is_main:
        print(f"model params: {model.num_params():,}  device: {device}  "
              f"amp_dtype: {str(amp_dtype).split('.')[-1]}  world_size: {world_size}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_ddp else model

    train_loader = MemmapTokenLoader(
        path=cfg_mod.data_train,
        batch_size=cfg_mod.batch_size,
        ctx_len=cfg.ctx_len,
        seed=cfg.seed + rank,
    )
    val_loader_factory = lambda: MemmapTokenLoader(
        path=cfg_mod.data_val,
        batch_size=cfg_mod.batch_size,
        ctx_len=cfg.ctx_len,
        seed=cfg.seed + 1 + rank,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_mod.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg_mod.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
    )

    from nanolgn.lgn_mlp import LGNMLPBlock, lgn_block_stats
    is_lgn = cfg_mod.lgn is not None
    if is_lgn:
        for block in raw_model.blocks:
            block.cache_ffn_out = True

    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(cfg_mod.max_steps):
        lr = cosine_lr(step, cfg_mod.warmup_steps, cfg_mod.max_steps,
                       cfg_mod.peak_lr, cfg_mod.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_mod.grad_clip)
        scaler.step(opt)
        scaler.update()

        if is_main and is_lgn and step < cfg_mod.log_block_stats_until and step % cfg_mod.log_every == 0:
            for i, block in enumerate(raw_model.blocks):
                if isinstance(block.ffn, LGNMLPBlock) and block.last_ffn_out is not None:
                    s = lgn_block_stats(block.ffn, block.last_ffn_out)
                    line = (f"    block {i}: norm={s['ffn_out_norm_mean']:.3f} "
                            f"max={s['ffn_out_max']:.3f} "
                            f"H(p)={s['gate_entropy_mean']:.3f} "
                            f"theta_in_range={s['threshold_in_range_frac']:.2f}")
                    if s["interconnect_entropy_mean"] is not None:
                        line += (f" H(C)={s['interconnect_entropy_mean']:.3f}"
                                 f" uniq_argmax={s['interconnect_unique_argmax_frac']:.2f}")
                    print(line)
        if is_lgn and step == cfg_mod.log_block_stats_until:
            for block in raw_model.blocks:
                block.cache_ffn_out = False
                block.last_ffn_out = None

        if is_main and step % cfg_mod.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:>6d}  lr {lr:.2e}  train_loss {loss.item():.4f}  "
                  f"elapsed {elapsed:.1f}s")

        if is_main and step > 0 and step % cfg_mod.eval_every == 0:
            val_loss = evaluate(model, val_loader_factory(), device,
                                max_tokens=cfg_mod.eval_tokens, amp_dtype=amp_dtype)
            ppl = math.exp(min(val_loss, 20.0))
            print(f"  >>> step {step}  val_loss {val_loss:.4f}  val_ppl {ppl:.2f}")

    if is_main:
        val_loss = evaluate(model, val_loader_factory(), device,
                            max_tokens=cfg_mod.eval_tokens, amp_dtype=amp_dtype)
        print(f"FINAL  val_loss {val_loss:.4f}  val_ppl {math.exp(min(val_loss, 20.0)):.2f}")

    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
