"""
Train a GPT from scratch on your Mac (Apple Silicon / MPS).

The training loop is the SAME five lines from lesson 1 — predict, measure loss,
backward, step, repeat — wrapped in the grown-up GPT-3 recipe: AdamW with weight
decay, a cosine learning-rate schedule with warmup, gradient clipping, gradient
accumulation, and bf16 mixed precision.

Examples:
  # quick sanity check (minutes)
  python train.py --preset shakespeare --max_iters 2000

  # the real run — watch it learn to write stories (hours on an M4 Air)
  python data.py tinystories --mb 200
  python train.py --preset tinystories --max_iters 20000

Resume / sample from a checkpoint with sample.py.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch

from model import PRESETS, GPT

HERE = Path(__file__).parent
ENC = tiktoken.get_encoding("gpt2")


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, data, block_size, batch_size, device, ctx, iters=50):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def lr_at(it, lr, warmup, max_iters, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="shakespeare", choices=list(PRESETS))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)   # effective batch = batch_size * grad_accum
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None, help="checkpoint path (default ckpt_<preset>.pt)")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    out = Path(args.out) if args.out else HERE / f"ckpt_{args.preset}.pt"

    # bf16 autocast (MPS supports it); float32 is the safe fallback
    use_amp = args.dtype == "bfloat16" and device != "cpu"
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if use_amp \
        else torch.autocast(device_type="cpu", enabled=False)

    data_dir = HERE / "data" / args.preset
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    cfg = PRESETS[args.preset]
    model = GPT(cfg).to(device)
    print(f"device={device}  dtype={'bf16' if use_amp else 'fp32'}  "
          f"params={model.num_params()/1e6:.1f}M  "
          f"tokens(train)={len(train_data)/1e6:.1f}M")
    print(f"effective batch = {args.batch_size} x {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} sequences of {cfg.block_size} tokens\n")

    opt = model.configure_optimizers(args.weight_decay, args.lr, betas=(0.9, 0.95))

    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args.lr, args.warmup, args.max_iters, args.min_lr)

        if it % args.eval_every == 0:
            vl = eval_loss(model, val_data, cfg.block_size, args.batch_size, device, ctx)
            dt = time.time() - t0
            print(f"iter {it:>6} | val loss {vl:.3f} | lr {opt.param_groups[0]['lr']:.1e} | {dt/60:.1f} min")
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "iter": it, "val_loss": vl}, out)

        if it % args.sample_every == 0 and it > 0:
            model.eval()
            start = torch.tensor([ENC.encode_ordinary("\n")], device=device)
            with torch.no_grad(), ctx:
                gen = model.generate(start, max_new_tokens=160, temperature=0.8)
            print("  --- sample ---\n  " +
                  ENC.decode(gen[0].tolist()).replace("\n", "\n  ") + "\n  --------------")
            model.train()

        if it == args.max_iters:
            break

        # one optimization step, with gradient accumulation
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    print(f"\ndone. best val loss {best_val:.3f}. checkpoint -> {out}")


if __name__ == "__main__":
    main()
