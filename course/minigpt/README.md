# minigpt — train a GPT from scratch on one Mac

The course capstone: take the nursery-rhyme `CharGPT` from lesson 4g and turn it
into a **real GPT** — the same architecture as GPT-3, just the small size you can
actually train on an Apple Silicon Mac.

> **GPT-3 is a *family*, not one model.** The GPT-3 paper (Table 2.1) defines 8
> sizes from 125M to 175B params, all with this exact recipe. The 175B one needs
> thousands of GPUs. What you train here is the baby of the same family — a true
> GPT-3 architecture, sized for your laptop.

## What's the same as GPT-3 / what changed from lesson 4g

| Piece | lesson 4g CharGPT | here (GPT-3 recipe) |
|---|---|---|
| token | single character | **BPE subword** (tiktoken gpt2, 50,257 vocab) |
| block | `TransformerEncoder` | decoder block, **pre-LayerNorm**, **GELU** |
| head | separate `Linear` | **weight-tied** to the embedding table |
| extras | — | dropout, **AdamW + cosine LR + warmup**, grad clip, bf16 |
| loop | predict → loss → step | **identical** (that's the whole point) |

## Setup

Uses the project venv at `../../.venv` (already has torch + tiktoken + tqdm).

```bash
cd "course/minigpt"
PY="../../.venv/bin/python"
```

## 1. Quick sanity check — Shakespeare (minutes)

```bash
$PY data.py shakespeare
$PY train.py --preset shakespeare --max_iters 3000
$PY sample.py --ckpt ckpt_shakespeare.pt --prompt "ROMEO:" --tokens 300
```

Tiny dataset, so it overfits fast — but it proves the whole machine works and you
watch loss fall and Shakespeare-ish text appear.

## 2. The real run — TinyStories (a few hours, watch it write English)

TinyStories is engineered so even a ~30M model writes **coherent stories**.

```bash
$PY data.py tinystories --mb 200        # download + tokenize ~200 MB of stories
$PY train.py --preset tinystories --max_iters 12000 --batch_size 16 --grad_accum 2
$PY sample.py --ckpt ckpt_tinystories.pt --prompt "Once upon a time" --tokens 300
```

Checkpoints save automatically whenever val loss improves, and the trainer prints
a sample every 500 iters so you can literally watch it learn.

## Realistic expectations on a MacBook Air M4 (16 GB, fanless)

- Throughput measured on this machine: **~50 iters/min** at the default ~30M /
  block-256 config (effective batch 16). Bigger effective batch = proportionally slower.
- **The Air has no fan** — long runs thermally throttle. Train overnight, keep it
  cool/plugged in, and don't expect a desktop GPU's pace.
- Memory is *not* the bottleneck (16 GB handles 30M easily in bf16). Heat + compute are.
- Want it faster? Two real upgrades: (a) rewrite in **Apple MLX** (native, ~2× on
  Mac), or (b) rent a cloud GPU for a few hours to train the true **gpt3-small**
  (125M) preset — that config is in `model.py` but is impractical on the Air.

## Files

- `model.py` — the GPT (architecture + presets: `shakespeare`, `tinystories`, `gpt3-small`)
- `data.py` — download + BPE-tokenize into `train.bin` / `val.bin`
- `train.py` — the training loop (MPS, bf16, cosine LR, checkpoints, live samples)
- `sample.py` — generate from a checkpoint

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
