# Capstone — sample solution (for the parent)

A reference implementation of the Lesson 6 capstone. Keep this folder
hidden from the student until they finish (or get stuck).

The solution is split into the three files the capstone asks for, plus
a fourth that runs the full pipeline end-to-end:

| File | What it does |
|---|---|
| `capstone_data.py` | Generates 2000 synthetic users (95% normal, 5% fraud), saves to `data.pt`. |
| `capstone_pretrain.py` | Pre-trains a tiny PRAGMA-style encoder with MLM on all events. Saves encoder to `encoder.pt`. |
| `capstone_classify.py` | Loads the frozen encoder, trains a linear head, evaluates. Also trains a random-init baseline for comparison, across several labelled-data sizes. |
| `run_all.sh` | One command to run all three in order. |

## Run

```bash
cd "/Users/fernando/Pragma LLM model/course/_solutions"
bash run_all.sh
```

## Expected results

The classifier on the pre-trained encoder reaches ~95% fraud recall with
the full labelled set. The random-init baseline keeps up when labels are
plentiful, but **falls behind sharply when labelled data is scarce** — which
is the whole pitch of foundation models.

The exact numbers will vary between runs (small dataset, lots of noise),
but the trend is robust.

## Design notes

- **Fraud signal**: bursts of `(purchase, tiny, night)` events with rare
  `login`s — a *contextual* pattern, not just a frequency one. This is
  important: if fraud were detectable from raw token counts alone, attention
  and pre-training wouldn't help.
- **User-level pooling**: mean over all output positions. PRAGMA uses a
  prepended `[USR]` token; mean-pooling works fine at this scale.
- **Tiny model**: 16-d, 2 layers, 2 heads. ~5k parameters. Trains in seconds.
