# Lesson 5 — Putting it all together

You now know the four pillars:

1. **Training** = nudging numbers (Lesson 1).
2. **Tokens + embeddings** = turning words into vectors (Lesson 2).
3. **Attention** = letting words look at each other (Lesson 3).
4. **MLM** = fill-in-the-blank as a self-supervised training game (Lesson 4).

Open `../pragma_mini.py` next to this file. We're going to read it again, but
this time you should understand every line.

## Re-reading `pragma_mini.py`

### Section 1: Vocabulary
```python
KEYS   = ["pet", "action", "place"]
VALUES = ["dog", "cat", "fish", "eat", "sleep", "play", "garden", "couch", "bowl"]
vocab  = [PAD, MASK] + KEYS + VALUES
```
Same idea as Lesson 2's vocab — just with **two kinds of tokens**: KEYS (field
names like "pet") and VALUES (actual contents like "dog"). PRAGMA does this
because banking data is `key: value` pairs (`Type: card_payment`), not free text.

### Section 2: Synthetic data + RULES
The rules dict is the hidden pattern. The model never sees `RULES` directly —
it has to discover it from examples.

### Section 3: `encode()`
Turn an event into a flat list of token IDs:
```
[("pet","dog"), ("action","eat")] → [pet, dog, action, eat] → [4, 7, 5, 10]
```
This is Lesson 2's "tokenise" step, plus the key/value structure from PRAGMA.

### Section 4: `mask()`
Identical to the training game in Lesson 4. For each value token, with 25%
probability:
- save the original ID into `labels`
- replace the input with `<mask>`

`labels[i] = -100` for everything we DIDN'T hide — so the loss ignores those.

### Section 5: `PragmaMini`
The model. Compare to Lesson 4's `TinyMLM`:
```python
self.emb   = nn.Embedding(V, d)
self.pos   = nn.Embedding(64, d)     # NEW: positional embedding
self.enc   = nn.TransformerEncoder(layer, layers)   # multiple blocks now
self.head  = nn.Linear(d, V)
```
Two new things vs. Lesson 4:
- **`pos`**: a positional embedding. Attention doesn't care about word order
  on its own (it's just a weighted sum), so we add a position vector to every
  token's embedding to tell the model "this one came first, this one second…"
- **`TransformerEncoder` (plural layers)**: stacks multiple encoder blocks.
  The output of one becomes the input to the next, so the model can build
  up more abstract representations as it goes deeper.

### Section 6: The training loop
Identical to Lesson 1's loop. Just with a Transformer instead of `w*x + b`.

### Section 7: `guess()`
Inference. Hide a value, run the model, read off the top prediction.

**You've now read every line.** Same recipe as BERT. Same recipe as PRAGMA.

---

## What real PRAGMA adds on top

Our toy is a faithful reduction, but the real paper scales the recipe in
several ways. Here's a brief tour, with references to the paper.

### 1. Time encoding (§2.2 "Temporal Information")

Banking events have timestamps. Real PRAGMA encodes time **twice**:

- **Log-seconds since the previous event** — `8·ln(1 + t/8)`. This squashes
  huge gaps (years) into a manageable range without losing precision for
  recent events (seconds).
- **Calendar features** — hour of day, day of week, day of month, each
  embedded with sine/cosine of the angle around the cycle. This is how the
  model learns "salaries arrive on the 1st of the month" or "bars cluster
  on Friday nights".

In our toy: skipped entirely. Add it as a stretch exercise!

### 2. Value type-specific tokenisation (§2.2 "Value")

Banking values are heterogeneous:
- **Numerical** (Amount = 14.99) → bucket by percentile. Each bucket = 1 token.
  Preserves magnitude and order.
- **Categorical** (Currency = "gbp") → 1 token per value.
- **Textual** (Description = "metal plan") → BPE subwords ("met", "al", "plan").

In our toy: every value is categorical (1 token each).

### 3. Two-branch architecture (§2.3)

Real PRAGMA has **three** Transformers:
- **Profile State Encoder** — your static info (region, plan, balance bucket).
- **Event Encoder** — each event encoded independently.
- **History Encoder** — runs over the sequence of events to mix them up.

Our toy has one Transformer for everything.

### 4. Scale (§2.3, Table 1)

| Model      | Parameters | d_model | Layers (P/E/H) | Heads |
|------------|------------|---------|----------------|-------|
| PRAGMA-S   | 10 M       | 192     | 1 / 5 / 2      | 3     |
| PRAGMA-M   | 100 M      | 512     | 3 / 16 / 6     | 8     |
| PRAGMA-L   | 1 B        | 1024    | 9 / 45 / 18    | 16    |
| **Ours**   | ~5 K       | 32      | – / 2 / –      | 2     |

PRAGMA-L is **about 200,000× larger** than our toy. Trained for 2 weeks on 32
NVIDIA H100 GPUs.

### 5. Smarter masking (§2.3.5)

Three kinds of masking mixed together:
- **Token-level** (15%) — like our toy.
- **Event-level** (10%) — hide a whole event, force the model to use other
  events for context.
- **Key-level** (10%) — pick a key (e.g., "Amount") and hide ALL its values
  in the sequence.

### 6. Downstream adaptation (§3.1)

The real payoff. After pre-training:
- **Embedding probe**: freeze the model. Stick a tiny linear classifier on
  top. Train it on labelled data (e.g., did this user default?). This is
  what your capstone (Lesson 6) does.
- **LoRA fine-tuning**: tune ~2–4% of weights via Low-Rank Adaptation. Better
  results, slightly more work.

### 7. Engineering (§2.4)

- **Sequence packing**: pack short event histories together to avoid wasted
  computation on padding.
- **Dynamic batching**: batches have a token budget, not a fixed user count.
- **Truncation**: events capped at 24 tokens, profile state at 200, users at
  6,500 events.

These don't change the model — they just make training tractable on real data.

---

## You're ready

You've understood the full picture. Next: build your own.

Move on to `06_capstone.md`.
