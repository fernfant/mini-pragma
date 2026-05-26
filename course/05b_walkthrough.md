# Lesson 5b — Walkthrough: streaming churn predictor

Companion to [`05b_streaming_churn.py`](05b_streaming_churn.py). Read them
side by side.

This file applies the **exact same recipe** as `pragma_mini.py` to a more
realistic problem: predicting which streaming-service users are about to
cancel. Every line is tagged with the lesson concept it uses — see the
legend below — so each time you see an idea reused, the intuition gets
deeper.

## 🧰 Lesson reference legend

- **L1** — the 5-line training loop
- **L1b** — architecture vs. training
- **L1c** — gradient descent details
- **L2** — tokens & embeddings
- **L3** — attention
- **L3b** — why Transformers won
- **L4** — masked language modelling
- **L5** — `pragma_mini.py`

If anything is fuzzy, click back to the lesson tagged.

---

## Section 1 — The data (L2)

```python
KEYS     = ["action", "genre", "time", "duration"]
ACTIONS  = ["start", "finish", "skip", "pause", "browse"]
GENRES   = ["comedy", "drama", "action", "documentary", "kids"]
TIMES    = ["morning", "afternoon", "evening", "night"]
DURATION = ["short", "medium", "long"]

PAD, MASK = "<pad>", "<mask>"
vocab  = [PAD, MASK] + KEYS + ACTIONS + GENRES + TIMES + DURATION
tok2id = {t: i for i, t in enumerate(vocab)}
V      = len(vocab)
```

> **L2 in action.** Same key-value vocabulary structure as `pragma_mini.py`
> (L5). Four keys instead of three; richer value sets. Total vocab: 23 tokens.

```python
def engaged_event():
    a = random.choices(ACTIONS, weights=[2, 8, 1, 1, 1])[0]   # mostly finishes
    g = random.choice(GENRES)
    t = random.choices(TIMES,   weights=[1, 2, 5, 2])[0]
    d = random.choices(DURATION, weights=[1, 3, 6])[0]
    return [("action", a), ("genre", g), ("time", t), ("duration", d)]

def churning_event(narrow_genre):
    a = random.choices(ACTIONS, weights=[3, 1, 6, 1, 4])[0]   # skips a lot
    g = random.choices([narrow_genre] * 6 + GENRES, k=1)[0]
    t = random.choices(TIMES,   weights=[2, 3, 2, 5])[0]
    d = random.choices(DURATION, weights=[7, 2, 1])[0]
    return [("action", a), ("genre", g), ("time", t), ("duration", d)]
```

> **Plain Python — no ML yet.** Defines two distinct event distributions.
> The model never sees these rules directly — it has to back them out from
> training examples. Same trick as `RULES` in `pragma_mini.py`.

Key signal for the model to discover:
- engaged users **finish** episodes and watch **long**.
- churning users **skip** a lot, watch **short**, narrow taste, mostly at **night**.

---

## Section 2 — Encoding events (L2)

```python
def encode_event(event):
    ids = []
    for k, v in event:
        ids.append(tok2id[k])
        ids.append(tok2id[v])
    return ids

def encode_user(events):
    return [tok for e in events for tok in encode_event(e)]
```

> **L2 in action.** Each event → 8 tokens (4 keys + 4 values).
> Each user has 15 events → **120 tokens per user**.
>
> Notice how every user becomes a flat token sequence — identical in
> shape to a sentence in a regular language model.

---

## Section 3 — Architecture (L1b + L2 + L3 + L3b)

> 🧰 **Concept bridge — every piece traced**
>
> | Component | Origin |
> |---|---|
> | `nn.Embedding(V, d)` | **[L2](02_walkthrough.md)** — token → vector |
> | `nn.Embedding(max_len, d)` (positional) | **[L5](05_putting_it_together.md)** — position embeddings |
> | `nn.TransformerEncoderLayer` (attention) | **[L3](03_walkthrough.md)** |
> | `nn.TransformerEncoderLayer` (feed-forward) | **[L1.5 Step 7.5](notebooks/lesson_01_5_from_linear_to_neural.ipynb)** |
> | `nn.TransformerEncoderLayer` (LayerNorm + residual) | **[L4e](04e_encoder_layer_from_scratch.py)** |
> | `nn.TransformerEncoder(layer, layers=2)` (stacking) | **[L1.5 Step 7.6](notebooks/lesson_01_5_from_linear_to_neural.ipynb)** — why depth matters |
> | `MLMHead` (vocab projection) | **[L4](04_walkthrough.md)** — the MLM head |
> | `ChurnHead` (mean-pool + Linear) | NEW — but each piece is `nn.Linear` + `tensor.mean()` |
>
> Same `Encoder` as `pragma_mini.py`. Only the heads change per task.

```python
class Encoder(nn.Module):
    def __init__(self, V, d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)              # (a)
        self.pos = nn.Embedding(max_len, d)        # (b)
        layer    = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)  # (c)
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.enc(self.emb(x) + self.pos(positions))
```

Identical to `pragma_mini.py`'s `PragmaMini.__init__`. The pieces:

| Line | Lesson | What it does |
|------|--------|--------------|
| `(a)` `nn.Embedding(V, d)` | **L2** | Token embedding table. 23 × 32 = 736 knobs. |
| `(b)` `nn.Embedding(max_len, d)` | (positional) | Lets the model know where in the sequence each token is. |
| `(c)` `TransformerEncoder(...)` | **L3** | Stack of 2 attention+FFN layers. Each layer is one round of "every word looks at every other word" (L3). |

> **L1b in action.** This is the architecture. **No training has happened
> yet** — these are just freshly random knobs.

```python
class MLMHead(nn.Module):
    def __init__(self, V, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d, V)
    def forward(self, h):
        return self.proj(h)
```

> The MLM output head — projects each context-aware vector back to vocab
> scores. Same shape as `pragma_mini.py`'s `self.head`.

```python
class ChurnHead(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d, 2)
    def forward(self, h):
        pooled = h.mean(dim=1)   # average over all positions
        return self.proj(pooled)
```

> **NEW — the downstream task head.** Unlike the MLM head (which predicts
> a word at each position), this head:
> 1. Averages the encoder's output across the user's 120-token sequence.
> 2. Projects the resulting single vector to 2 numbers — one for "engaged",
>    one for "churning".
>
> This is what the PRAGMA paper calls an **embedding probe** (§3.1).

---

## Section 4 — Pre-training (L4 in action)

```python
def mlm_mask(X_batch, p=0.20):
    X = X_batch.clone()
    y = torch.full_like(X, -100)
    is_value = ~torch.isin(X, KEY_IDS)
    pick = (torch.rand_like(X, dtype=torch.float) < p) & is_value
    y[pick] = X[pick]
    X[pick] = tok2id[MASK]
    return X, y
```

> **L4 in action.** For each value token (not key tokens), with 20%
> probability: remember the truth, replace with `<mask>`. Same masking
> idea as `pragma_mini.py` — just vectorised across the batch instead
> of looping.

```python
encoder  = Encoder(V)
mlm_head = MLMHead(V)
opt      = torch.optim.AdamW(list(encoder.parameters()) + list(mlm_head.parameters()), lr=3e-3)
loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)

for step in range(2000):
    idx       = torch.randint(0, N_USERS, (64,))
    xb, yb    = mlm_mask(X[idx])
    h         = encoder(xb)
    logits    = mlm_head(h)
    loss      = loss_fn(logits.reshape(-1, V), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
```

> **L1 + L1c + L4.** This is the **5-line training loop** you've now seen
> three times:
>
> 1. **Predict** — `logits = mlm_head(encoder(xb))`
> 2. **Measure wrongness** — `loss = loss_fn(...)`
> 3. **Clear notes** — `opt.zero_grad()`
> 4. **Compute gradients for every knob** — `loss.backward()` (L1c)
> 5. **Nudge every knob** — `opt.step()`
>
> The model has ~22,000 knobs. `loss.backward()` is computing 22,000
> gradients in this single line. **Same recipe as Lesson 1's `w` and `b`.**

```python
import copy
pretrained_encoder = copy.deepcopy(encoder)
```

> Save a snapshot of the pre-trained encoder. We'll need a clean copy
> later when we build the downstream classifier.

---

## Section 5 — Downstream task: predict churn

```python
perm   = torch.randperm(N_USERS)
split  = int(N_USERS * 0.8)
tr_idx, te_idx = perm[:split], perm[split:]
X_te, y_te     = X[te_idx], y[te_idx]
```

> Train/test split. 80% for training the classifier, 20% to evaluate.

```python
def freeze(mod):
    for p in mod.parameters(): p.requires_grad = False
    mod.eval()
```

> **L1b in action.** Setting `requires_grad = False` on a parameter
> tells PyTorch *"don't update this knob during training"*. We're going
> to freeze the encoder so that gradient descent only nudges the classifier
> head — preserving everything the encoder learned during pre-training.

```python
def train_classifier(encoder, X_tr, y_tr, epochs=200, freeze_encoder=True):
    head = ChurnHead()
    if freeze_encoder:
        freeze(encoder)
        params = list(head.parameters())
    else:
        params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        h      = encoder(X_tr)
        logits = head(h)
        loss   = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    ...
```

> **L1 + L1c again.** Identical 5-line loop. Now with the **churn labels**
> as the truth. Two modes:
>
> - `freeze_encoder=True` — only the head gets nudged (~66 knobs). The
>   encoder is a fixed feature extractor.
> - `freeze_encoder=False` — everything gets nudged. This is the
>   "baseline" — no pre-training; learn end-to-end from scratch.

---

## Section 6 — The comparison

```python
for n_labels in [20, 50, 200, 1000]:
    sub = tr_idx[:n_labels]
    ...
    # A. Frozen pre-trained encoder + classifier head only
    enc_a = copy.deepcopy(pretrained_encoder)
    acc_a, rec_a = train_classifier(enc_a, X_tr, y_tr, freeze_encoder=True)
    # B. Random-init encoder, train everything end-to-end
    enc_b = Encoder(V)
    acc_b, rec_b = train_classifier(enc_b, X_tr, y_tr, freeze_encoder=False)
```

> Try 4 settings: train the classifier with 20, 50, 200, or 1000 labelled
> users. For each, compare:
>
> - **Pre-trained + frozen** (the foundation-model recipe).
> - **Random-init + end-to-end** (no pre-training).

### What the results look like

```
 labels | pretrained acc  pretrained recall | baseline acc  baseline recall
------------------------------------------------------------------------------------
     20 |          0.905              0.000 |        0.905            0.000
     50 |          1.000              1.000 |        0.965            0.632
    200 |          1.000              1.000 |        0.980            0.789
   1000 |          1.000              1.000 |        1.000            1.000
```

Things to notice:

- **20 labels: both fail.** Even pre-training can't extract a signal from
  20 examples of a rare class. (`accuracy ≈ 0.91` is the "always predict
  engaged" baseline because 91% of users are engaged.)
- **50 labels: pre-training wins dramatically.** Frozen probe catches
  **100% of churners**. Baseline only catches 63%.
- **200 labels: pre-training still ahead.** 100% vs 79%.
- **1000 labels: they converge.** Once you have enough labels, even a
  randomly-initialised model can learn the pattern.

> 🔑 **The pre-trained encoder is most valuable when labelled data is
> scarce.** This is the entire foundation-model pitch — and the reason
> Revolut built PRAGMA. Labels are expensive; raw events are free.

---

## What you just saw, lesson by lesson

| Step | Lesson(s) used |
|---|---|
| Tokenise events into key-value pairs | L2, L5 |
| Build an encoder with embedding + position + attention layers | L1b, L2, L3 |
| Pre-train with fill-in-the-blank, no labels | L1, L1c, L4 |
| Freeze the encoder | L1b |
| Add a small classifier head on top | L1b |
| Train just the head on labelled data | L1, L1c |
| Compare to a random-init baseline | scientific method |

This is **PRAGMA**, just smaller. Same architecture choices. Same
pre-training objective. Same downstream-adaptation strategy.

---

## Now you're ready for the capstone

Next up: [Lesson 6 — Capstone](06_capstone.md). You build the same
workflow yourself, but for **fraud detection**. You design the synthetic
data, write the pre-training loop, build the classifier head, and prove
that pre-training mattered.

You've now seen this recipe applied to:
1. Cats and barks (Lesson 4)
2. Pets and places (`pragma_mini.py`, Lesson 5)
3. Streaming and churn (Lesson 5b, you just finished)

Your turn. 🚀
