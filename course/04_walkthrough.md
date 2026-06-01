# Lesson 4 — line-by-line walkthrough

Companion to `04_transformer_and_mlm.py`. Read side-by-side with the code.

This is the big payoff. We combine everything from Lessons 1–3:
- **Lesson 1**: the training loop (guess → loss → backward → step).
- **Lesson 2**: tokens and embeddings (text → IDs → vectors).
- **Lesson 3**: attention (vectors looking at each other).

…and turn it into a tiny working Transformer that learns the fill-in-the-blank game.

---

### Lines 22–25: imports and seeds

```python
import random
import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)
```

Nothing new. We also seed Python's `random` module because we'll use it (in `make_example` below) to pick which token to hide.

---

### Lines 30–33: vocab — same as Lesson 2

```python
vocab = ["<pad>", "<mask>", "dog", "cat", "fish", "bark", "meow", "swim"]
tok2id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
```

We give `len(vocab)` a name (`V`) because we'll use it in a few places (size of the embedding table, size of the output layer).

---

### Line 35: the patterns the model must discover

```python
PAIRS = [("dog", "bark"), ("cat", "meow"), ("fish", "swim")]
```

These are the hidden rules. The model never sees this list directly — it only sees individual training examples. The point of training is for the model to figure out these patterns on its own.

---

### Lines 40–47: making a training example

```python
def make_example():
    animal, sound = random.choice(PAIRS)
    ids = [tok2id[animal], tok2id[sound]]
    labels = [-100, -100]
    hide_position = random.choice([0, 1])
    labels[hide_position] = ids[hide_position]
    ids[hide_position] = tok2id["<mask>"]
    return ids, labels
```

This is the **fill-in-the-blank game** in code. Let's walk through one call:

1. Pick a random pair, say `("dog", "bark")`.
2. Convert to IDs: `ids = [2, 5]`.
3. Initialise `labels = [-100, -100]`. The `-100` is a magic number — `CrossEntropyLoss` will skip any position where the label is `-100`. So `-100` means "I'm not asking the model to predict this position; just leave it alone."
4. Pick a random position to hide — either 0 or 1.
5. Save the original ID at that position into `labels` — this is the answer key.
6. Replace the original ID at that position with `tok2id["<mask>"]` — this is what the model will actually see.

After this, we might return:
- `ids = [1, 5]` (where 1 is `<mask>`) and `labels = [2, -100]` — "guess what should go in position 0; we hid 'dog' there."
- OR `ids = [2, 1]` and `labels = [-100, 5]` — "guess position 1; we hid 'bark' there."

This is **the** training signal that BERT-style models use, including PRAGMA. No human labels anything. The data labels itself, because we know which word we hid.

---

### Lines 56–67: the model

```python
class TinyMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.enc  = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            batch_first=True,
        )
        self.head = nn.Linear(16, V)

    def forward(self, x):
        h = self.emb(x)
        h = self.enc(h)
        return self.head(h)
```

This is our first time defining a model as a Python class. Let's go piece by piece.

> 🧰 **Concept bridge — where each piece comes from**
>
> | TinyMLM component | Lesson where introduced |
> |---|---|
> | `nn.Embedding(V, 16)` | **[L2](02_walkthrough.md)** — token → vector lookup table |
> | `nn.TransformerEncoderLayer` (attention part) | **[L3](03_walkthrough.md)** — every word looks at every other |
> | `nn.TransformerEncoderLayer` (feed-forward part) | **[L1.5 Step 7.5](notebooks/lesson_01_5_from_linear_to_neural.ipynb)** — same MLP you trained on the parabola |
> | `nn.Linear(16, V)` (output head) | **[L1](01_walkthrough.md)** — linear projection (`y = Wx + b`) |
> | The 5-line training loop below | **[L1](01_walkthrough.md)** — predict → loss → zero_grad → backward → step |
> | Cross-entropy loss | **[L1c](01c_gradient_descent.md)** — for classification (not MSE) |
>
> **You've already learned every piece.** Nothing in `TinyMLM` is new — we're just putting the pieces together.

**`class TinyMLM(nn.Module):`** — by inheriting from `nn.Module`, we get:
- automatic parameter tracking (so the optimiser can find all the weights to tune),
- a `.to(device)` method for moving to GPU,
- a `.eval()` / `.train()` mode toggle,
- and a few other niceties. Every neural network in PyTorch inherits from `nn.Module`.

**Three sub-modules:**

1. **`self.emb = nn.Embedding(V, 16)`** — the embedding table from Lesson 2. 8 words, each represented by 16 numbers. (We use 16 dims instead of 4 because we'll be doing attention over them and want a bit more capacity.)

2. **`self.enc = nn.TransformerEncoderLayer(...)`** — one Transformer block. PyTorch has it built in, so we don't have to write attention by hand. The arguments:
   - `d_model=16` — input/output dimension. Must match our embedding dim.
   - `nhead=2` — 2 attention heads (see Lesson 3's multi-head explanation).
   - `dim_feedforward=32` — inside every Transformer block, there's an attention layer AND a small feed-forward neural network. `dim_feedforward` is the size of that internal feed-forward layer. Usually 2-4x larger than `d_model`.
   - `batch_first=True` — tells PyTorch that the first dimension of input tensors is the batch (not the sequence). The other convention (sequence first) was historical and confusing; `batch_first=True` is now the standard.

   What's inside this one line? Roughly:
   ```
   input
     → multi-head attention
     → add input back (residual connection)
     → normalise
     → feed-forward neural network
     → add previous back (another residual)
     → normalise
   output
   ```
   This is the canonical "Transformer block" from the 2017 paper, all packaged into one PyTorch call.

3. **`self.head = nn.Linear(16, V)`** — a regular linear layer (essentially `y = Wx + b`, but with W being a matrix). It maps every 16-dim vector back to V=8 numbers — one "score" per possible vocab word. The biggest score is the model's guess.

**`def forward(self, x):`** — defines what happens when you call `model(x)`. Three lines:

1. `h = self.emb(x)` — turn token IDs into vectors. Shape goes from `(batch, seq_len)` to `(batch, seq_len, 16)`.
2. `h = self.enc(h)` — run the attention block. Shape stays `(batch, seq_len, 16)`. Each word's vector now incorporates info from the other words.
3. `return self.head(h)` — project each word's 16-dim vector to 8 vocab scores. Shape becomes `(batch, seq_len, V)`. These are called "logits" — raw, un-normalised scores.

That's our entire model. Three lines of computation. Notice how short it is — PyTorch's pre-built modules do a lot of the heavy lifting.

---

### Lines 71–79: the training setup

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
```

**`AdamW`** is an upgraded version of SGD (the optimiser from Lesson 1). It's smarter about adjusting the step size for each parameter individually, and converges faster on most problems. It's the default for Transformer training. PRAGMA also uses AdamW (combined with another optimiser called Muon, per the paper).

`model.parameters()` returns ALL the tunable numbers in the model — the embedding table, every weight matrix inside the Transformer block, the final linear head. `nn.Module`'s parameter tracking is what makes this one-liner possible.

`lr=3e-3` = `0.003`. A common learning rate for tiny models.

**`CrossEntropyLoss`** is the standard loss for "pick one out of N classes" tasks. It does two things:
1. Applies softmax to the model's logits to turn them into probabilities.
2. Computes `-log(probability_of_the_correct_answer)`. If the model gave the correct answer high probability, the loss is small. If it gave it low probability, the loss is high.

`ignore_index=-100` is what lets us use `-100` in our labels to mean "skip this position." Loss positions with label `-100` contribute nothing.

---

### Lines 81–93: the training loop

```python
for step in range(500):
    batch = [make_example() for _ in range(32)]
    ids, labels = zip(*batch)
    x = torch.tensor(ids)
    y = torch.tensor(labels)

    logits = model(x)
    loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(f"  step {step:3d}   loss {loss.item():.3f}")
```

**This is the exact same loop from Lesson 1.** Look at the inner five lines:

```python
logits = model(x)             # guess
loss   = loss_fn(...)         # how wrong?
opt.zero_grad()               # clear yesterday's notes
loss.backward()               # which way is less wrong?
opt.step()                    # nudge
```

Identical structure. The only differences are:
- `model(x)` instead of `w * x + b` — the "rule" is a Transformer now instead of a line.
- `loss_fn(...)` instead of `((y_pred - y_true) ** 2).mean()` — classification loss instead of regression loss.
- Otherwise: same loop.

**A few new bits:**

- **`batch = [make_example() for _ in range(32)]`** — instead of training on one example at a time, we use **batches** of 32. Faster, and gradients are less noisy when averaged across many examples.
- **`ids, labels = zip(*batch)`** — Python trick to "unzip" a list of pairs into two parallel lists.
- **`torch.tensor(ids)`** — stack the 32 examples into one big tensor of shape `(32, 2)`. Now we can do everything in one batched pass through the model.
- **`logits.reshape(-1, V)`** — `logits` has shape `(32, 2, V)`. The loss function expects `(N, V)` and `(N,)`. So we flatten the first two dimensions into one. `-1` means "compute this dimension from the others" — it works out to 64.
- **`y.reshape(-1)`** — same flattening for the labels: `(32, 2)` → `(64,)`.

The `-100` positions in `y` get skipped by `CrossEntropyLoss`, so only the positions we actually masked contribute to the loss. The model gets gradient signal only on the positions where we asked it to guess.

---

### Lines 97–105: testing the trained model

```python
def fill_blank(animal_or_sound, position):
    ids = [tok2id["<mask>"], tok2id["<mask>"]]
    ids[1 - position] = tok2id[animal_or_sound]
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)
    return vocab[logits[0, position].argmax().item()]
```

A tiny inference function. Given a word and which position to fill:
1. Start with both positions masked.
2. Put the known word in the OTHER position.
3. Wrap in a batch of 1 (`torch.tensor([ids])` adds an outer dimension).
4. **`with torch.no_grad():`** — tell PyTorch "I'm not training, don't bother tracking gradients." Saves memory and runs slightly faster.
5. Run the model.
6. **`logits[0, position]`** — pull out the logits for the position we want. Shape `(V,)` — one score per vocab word.
7. **`.argmax()`** — find the index of the biggest score. That's the model's top guess.
8. Look up that index in `vocab` to get the word back.

Sample output after training:
```
Fill in the blank:
  dog   -> bark
  cat   -> meow
  fish  -> swim
And going the other way:
  ?     -> bark    (model guesses: dog)
  ?     -> meow    (model guesses: cat)
  ?     -> swim    (model guesses: fish)
```

The model has learned the pairs. It can fill in either direction — animal given sound, or sound given animal — because it was trained on both with equal probability.

This is BERT in 30 lines of model code.

---

## What you should take away

You now know how to:
- Define a Transformer model with three building blocks (embed, encode, project).
- Generate masked-token training examples from raw data.
- Train using the standard 5-line PyTorch loop.
- Use the trained model to fill in blanks.

These are exactly the same pieces that go into `pragma_mini.py` (which uses a richer vocab and longer sequences) and into real PRAGMA at billion-event scale.

### The big-picture summary in one diagram

```
   raw data
      ↓                 ← Lesson 2: vocab + tokenisation
   token IDs
      ↓                 ← Lesson 2: nn.Embedding lookup
   vectors
      ↓                 ← Lesson 3: attention (TransformerEncoderLayer)
   contextualised vectors
      ↓                 ← Lesson 4: nn.Linear "MLM head"
   vocab scores (logits)
      ↓                 ← Lesson 4: CrossEntropyLoss vs masked labels
   loss
      ↓                 ← Lesson 1: backward + step
   slightly better model
```

Now go re-read `pragma_mini.py`. Every line will make sense.

Move on to **`05_putting_it_together.md`** for a guided re-reading and a tour of what real PRAGMA adds on top.
