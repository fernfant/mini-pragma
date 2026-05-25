# Lesson 3 — line-by-line walkthrough

Companion to `03_attention.py`. Read side-by-side with the code.

This is **the** big lesson. Attention is the heart of the Transformer. Once you've understood this file, every modern AI model — GPT, BERT, PRAGMA, image transformers, audio transformers, all of them — uses some variation of what's in here.

---

### Lines 18–22: imports

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)
```

`torch.nn.functional` (conventionally aliased to `F`) is the "functions" half of PyTorch's neural network toolkit. We'll use `F.softmax` below.

Everything else is the same setup from Lesson 1/2.

---

### Lines 26–27: pretend we already embedded a sentence

```python
words = ["the", "dog", "chased", "cat"]
d = 4
x = torch.randn(len(words), d)
```

**What it does:** creates a `(4, 4)` tensor of random numbers — pretending that these are the embedding vectors for the 4 words "the", "dog", "chased", "cat".

In a real model these would come from `nn.Embedding(...)` like in Lesson 2. We skip that step and just inject random vectors because we're focused on what comes *after* the embedding lookup.

- 4 rows = 4 words.
- 4 columns = each word is represented by 4 numbers (its embedding).

`torch.randn` generates random numbers from a normal distribution (mean 0, variance 1). It's how PyTorch initialises weights by default.

---

### Lines 33–35: scores — every pair of words gets a number

```python
scores = x @ x.T
```

**One line of code, lots of meaning.** Let's slow down.

- `x` has shape `(4, 4)`: 4 words, each a 4-dim vector.
- `x.T` is the *transpose* — shape `(4, 4)` becomes `(4, 4)` (still, because it's square), but rows and columns swap roles. Think of it as "tipped on its side".
- `@` is the **matrix multiplication** operator in Python. `x @ x.T` computes a `(4, 4)` matrix where:

```
scores[i][j] = (vector for word i)  ·  (vector for word j)
```

In other words: **scores[i][j] is the dot product between word i and word j**. Remember from Lesson 2 — dot product measures how similar two vectors are.

So `scores` is a 4×4 grid of "how much does word i resemble word j?" numbers:

|        | the | dog | chased | cat |
|--------|-----|-----|--------|-----|
| **the**    | ?   | ?   | ?      | ?   |
| **dog**    | ?   | ?   | ?      | ?   |
| **chased** | ?   | ?   | ?      | ?   |
| **cat**    | ?   | ?   | ?      | ?   |

The diagonal (`the·the`, `dog·dog`, …) will be biggest — a word looks most like itself.

**Why "matrix multiplication" computes all pair-wise dot products at once:** when you multiply two matrices `A` and `B`, the entry at row i, column j of the result is the dot product of row i of A and column j of B. Since `x.T`'s columns are `x`'s rows, the result's `[i][j]` is `x[i] · x[j]`. Math just happens to be set up to give us exactly what we want in one operation.

This is one of those "tricks" that makes Transformers run fast on GPUs — they're built to do giant matrix multiplications insanely quickly.

---

### Line 38: softmax — scores become probabilities

```python
attention = F.softmax(scores, dim=-1)
```

**What softmax does:** takes a row of numbers (positive, negative, anything) and turns them into a row of probabilities (all positive, all summing to 1).

If you feed in `[2.0, 1.0, 0.5, -3.0]`, you get back something like `[0.65, 0.24, 0.10, 0.01]` — a probability distribution. Bigger inputs get bigger outputs. Very negative inputs get squashed to nearly 0.

**`dim=-1`** tells softmax: "apply this to the LAST axis." For a 2-D tensor, that means each *row* gets its own softmax, independently. So each row of `attention` is a probability distribution.

**What we now have:** a 4×4 matrix where row `i` answers the question:

> "When I'm word `i`, how much should I pay attention to each of the 4 words?"

The row sums to 1 — like a budget. Word `i` has 100% attention to distribute, and softmax says how to spread it across the other words.

Sample output from the actual run:
```
                the      dog   chased      cat
  the         0.850    0.023    0.118    0.009
  dog         0.001    0.988    0.001    0.009
  chased      0.245    0.026    0.698    0.031
  cat         0.008    0.148    0.014    0.830
```

Each row sums to 1. "Dog" spends 98.8% of its attention on itself (because in our random setup, it happens to be most similar to itself). In a trained model, words would learn to spread their attention more usefully.

---

### Line 43: applying the weights

```python
new_x = attention @ x
```

**One more matrix multiplication.** Let's understand what it's computing.

- `attention` has shape `(4, 4)` — 4 rows of weights.
- `x` has shape `(4, 4)` — 4 word vectors.
- Result `new_x` has shape `(4, 4)` — 4 new word vectors.

For each word `i`:

```
new_x[i] = attention[i][0] * x[0]
        + attention[i][1] * x[1]
        + attention[i][2] * x[2]
        + attention[i][3] * x[3]
```

This is a **weighted average** of all the word vectors, where the weights come from the attention row for word `i`. Each word's new vector is a blend of all the other word vectors, weighted by how much it cares about each.

**This is the moment information mixes.** Before attention, each word's vector knows nothing about the other words. After attention, each word's vector is a custom mixture of *all* the words. This is how the model gets contextual understanding: "the word 'bank' near 'river' gets a different vector than 'bank' near 'money'."

---

### Lines 56–58: real attention uses three projections

```python
Wq = torch.randn(d, d)
Wk = torch.randn(d, d)
Wv = torch.randn(d, d)
```

The naïve version uses the same vector for "what am I looking for" and "what do I offer." Real attention separates these. We create three `(d, d)` matrices of random numbers — they're the three lenses the model will look through:

- `Wq` — the **Query** projection: "transform x into 'what I'm looking for'."
- `Wk` — the **Key** projection: "transform x into 'what I offer to others looking'."
- `Wv` — the **Value** projection: "transform x into 'what I actually contribute when matched'."

In a real model, **these matrices are learned** — they have `requires_grad=True` and get nudged during training, just like `w` and `b` in Lesson 1. Here we just use random ones to demonstrate the shape of the computation.

---

### Lines 60–62: applying the three lenses

```python
Q = x @ Wq
K = x @ Wk
V = x @ Wv
```

Apply each projection to all 4 word vectors. Each is matrix multiplication, each produces a `(4, 4)` tensor (4 words, still 4 dims each).

Now we have **three different versions of the same sentence**:
- `Q[i]` is "what word i is looking for in others"
- `K[i]` is "what word i offers to others looking at it"
- `V[i]` is "what word i contributes to the final answer if matched"

The whole point: the model gets to LEARN how to convert each word into a query, a key, and a value separately. This is much more flexible than just using the raw embedding for everything.

---

### Lines 67–69: scaled dot-product attention

```python
scores = Q @ K.T / (d ** 0.5)
attention = F.softmax(scores, dim=-1)
output = attention @ V
```

Same three steps as the naïve version, but now using Q, K, V instead of x, x, x:

1. **`Q @ K.T`** — dot every query against every key. Now we're matching "what I'm looking for" against "what others offer".
2. **`/ (d ** 0.5)`** — divide by the square root of `d`. This is the "scaled" part. Why? Random vectors of length 4 have dot products of size roughly √4 = 2. Vectors of length 1024 have dot products of size roughly 32. Big dot products make softmax very "spiky" (one number near 1, the rest near 0), which gives tiny gradients and hurts learning. Dividing by √d keeps the scores in a sensible range regardless of how big you make your embeddings.
3. **`F.softmax(scores, dim=-1)`** — turn scores into probability rows.
4. **`attention @ V`** — weighted sum of value vectors using the attention weights.

The whole formula in math notation:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

**Memorise this.** It's the single most important formula in modern AI. It's what GPT, BERT, PRAGMA, image transformers, all of them, are built around. Every architectural variation you've heard of (multi-head attention, sparse attention, sliding window, FlashAttention) is just a tweak on this one equation.

---

### The "multi-head attention" mention

```python
# Multi-head attention runs several Q/K/V's in parallel and concatenates outputs.
# Each "head" can specialise in a different kind of relationship.
```

**In one sentence:** instead of running the attention formula once with full-size Q/K/V, you split it into several smaller "heads" (typically 8, 16, or more) and run them in parallel. Each head can learn to focus on a different kind of pattern — one head might learn grammar, another might learn subject-object pairing, another might learn nearby words.

You concatenate the outputs of all heads at the end. The total compute is the same as one big attention, but the model gets multiple "views" of the relationships.

PRAGMA-Large uses 16 attention heads in its history encoder. Our toy doesn't bother — one head is enough for animals.

---

## Why attention is such a big deal

Before attention (~2017), models read text **left to right**, one word at a time. Each word could only "remember" what came before, and even then only fuzzily (the further back, the more forgotten). Sentences with long-range dependencies — like "the dog *that the cat next door barked at all morning* finally bit me" — were really hard.

Attention solved this by saying: **every word can look at every other word, in parallel, directly.** No fuzzy memory. No left-to-right bottleneck. A word at the end of the sentence has equal access to the word at the start as it does to the word right before it.

It also turned out to be massively parallelisable on GPUs — which meant you could scale models up much further than before. That's the whole reason "the Transformer" became the dominant architecture.

---

## What you should take away

> 1. **Attention lets every word look at every other word and decide what's relevant.**
> 2. **Mechanically, it's: turn each vector into Q, K, V → dot-product Q against K → softmax → use as weights to mix V.**
> 3. **The formula `softmax(Q·Kᵀ/√d) · V` is the heart of every modern AI model.**

Now you're ready for Lesson 4, where we put attention + embedding + a training loop together into a real (if tiny) Transformer.
