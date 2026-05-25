# Lesson 2 — line-by-line walkthrough

Companion to `02_tokens_and_embeddings.py`. Read side-by-side with the code.

This lesson teaches **how text becomes numbers** so a model can work with it. There's no training in this file — it's all setup. Don't worry, we'll train again in Lesson 4.

---

### Lines 20–22: imports and reproducibility

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)
```

- `import torch` — same as before.
- `import torch.nn as nn` — pulls in PyTorch's "neural network" toolkit. This is where pre-built building blocks live (embedding tables, attention layers, linear layers, etc.). We'll use `nn.Embedding` below.
- `torch.manual_seed(0)` — reproducibility. Same as Lesson 1.
- `torch.set_printoptions(precision=2, sci_mode=False)` — when we print tensors, show only 2 decimal places and don't use scientific notation (no `1.23e-05`). Just cosmetic.

---

### Lines 30–32: the vocabulary

```python
vocab = ["<pad>", "<mask>", "dog", "cat", "fish", "bark", "meow", "swim"]
tok2id = {w: i for i, w in enumerate(vocab)}
```

**What it does:** defines all the "words" the model is allowed to know about.

- `vocab` is a Python list of 8 strings.
- `tok2id` is a Python dict mapping `"dog" → 2`, `"cat" → 3`, etc. The position in the list is the word's ID.

**Why this matters:** a computer can't multiply, subtract, or do gradient descent on the string `"dog"`. It can do all of that on the number `2`. So **step 1** of using ANY language model is: pick a fixed vocabulary, assign each word a unique integer ID.

**The two special tokens:**
- `<pad>` (ID 0) — a placeholder. If you have a batch of sentences of different lengths (3 words, 5 words, 4 words), you pad the short ones with `<pad>` until they're all the same length, so they can fit in a single tensor.
- `<mask>` (ID 1) — the "hidden" token. In Lesson 4 we'll hide some words by replacing them with `<mask>` and ask the model to guess what was there. This is the BERT training game.

**Mental model:** the vocabulary is a numbered list of flashcards. Each word in the world either has a flashcard or it doesn't. Real models like PRAGMA have ~28,000 flashcards. Ours has 8.

---

### Lines 41–43: encoding a sentence

```python
sentence = ["dog", "bark"]
ids = [tok2id[w] for w in sentence]
print(f"\nSentence {sentence}  ->  ids {ids}")
```

**What it does:** turns the sentence `["dog", "bark"]` into the list `[2, 5]`. Each word is looked up in `tok2id` and replaced by its ID.

This is **tokenisation**. The very first step of every language model pipeline. Real models do something fancier (BPE, subwords) but the principle is identical.

---

### Lines 55–56: the embedding table

```python
embedding_dim = 4
emb = nn.Embedding(len(vocab), embedding_dim)
```

**What it does:** creates a 2-D table of numbers. Specifically, an `8 × 4` table:
- **8 rows** — one per word in the vocab.
- **4 columns** — each word becomes a vector of 4 numbers.

The numbers start out **random**. They have no meaning yet.

**What is an embedding?** Think of it as giving each word a position in 4-dimensional space. Two words that get used in similar contexts will, *after training*, end up at similar positions. Two unrelated words will end up far apart. This is how a model learns "meaning" — it's just geometry in some high-dimensional space.

Real Transformers use embedding dimensions of 512, 768, 1024, or higher. PRAGMA-Large uses 1024. We use 4 because we can print all 4 numbers and look at them with our eyes.

**Why `nn.Embedding` is just a lookup:** the operation `emb(torch.tensor(2))` literally just returns row 2 of the table. That's it. No math. It IS a learnable table — during training, PyTorch will adjust the numbers in each row — but the "looking up" itself is a single indexing operation.

---

### Lines 58–59: inspecting the table

```python
print(f"\nEmbedding table shape: {tuple(emb.weight.shape)}")
print(f"  ({len(vocab)} words in vocab, each represented by {embedding_dim} numbers)")
```

`emb.weight` is the actual table. `.shape` tells us its dimensions: `(8, 4)`. We print it as a tuple just because it looks nicer.

---

### Lines 62–65: looking up a few words

```python
for word in ["dog", "cat", "fish"]:
    i = tok2id[word]
    v = emb(torch.tensor(i)).detach()
    print(f"  {word:5s} ->  {v}")
```

For each word: look up its ID, get its 4-number vector, print it.

Two tiny details:
- `torch.tensor(i)` — wraps the integer in a tensor, because `nn.Embedding` expects tensors as input (not raw Python ints).
- `.detach()` — says "I'm just looking, not training." Embeddings normally track gradients (so they can be trained); `.detach()` strips that tracking off, so PyTorch doesn't complain when we print.

Sample output:
```
dog   ->  tensor([ 0.32, -1.26,  0.35,  0.31])
cat   ->  tensor([ 0.12,  1.24,  1.12, -0.25])
fish  ->  tensor([-1.35, -1.70,  0.57,  0.79])
```

Those vectors are pure noise right now. After training (in Lesson 4 or in a real model), they'd start to encode meaning: `"dog"` and `"cat"` would drift toward similar values because they appear in similar contexts (both pets, both make sounds, both eat).

---

### Lines 71–75: encoding a full sentence

```python
ids_tensor = torch.tensor(ids)
vectors = emb(ids_tensor)
print(f"\nSentence vectors:  shape {tuple(vectors.shape)}  (2 words, 4 numbers each)")
print(vectors.detach())
```

**Key trick:** if you pass `nn.Embedding` a tensor of N IDs, it gives you back a tensor of shape `(N, embedding_dim)`. In our case: 2 words → shape `(2, 4)`.

This is how every Transformer starts:
- Input: a sequence of token IDs, e.g. `[2, 5]`.
- After embedding lookup: a sequence of vectors, e.g. a `(2, 4)` tensor.
- Everything from here on operates on vectors. The original strings are long forgotten.

---

### Lines 80–88: measuring word similarity

```python
def similarity(a, b):
    return torch.dot(emb(torch.tensor(tok2id[a])), emb(torch.tensor(tok2id[b]))).item()

print("\nSimilarity (random embeddings, all meaningless for now):")
print(f"  dog · cat   = {similarity('dog', 'cat'):.2f}")
print(f"  dog · fish  = {similarity('dog', 'fish'):.2f}")
print(f"  bark · meow = {similarity('bark', 'meow'):.2f}")
```

**What it does:** computes the *dot product* of two word vectors. A dot product is a single number that measures how much two vectors "agree":
- Big positive → vectors point in similar directions → words are similar.
- Near zero → vectors are unrelated.
- Big negative → vectors point in opposite directions.

Right now the numbers are random (e.g., `dog · cat = -1.21`). Meaningless. **But** if you trained these embeddings on real text:
- `dog · cat` would become a big positive number (both are pets).
- `bark · meow` would also become positive (both are animal sounds).
- `dog · couch` would be smaller.

Dot products are the heart of **attention** (Lesson 3) and the standard way to measure embedding similarity in production systems (search engines, recommendation systems, RAG).

---

## What you should take away

Three new ideas on top of Lesson 1:

> 1. **Tokenisation**: every word in the world gets a unique integer ID.
> 2. **Embedding**: every ID gets looked up in a table to produce a vector of numbers.
> 3. **The embedding table is learned**. It starts random; training reshapes it so similar words end up at similar positions in space.

The embedding lookup is the **first** thing that happens inside every Transformer, including PRAGMA. It's how raw text becomes numerical input the model can actually compute on.

In Lesson 3, we'll see what the model DOES with these vectors once it has them.
