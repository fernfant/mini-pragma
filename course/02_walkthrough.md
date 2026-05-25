# Lesson 2 — line-by-line walkthrough

Companion to `02_tokens_and_embeddings.py`. Read side-by-side with the code.

This lesson teaches **how text becomes numbers** so a model can work with it. There's no training in this file — it's all setup. Don't worry, we'll train again in Lesson 4.

---

## 🍕 Before the code: the big idea (forget words for a moment)

The leap from "the word `dog`" to "the list of numbers `[0.32, -1.26, 0.35, 0.31]`" is huge. Let's build the intuition with something concrete first.

### Describing a person as a list of numbers

How much (0 to 10) do you like each of these?

- Ice cream: ___
- Soccer: ___
- Video games: ___
- Cleaning your room: ___

Say you answered **[10, 7, 9, 1]**. Congratulations — *you* are now a list of 4 numbers.

Your best friend might be **[9, 8, 10, 2]** — very similar. Grandma might be **[4, 0, 0, 9]** — very different.

| | Ice cream | Soccer | Games | Cleaning |
|---|---|---|---|---|
| **You** | 10 | 7 | 9 | 1 |
| **Friend** | 9 | 8 | 10 | 2 |
| **Grandma** | 4 | 0 | 0 | 9 |

### How does a computer tell who's similar to whom?

The computer only sees the number lists. It needs a math formula that turns *"are these number lists alike?"* into a single answer.

**The formula is dumb-simple. Multiply matching slots. Add up the products.** That's the *dot product*.

```
You · Friend  =  10×9 + 7×8 + 9×10 + 1×2  =  90 + 56 + 90 + 2  =  238    (big = similar!)
You · Grandma =  10×4 + 7×0 + 9×0 + 1×9   =  40 +  0 +  0 + 9  =   49    (small = not similar)
```

**Why this works**:

- When you BOTH score high on a thing, the product is big (`10 × 9 = 90`) → strong agreement → boosts the score.
- When one scores high and the other scores low, the product is small (`9 × 0 = 0`) → disagreement → doesn't help the score.
- Big total = "you agree about lots of things" = similar. Small total = not similar.

### Now the punchline — words work the exact same way

Every word in the computer's vocabulary gets a list of numbers. Pretend `dog` looks like this (if the slots had human-readable names):

| | is_animal | makes_sound | is_pet | is_furry |
|---|---|---|---|---|
| `dog` | 8 | 7 | 9 | 9 |
| `cat` | 8 | 6 | 9 | 9 |
| `bark` | 0 | 9 | 0 | 0 |
| `spaceship` | 0 | 1 | 0 | 0 |

`dog` and `cat` have nearly identical lists → big dot product → SIMILAR. `dog` and `spaceship` have tiny overlap → small dot product → NOT similar. The computer can tell *purely from the numbers*, without ever knowing what the words mean.

### The mind-blowing twist

In real AI, **the slots aren't given human-readable names**. The computer doesn't know what "is_furry" means. It just sees:

```
dog  →  [0.32, -1.26, 0.35, 0.31]
```

Four mystery numbers. **The computer invents its own categories** during training. Maybe slot 1 ends up meaning "animal-ness" — but nobody told it to. Maybe slot 2 means something humans can't put a name on at all. It doesn't matter. What matters: *similar words end up with similar number lists*, so the dot product comes out big.

These mystery-number lists are called **embeddings**. Each word's list is its embedding vector. The size of the list (4 in our toy, 1024 in PRAGMA-Large) is called the *embedding dimension*.

OK — now the code makes sense. Let's go.

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

**What it does:** computes the *dot product* of two word vectors — same math we did with you-vs-friend-vs-grandma at the top of this file. Multiply matching slots, add the products.

Let's actually do one by hand. From the program's output, suppose:
```
dog  =  [ 0.32, -1.26,  0.35,  0.31]
fish =  [-1.35, -1.70,  0.57,  0.79]
```

Then:
```
dog · fish  =  (0.32 × -1.35) + (-1.26 × -1.70) + (0.35 × 0.57) + (0.31 × 0.79)
            =  -0.43 + 2.14 + 0.20 + 0.24
            =   2.15
```

Same arithmetic, exactly. Just with smaller, signed numbers (the embeddings can go negative — that just means "below average on this hidden feature").

What the result means:
- **Big positive** → vectors point in similar directions → words are similar.
- **Near zero** → vectors are unrelated.
- **Big negative** → vectors point in opposite directions (one has "high" where the other has "low" on the same hidden feature).

Right now the numbers are random (e.g., `dog · cat = -1.21`, which falsely says "dog and cat have nothing in common"). Meaningless. **But** if you trained these embeddings on real text:
- `dog · cat` would slide up to become a big positive number (both are pets).
- `bark · meow` would also become big positive (both are animal sounds).
- `dog · couch` would stay smaller.

Dot products are the heart of **attention** (Lesson 3) and the standard way to measure embedding similarity in production systems (search engines, recommendation systems, RAG).

---

## What you should take away

Three new ideas on top of Lesson 1:

> 1. **Tokenisation**: every word in the world gets a unique integer ID.
> 2. **Embedding**: every ID gets looked up in a table to produce a vector of numbers.
> 3. **The embedding table is learned**. It starts random; training reshapes it so similar words end up at similar positions in space.

The embedding lookup is the **first** thing that happens inside every Transformer, including PRAGMA. It's how raw text becomes numerical input the model can actually compute on.

In Lesson 3, we'll see what the model DOES with these vectors once it has them.
