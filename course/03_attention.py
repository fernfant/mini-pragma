"""
Lesson 3 — Attention: words looking at each other

This is the big idea of the Transformer. The paper that started it all was
literally called "Attention Is All You Need" (2017).

The intuition:
    When you read "the dog chased the cat because IT was scared", what does
    "it" refer to? Your brain looks at every word in the sentence and figures
    out that "it" probably refers to "cat" (or maybe "dog").

ATTENTION lets the model do the same thing: every word looks at every other
word and decides which ones are important for understanding itself.

Run:  python3 03_attention.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

# Pretend these vectors came out of the embedding lookup from Lesson 2.
words = ["the", "dog", "chased", "cat"]
d = 4                                     # vector dimension
x = torch.randn(len(words), d)            # shape (4 words, 4 numbers each)

print("Input vectors x (one row per word):")
print(x)

# ----------------------------------------------------------------------------
# Naïve attention — the simplest possible version
# ----------------------------------------------------------------------------
# For each pair of words, compute a SCORE = how much they match.
# The simplest score: the dot product of their vectors.
scores = x @ x.T                          # shape (4, 4)

# Convert scores into probabilities with softmax (each row sums to 1).
# Now row i says: "of all the other words, how much should I focus on each?"
attention = F.softmax(scores, dim=-1)

print("\nAttention weights (each row is a probability distribution):")
print("              " + "  ".join(f"{w:>7s}" for w in words))
for i, w in enumerate(words):
    print(f"  {w:10s}  " + "  ".join(f"{a:7.3f}" for a in attention[i]))

# Each word's NEW vector = weighted average of all word vectors.
# So now every word's representation has information about its neighbours.
new_x = attention @ x
print("\nOutput vectors (after attention — each word now 'knows' about the others):")
print(new_x)

# ----------------------------------------------------------------------------
# Real attention — Query, Key, Value
# ----------------------------------------------------------------------------
# The naïve version uses the same vector for "what am I looking for" and
# "what do I offer". Real attention separates these with three learned
# projections:
#
#     Q (query) — "what am I looking for?"
#     K (key)   — "what do I offer to others looking?"
#     V (value) — "what do I actually contribute if matched?"
#
# Each is just `x` multiplied by a learned matrix.

# These three matrices would be LEARNED during training. We initialise them
# randomly here just to show the shape of the computation.
Wq = torch.randn(d, d)
Wk = torch.randn(d, d)
Wv = torch.randn(d, d)

Q = x @ Wq                                # (4, 4)
K = x @ Wk                                # (4, 4)
V = x @ Wv                                # (4, 4)

# Score every query against every key.
# Divide by sqrt(d) to keep numbers from getting huge — this is the
# "scaled" in "scaled dot-product attention".
scores = Q @ K.T / (d ** 0.5)
attention = F.softmax(scores, dim=-1)
output = attention @ V                    # apply weights to values

print("\nReal scaled dot-product attention output:")
print(output.detach())

# That's it. That's the formula at the heart of every modern AI model —
# from GPT to BERT to PRAGMA:
#
#     Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
#
# Stack a few of these on top of each other (with some normalisation and
# feed-forward layers in between) and you have a Transformer.

# ----------------------------------------------------------------------------
# Multi-head attention (briefly)
# ----------------------------------------------------------------------------
# Instead of computing attention once, you compute it several times in
# parallel with different Q/K/V matrices and concatenate the outputs. Each
# "head" can specialise in a different kind of relationship — one might
# attend to nearby words, another to verbs, another to subject-object
# patterns. PRAGMA-L uses 16 heads.

# ----------------------------------------------------------------------------
# Exercises
# ----------------------------------------------------------------------------
# 1. Add two more words ("the", "mouse") for a 6-word sentence. What's the
#    shape of the attention matrix?
#
# 2. Replace `x = torch.randn(...)` with vectors where words 1 and 3 are
#    identical (e.g., x[1] = x[3]). What do you notice in the attention
#    matrix? (Words that look alike attend to each other strongly.)
#
# 3. The softmax has a property: large scores become very close to 1, and
#    small scores become very close to 0. Try printing `scores` and
#    `attention` side by side to see this.
#
# 4. (Stretch) Why do we divide by sqrt(d) in the scaled version? Hint:
#    dot products of long random vectors get big — and softmax of big
#    numbers becomes too sharp (one-hot), which makes learning hard.
