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

Mechanically, attention is THREE STEPS:

  1. SCORES  — for every pair of words, compute their dot product (Lesson 2).
               Big dot product = related.
  2. SOFTMAX — turn each row of scores into a "100% attention budget"
               (percentages that sum to 1). The biggest score wins most.
  3. MIX     — each word's new vector = a weighted blend of all word vectors,
               weighted by its attention budget. Now each word's vector has
               absorbed context from the others.

For a tiny worked example you can do on paper (just 2 words, 2-dim
embeddings, every multiplication shown), see the companion walkthrough:

  03_walkthrough.md  (read it first if any of this feels abstract)

The code below does the same three steps with 4 words and 4-dim embeddings.

Run:  python3 03_attention.py
"""
import torch                                    # PyTorch tensors.
import torch.nn.functional as F                 # F.softmax used below.

torch.manual_seed(0)                            # Reproducible RNG.
torch.set_printoptions(precision=2, sci_mode=False)     # Clean tensor printing.

# Pretend these vectors came out of the embedding lookup from Lesson 2.
words = ["the", "dog", "chased", "cat"]         # 4-word example sentence.
d = 4                                            # Vector dimension per word.
x = torch.randn(len(words), d)                  # Random "embeddings". Shape (4, 4).

print("Input vectors x (one row per word):")
print(x)

# ----------------------------------------------------------------------------
# Naïve attention — the simplest possible version
# ----------------------------------------------------------------------------

# STEP 1 — SCORES. For every pair of words, compute their dot product.
# x @ x.T does all 16 pair-wise dot products at once. Big number = related.
# (Same trick we did by hand for "the · cat" in the walkthrough.)
scores = x @ x.T                                # Shape (4, 4). scores[i,j] = x[i] · x[j].

# STEP 2 — SOFTMAX. Turn each row of raw scores into percentages summing to 1.
# Each row is now a "100% attention budget" — how much this word spends on
# each other word. Biggest score in the row wins most of the budget.
attention = F.softmax(scores, dim=-1)           # dim=-1 = softmax along each row.

print("\nAttention weights (each row is a probability distribution):")
print("              " + "  ".join(f"{w:>7s}" for w in words))
for i, w in enumerate(words):                   # Print one row per word.
    print(f"  {w:10s}  " + "  ".join(f"{a:7.3f}" for a in attention[i]))

# STEP 3 — MIX. Each word's new vector is a weighted blend of all word vectors,
# using its attention budget as the weights. Words that started bland
# (small self-score) absorb flavor from their neighbours. Strong words
# (big self-score) stay mostly themselves.
new_x = attention @ x                           # Weighted blend. Same shape (4, 4).
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
Wq = torch.randn(d, d)                          # Query projection matrix.
Wk = torch.randn(d, d)                          # Key projection matrix.
Wv = torch.randn(d, d)                          # Value projection matrix.

Q = x @ Wq                                      # Apply query lens. Shape (4, 4).
K = x @ Wk                                      # Apply key lens.
V = x @ Wv                                      # Apply value lens.

# Score every query against every key.
# Divide by sqrt(d) to keep numbers from getting huge — this is the
# "scaled" in "scaled dot-product attention".
scores = Q @ K.T / (d ** 0.5)                   # Scaled dot-product scores.
attention = F.softmax(scores, dim=-1)           # Convert to attention weights.
output = attention @ V                          # Apply weights to V.

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
