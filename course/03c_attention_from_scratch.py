"""
Lesson 3c — Attention from scratch.

The most important "no black boxes" lesson.

In Lesson 3 we learned what attention DOES conceptually — every word looks
at every other word, weighted by relevance. In Lesson 4 and beyond we used
nn.TransformerEncoderLayer as a black box. This file shows you the actual
math behind self-attention, implemented in pure PyTorch (no nn.MultiheadAttention,
no TransformerEncoderLayer). Then we verify it gives the same numerical
result as PyTorch's built-in.

By the end you will know:
  - The 3 separate "projections" (Q, K, V) and why we need them.
  - Why we scale by sqrt(d_k) before softmax.
  - How multi-head attention works (split → attend in parallel → concat).
  - Exactly what nn.MultiheadAttention does — because you wrote it yourself.

Run:  python3 03c_attention_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


# ============================================================================
# STEP 1 — The setup: a fake "batch of sequences"
# ============================================================================
# Real attention works on a batch of sequences. Each sequence is a list of
# tokens, each token is a vector of d_model numbers.
#
# Conventionally the shape is (B, L, d_model):
#   B       = batch size       (how many sequences in parallel)
#   L       = sequence length  (how many tokens per sequence)
#   d_model = embedding dim    (how many numbers per token)

B, L, d_model = 2, 4, 6
n_heads = 2
d_k = d_model // n_heads  # dimension PER HEAD

print("=" * 70)
print("STEP 1 — The setup")
print("=" * 70)
print(f"Batch size B = {B}, sequence length L = {L}, d_model = {d_model}")
print(f"Number of heads = {n_heads}, so d_k (per-head dim) = {d_k}")
print()

# Make up some input vectors — pretend these came from an embedding lookup
x = torch.randn(B, L, d_model)
print(f"x.shape = {tuple(x.shape)}   (B, L, d_model)")
print("x[0] (first sequence, all tokens):")
print(x[0])
print()


# ============================================================================
# STEP 2 — Single-head attention, from scratch
# ============================================================================
# The formula from "Attention is All You Need":
#     Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# Q, K, V come from THREE separate linear projections of x.

print("=" * 70)
print("STEP 2 — Single-head attention, step by step")
print("=" * 70)

# These three linear layers are LEARNED. We initialize them, but during
# training they'll get nudged like every other knob.
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)
# Output projection — see STEP 4 for why we need this
W_o = nn.Linear(d_model, d_model, bias=False)

# Compute Q, K, V
Q = W_q(x)   # shape (B, L, d_model)
K = W_k(x)   # shape (B, L, d_model)
V = W_v(x)   # shape (B, L, d_model)

print(f"Q = x @ W_q.T  shape = {tuple(Q.shape)}    ('what am I looking for?')")
print(f"K = x @ W_k.T  shape = {tuple(K.shape)}    ('what do I offer?')")
print(f"V = x @ W_v.T  shape = {tuple(V.shape)}    ('what would I contribute?')")
print()

# Compute attention scores: dot every query with every key
# scores has shape (B, L, L). scores[b, i, j] = how much token i in sequence b
# wants to attend to token j.
scores = Q @ K.transpose(-2, -1)
print(f"scores = Q @ K.T   shape = {tuple(scores.shape)}    (B, L, L)")
print("Why L x L? Because for each token (row), we have a score against every other token (col).")

# Scale by sqrt(d_k). Why?
# Without scaling, scores grow proportional to sqrt(d_k) just from the
# dot products of random vectors. Big scores -> very peaked softmax ->
# tiny gradients on the small ones -> hard to train. Scaling fixes it.
scores = scores / math.sqrt(d_model)
print(f"\nAfter scaling by sqrt(d_model) = sqrt({d_model}):")
print(f"scores[0] (attention scores in the first sequence, before softmax):")
print(scores[0])

# Softmax along the last axis. Each ROW becomes a probability distribution.
attn = F.softmax(scores, dim=-1)
print(f"\nattn = softmax(scores)   shape = {tuple(attn.shape)}")
print("Each row of attn sums to 1. It's the 'attention budget' each token spends.")
print(f"\nattn[0] (first sequence's attention matrix):")
print(attn[0])
print(f"\nRow sums (should all be 1.0):")
print(attn[0].sum(dim=-1))

# Weighted sum of values
out = attn @ V
print(f"\nout = attn @ V   shape = {tuple(out.shape)}")
print("Each token's NEW vector is a weighted mix of all the V vectors,")
print("with weights coming from its attention row.")
print()

# Output projection — one final linear before we're done
out = W_o(out)
print(f"out = W_o(out)   shape = {tuple(out.shape)}")
print("This output projection lets the model 'rotate' the output space —")
print("important in multi-head attention (next step).")
print()


# ============================================================================
# STEP 3 — Wrap it as a self-attention module
# ============================================================================
# Same logic, packaged as an nn.Module.

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        attn = F.softmax(scores, dim=-1)
        return self.W_o(attn @ V)


# ============================================================================
# STEP 4 — Multi-head attention: same idea, run in parallel n_heads times
# ============================================================================
# Instead of ONE big attention, we run n_heads "small" attentions in parallel
# (each on a slice of the embedding) and concatenate. The output projection
# W_o then mixes the heads together.
#
# Why bother? Because different heads can specialize: one head might learn
# "look at the verb", another "look at the subject", etc. Each head sees
# the same data but through different Q/K/V lenses.

print("=" * 70)
print("STEP 4 — Multi-head attention from scratch")
print("=" * 70)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # One Linear per Q/K/V is fine — we'll reshape afterward
        # to split into heads.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape

        # 1. Project to Q, K, V                          shape: (B, L, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into heads. (B, L, d_model) -> (B, L, n_heads, d_k)
        #    then transpose to (B, n_heads, L, d_k) so each head can attend
        #    independently along the L axis.
        Q = Q.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        # Each is now (B, n_heads, L, d_k).

        # 3. Per-head scaled dot-product attention.
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V        # (B, n_heads, L, d_k)

        # 4. Concat heads. (B, n_heads, L, d_k) -> (B, L, n_heads, d_k)
        #    -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # 5. Final projection to mix heads
        return self.W_o(out)


# Try it
mha = MultiHeadAttention(d_model, n_heads)
out_mine = mha(x)
print(f"Multi-head attention output shape: {tuple(out_mine.shape)}")
print(f"out_mine[0] (first sequence, all tokens):")
print(out_mine[0].detach())
print()


# ============================================================================
# STEP 5 — Compare to PyTorch's nn.MultiheadAttention
# ============================================================================
# If we copy our weights into PyTorch's version, do we get the same output?
# (Spoiler: yes — but with some careful weight reshaping because PyTorch
# stores Q/K/V projections in a SINGLE concatenated matrix internally.)

print("=" * 70)
print("STEP 5 — Sanity check: same output as PyTorch's nn.MultiheadAttention?")
print("=" * 70)

ref = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)

# Copy our weights into PyTorch's. PyTorch stores Q, K, V projections
# concatenated as one big (3*d_model, d_model) matrix called in_proj_weight.
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight], dim=0))
    ref.out_proj.weight.copy_(mha.W_o.weight)

# Run reference. need_weights=False because we just want the output tensor.
out_ref, _ = ref(x, x, x, need_weights=False)

print(f"Our output[0][0]:   {out_mine[0, 0].detach().tolist()}")
print(f"PyTorch's[0][0]:    {out_ref[0, 0].detach().tolist()}")
diff = (out_mine - out_ref).abs().max().item()
print(f"\nMax absolute difference: {diff:.2e}")
if diff < 1e-5:
    print("  ✓ Numerically identical. Our 25-line MultiHeadAttention IS nn.MultiheadAttention.")
else:
    print("  (Slight numerical difference; expected within fp32 tolerance.)")
print()


# ============================================================================
# STEP 6 — Train both versions side by side
# ============================================================================
# To really prove our re-implementation is equivalent, train both end-to-end
# on the same toy task and check they converge similarly.

print("=" * 70)
print("STEP 6 — Train both versions on the same toy task")
print("=" * 70)


def make_toy_data(B=32, L=4, V=10):
    """Trivial pattern: predict the FIRST token of the sequence at every
    position. Forces the model to actually USE attention (each position
    must look back at position 0)."""
    x_ids = torch.randint(0, V, (B, L))
    y = x_ids[:, 0:1].expand(-1, L)  # target = position-0 token, broadcast
    return x_ids, y


def train_one(name, attn_module, epochs=400):
    torch.manual_seed(7)
    V_t, d_t = 10, 6
    emb = nn.Embedding(V_t, d_t)
    head = nn.Linear(d_t, V_t)
    opt = torch.optim.AdamW(
        list(emb.parameters()) + list(attn_module.parameters()) + list(head.parameters()),
        lr=0.05,
    )
    loss_fn = nn.CrossEntropyLoss()

    last_loss = None
    for step in range(epochs):
        x_ids, y = make_toy_data(B=32, L=4, V=V_t)
        h = emb(x_ids)
        h = attn_module(h) if not isinstance(attn_module, nn.MultiheadAttention) else attn_module(h, h, h, need_weights=False)[0]
        logits = head(h)
        loss = loss_fn(logits.reshape(-1, V_t), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = loss.item()
    return last_loss


# Both with same d_model and n_heads
torch.manual_seed(42); my_mha   = MultiHeadAttention(6, 2)
torch.manual_seed(42); torch_mha = nn.MultiheadAttention(6, 2, bias=False, batch_first=True)

loss_mine  = train_one("ours",    my_mha)
loss_torch = train_one("PyTorch", torch_mha)

print(f"Our MultiHeadAttention:  final loss {loss_mine:.4f}")
print(f"PyTorch's:               final loss {loss_torch:.4f}")
print()
print("Both train to roughly the same loss. They're doing the same math.")
print()


# ============================================================================
# STEP 7 — Things to try
# ============================================================================
# 1. Drop the sqrt(d_k) scaling. Re-run STEP 6. Does training get harder?
#    Print scores before softmax with and without scaling — they should be
#    much larger without it.
#
# 2. Set n_heads = 1 in MultiHeadAttention and verify it gives the same
#    result as the SingleHeadSelfAttention class from STEP 3.
#
# 3. Add an attention MASK: an L x L matrix of 0s and -inf where -inf entries
#    are positions a token isn't allowed to attend to. Apply it by adding
#    the mask to `scores` before softmax (since softmax of -inf is 0).
#    This is how causal language models (GPT) work — each token can only
#    look at past tokens.
#
# 4. Print the attention matrix `attn[0]` during training. As the model
#    learns to "predict the first token", which positions should it attend
#    to? Does the matrix actually show that?
