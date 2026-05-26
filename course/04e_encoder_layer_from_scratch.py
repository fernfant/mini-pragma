"""
Lesson 4e — nn.TransformerEncoderLayer, decomposed and rebuilt from scratch.

Every Transformer architecture (BERT, GPT, PRAGMA) is just a STACK of these
encoder layers. This file opens that black box.

ONE encoder layer is built from FIVE small pieces:

      x ──► LayerNorm ──► MultiHeadAttention ──► add x (residual) ──► h
                                                                       │
            LayerNorm ──► FeedForward ──► add h (residual) ──► output ◄┘

We implement each piece from scratch. We then build the full layer and
verify it gives the same result as nn.TransformerEncoderLayer.

By the end you will know:
  - What "pre-norm" means and why modern Transformers use it.
  - Why residual connections (the `+ x`) are essential for deep nets.
  - What the feed-forward sub-layer does that attention doesn't.
  - That nn.TransformerEncoderLayer is ~60 lines of code you could write.

Run:  python3 04e_encoder_layer_from_scratch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


# ============================================================================
# STEP 1 — LayerNorm from scratch
# ============================================================================
# What it does: for each VECTOR (per token, per sample), normalise to have
# mean 0 and standard deviation 1. Then apply a learned scale and shift.
#
# Why: keeps activations from blowing up or shrinking to zero as you stack
# many layers. Without it, deep networks are very hard to train.

print("=" * 70)
print("STEP 1 — LayerNorm from scratch")
print("=" * 70)


class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))   # learned scale
        self.beta  = nn.Parameter(torch.zeros(dim))  # learned shift
        self.eps = eps

    def forward(self, x):
        # x shape: (..., dim) — normalise along last axis
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


# Sanity check: compare to PyTorch
x = torch.randn(2, 4, 6)
mine = MyLayerNorm(6)
ref  = nn.LayerNorm(6)
mine.gamma.data.copy_(ref.weight.data)
mine.beta.data.copy_(ref.bias.data)
diff = (mine(x) - ref(x)).abs().max().item()
print(f"Max diff vs nn.LayerNorm: {diff:.2e}")
print("After normalisation, each token vector has mean 0 and var 1:")
print(f"  mean across last dim: {mine(x)[0, 0].mean().item():.4f}")
print(f"  var across last dim:  {mine(x)[0, 0].var(unbiased=False).item():.4f}")
print()


# ============================================================================
# STEP 2 — Feed-forward sub-layer from scratch
# ============================================================================
# What it does: applies a 2-layer MLP to EACH token independently.
#     x  ─► Linear(d, 4d) ─► GELU ─► Linear(4d, d)  ─►  output
#
# Why: attention mixes information ACROSS tokens. The feed-forward mixes
# information WITHIN each token's vector. Together they give the model
# enough capacity to learn rich patterns.
#
# The "4× expansion" (d → 4d → d) is convention from the original
# Transformer paper. It gives the model a high-dim intermediate space.

print("=" * 70)
print("STEP 2 — FeedForward sub-layer")
print("=" * 70)


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_mult=4):
        super().__init__()
        d_hidden = d_model * hidden_mult
        self.up   = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


ff = FeedForward(6)
out = ff(x)
print(f"In:  {tuple(x.shape)}    →    Out: {tuple(out.shape)}")
print(f"Parameters: {sum(p.numel() for p in ff.parameters())}")
print()


# ============================================================================
# STEP 3 — MultiHeadAttention (recap from L3c)
# ============================================================================
# Already built in L3c. Including a minimal version here so this file is
# self-contained.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================================
# STEP 4 — Put it all together: one encoder layer
# ============================================================================
# This is what nn.TransformerEncoderLayer wraps. Let's write it ourselves.
#
# Architecture (post-norm — what PyTorch uses by default):
#
#   x ── MultiHead ── add x ── LayerNorm ── FeedForward ── add ── LayerNorm ── out
#         (residual)              (norm1)                (residual)  (norm2)
#
# The residual "+ x" connections are CRITICAL. Without them, deep nets
# vanish their gradients and can't be trained. Residuals are a "highway"
# — they let gradient flow directly through, while the rest of the layer
# learns the "delta".

print("=" * 70)
print("STEP 4 — A single encoder layer, from scratch")
print("=" * 70)


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = FeedForward(d_model, hidden_mult=(dim_feedforward // d_model) if dim_feedforward else 4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Sub-layer 1: self-attention + residual + norm
        x = self.norm1(x + self.attn(x))
        # Sub-layer 2: feed-forward + residual + norm
        x = self.norm2(x + self.ff(x))
        return x


layer = MyTransformerEncoderLayer(d_model=6, n_heads=2, dim_feedforward=24)
out = layer(x)
print(f"Input shape:  {tuple(x.shape)}")
print(f"Output shape: {tuple(out.shape)}")
print(f"Parameters in one layer: {sum(p.numel() for p in layer.parameters())}")
print()
print("That's it. ~50 lines of model code = one Transformer block.")
print()


# ============================================================================
# STEP 5 — Verify against nn.TransformerEncoderLayer
# ============================================================================
# Build PyTorch's reference layer with the same architecture, copy weights
# across, and check the forward pass produces matching output.

print("=" * 70)
print("STEP 5 — Numerical equivalence with nn.TransformerEncoderLayer")
print("=" * 70)

ref = nn.TransformerEncoderLayer(
    d_model=6,
    nhead=2,
    dim_feedforward=24,
    activation="gelu",
    batch_first=True,
    norm_first=False,    # post-norm, matches our implementation
    dropout=0.0,         # disable dropout for exact comparison
)
ref.eval()  # eval mode just to be extra sure dropout (if any) is off

# Copy weights. PyTorch stores Q/K/V projections in a single concatenated
# matrix called in_proj_weight (with optional in_proj_bias).
with torch.no_grad():
    # Attention weights
    ref.self_attn.in_proj_weight.copy_(torch.cat(
        [layer.attn.W_q.weight, layer.attn.W_k.weight, layer.attn.W_v.weight], dim=0))
    ref.self_attn.in_proj_bias.zero_()
    ref.self_attn.out_proj.weight.copy_(layer.attn.W_o.weight)
    ref.self_attn.out_proj.bias.zero_()
    # Feed-forward weights
    ref.linear1.weight.copy_(layer.ff.up.weight)
    ref.linear1.bias.copy_(layer.ff.up.bias)
    ref.linear2.weight.copy_(layer.ff.down.weight)
    ref.linear2.bias.copy_(layer.ff.down.bias)
    # Layer norms
    ref.norm1.weight.copy_(layer.norm1.weight)
    ref.norm1.bias.copy_(layer.norm1.bias)
    ref.norm2.weight.copy_(layer.norm2.weight)
    ref.norm2.bias.copy_(layer.norm2.bias)

out_mine = layer(x)
out_ref  = ref(x)
diff = (out_mine - out_ref).abs().max().item()
print(f"Max absolute difference: {diff:.2e}")
if diff < 1e-5:
    print("  ✓ Our hand-built encoder layer is numerically identical to PyTorch's.")
    print("  ✓ You now know exactly what nn.TransformerEncoderLayer does.")
print()


# ============================================================================
# STEP 6 — How to USE this: stack multiple layers
# ============================================================================
# PRAGMA stacks 18 of these. Stacking is trivial — just chain them.

print("=" * 70)
print("STEP 6 — Stacking layers into a full encoder")
print("=" * 70)


class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([
            MyTransformerEncoderLayer(d_model, n_heads, dim_feedforward)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


enc = MyTransformerEncoder(d_model=32, n_heads=4, n_layers=3, dim_feedforward=128)
x_test = torch.randn(2, 10, 32)
print(f"3-layer encoder, 32 d_model, 4 heads")
print(f"Input:  {tuple(x_test.shape)}")
print(f"Output: {tuple(enc(x_test).shape)}")
print(f"Total knobs: {sum(p.numel() for p in enc.parameters()):,}")
print()
print("Same architecture used in:")
print("  - pragma_mini.py  (2 layers,  32-d,  2 heads)")
print("  - PRAGMA-Small    (5 layers,  192-d, 3 heads)")
print("  - PRAGMA-Large    (18 layers, 1024-d, 16 heads)")
print()


# ============================================================================
# STEP 7 — Things to try
# ============================================================================
# 1. Replace the post-norm version with PRE-NORM: instead of
#       x = self.norm1(x + self.attn(x))
#    write
#       x = x + self.attn(self.norm1(x))
#    Pre-norm tends to train more stably in deep networks. Try training
#    a deep stack (10+ layers) with each version and observe.
#
# 2. Remove the residual connections — drop the `x +`. Train a 5-layer
#    encoder. You'll find it doesn't train well (or at all). This is the
#    "vanishing gradient" problem that residuals solve.
#
# 3. Remove the feed-forward sub-layer (just attention + norm + residual).
#    The model can still learn but with less capacity. Measure the loss
#    difference on a real task.
#
# 4. Replace GELU with ReLU. Modern Transformers use GELU; the difference
#    is small but consistent. Try both on the same task and compare.
#
# 5. Implement DROPOUT in attention and FFN (the standard Transformer has
#    dropout layers after attention and after FFN). Skip the dropout at
#    eval time. See if it helps generalisation on the real PRAGMA tasks.
