"""
Lesson 4d — The full Transformer block.

Phase 2, step 4 of 4 — the finale.

L4c showed that attention ALONE doesn't beat mean-pool. The architecture
needs three more pieces:

  1. RESIDUAL CONNECTION around attention   → gradient highway
  2. FEED-FORWARD sub-layer                  → mix info within each token
  3. LAYER NORM                              → stable training

This file adds all three. The result is one full Transformer encoder block —
the same thing we built in L4e ("from scratch"). Should clearly beat both
L4b (mean-pool) and L4c (naked attention).

After this lesson, we've rebuilt pragma_mini.py from the ground up.

Run:  python3 04d_full_block.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)


# ============================================================================
# STEP 1 — Same data as L4a, L4b, L4c
# ============================================================================

KEYS   = ["pet", "action", "place"]
VALUES = ["dog", "cat", "fish", "eat", "sleep", "play", "garden", "couch", "bowl"]
PAD, MASK = "<pad>", "<mask>"
vocab  = [PAD, MASK] + KEYS + VALUES
tok2id = {t: i for i, t in enumerate(vocab)}
V = len(vocab)

RULES = {
    "dog":  {"action": ["eat", "play"],   "place": ["garden", "bowl"]},
    "cat":  {"action": ["sleep", "play"], "place": ["couch", "bowl"]},
    "fish": {"action": ["eat", "sleep"],  "place": ["bowl"]},
}

def random_event():
    pet = random.choice(list(RULES))
    act = random.choice(RULES[pet]["action"])
    plc = random.choice(RULES[pet]["place"])
    return {"pet": pet, "action": act, "place": plc}

def encode_event(ev):
    ids = []
    for k in KEYS:
        ids.append(tok2id[k])
        ids.append(tok2id[ev[k]])
    return ids

def make_mlm_example(ev, mask_field):
    ids = encode_event(ev)
    pos = 2 * KEYS.index(mask_field) + 1
    target_value_id = ids[pos]
    ids[pos] = tok2id[MASK]
    return ids, pos, target_value_id

def build_dataset(events):
    Xs, positions, ys = [], [], []
    for ev in events:
        for k in KEYS:
            ids, pos, tgt = make_mlm_example(ev, k)
            Xs.append(ids); positions.append(pos); ys.append(tgt)
    return (torch.tensor(Xs, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))

train = [random_event() for _ in range(5000)]
test  = [random_event() for _ in range(1000)]
X_tr, pos_tr, y_tr = build_dataset(train)
X_te, pos_te, y_te = build_dataset(test)


# ============================================================================
# STEP 2 — The model: Full Transformer block
# ============================================================================
# Architecture:
#
#   ids → Embedding → [+ pos enc] → ENCODER BLOCK → take MASK pos → Linear → logits
#
# Where ENCODER BLOCK is:
#
#   x ──► Attention ──► add x ──► LayerNorm ──► FeedForward ──► add ──► LayerNorm ──► out
#                       (resid)    (norm1)                    (resid)    (norm2)
#
# We use multi-head attention now (2 heads).

D = 16


class MultiHeadAttention(nn.Module):
    """Multi-head from L3c, no surprises."""
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.n_heads, self.d_k = d, n_heads, d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, self.d)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """ONE full encoder block: attention + FFN + residuals + LayerNorms."""
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn  = MultiHeadAttention(d, n_heads)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        # Residual + norm around attention
        x = self.norm1(x + self.attn(x))
        # Residual + norm around FFN
        x = self.norm2(x + self.ff(x))
        return x


class FullModel(nn.Module):
    """Embedding + position + Transformer block + linear head."""
    def __init__(self, V, d, n_heads, max_len=16):
        super().__init__()
        self.emb   = nn.Embedding(V, d)
        self.pos   = nn.Embedding(max_len, d)
        self.block = TransformerBlock(d, n_heads)
        self.head  = nn.Linear(d, V)

    def forward(self, ids, positions=None):
        B, L = ids.shape
        pos_ids = torch.arange(L, device=ids.device).expand(B, L)
        h = self.emb(ids) + self.pos(pos_ids)
        h = self.block(h)
        if positions is not None:
            mask_h = h[torch.arange(B), positions]
        else:
            mask_h = h.mean(dim=1)
        return self.head(mask_h)


model = FullModel(V, D, n_heads=2)
total = sum(p.numel() for p in model.parameters())
print("=" * 70)
print("STEP 2 — Full Transformer block")
print("=" * 70)
print(f"Total parameters: {total}")
print("Pieces:")
print(f"  - nn.Embedding(V, d)       (token embedding)")
print(f"  - nn.Embedding(max_len, d) (position embedding — NEW vs L4c)")
print(f"  - MultiHeadAttention       (now 2 heads — was 1 in L4c)")
print(f"  - residual + LayerNorm     (NEW vs L4c)")
print(f"  - feed-forward + GELU      (NEW vs L4c)")
print(f"  - another residual + LayerNorm")
print(f"  - nn.Linear(d, V)          (output head)")
print()


# ============================================================================
# STEP 3 — Train
# ============================================================================

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

print("=" * 70)
print("STEP 3 — Training")
print("=" * 70)
BATCH = 128
N_STEPS = 2000
for step in range(N_STEPS):
    idx = torch.randint(0, X_tr.size(0), (BATCH,))
    xb = X_tr[idx]; yb = y_tr[idx]; pb = pos_tr[idx]
    logits = model(xb, pb)
    loss = loss_fn(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0 or step == N_STEPS - 1:
        print(f"  step {step:4d}   loss {loss.item():.3f}")
print()


# ============================================================================
# STEP 4 — Evaluate
# ============================================================================

model.eval()
with torch.no_grad():
    logits = model(X_te, pos_te)
    pred = logits.argmax(-1)
    acc = (pred == y_te).float().mean().item()
    ce  = loss_fn(logits, y_te).item()

print("=" * 70)
print("STEP 4 — The Phase 2 leaderboard")
print("=" * 70)
print(f"  L4a (counts):                         63.4% acc,  0.670 CE   (zero params)")
print(f"  L4b (emb + mean-pool + linear):       64.2% acc,  0.558 CE   (~400 params)")
print(f"  L4c (emb + naked attention):          53.6% acc,  0.902 CE   (~1.5k params)")
print(f"  L4d (FULL TRANSFORMER BLOCK):         {acc * 100:.1f}% acc,  {ce:.3f} CE   ({total} params)")
print()
print("L4d should beat ALL the previous lessons. If it doesn't, train longer")
print("or check the seed.")
print()


# ============================================================================
# STEP 5 — What did the residuals + FFN + LayerNorm actually fix?
# ============================================================================
# The residual connection lets gradients flow backward through the layer
# without having to pass through the (initially random, hard-to-train)
# attention mechanism. Without it, gradients have a hard time reaching
# the embedding weights — they have to push through the attention
# bottleneck.
#
# The FFN gives the model capacity to combine the attended-to information
# into a final prediction. Attention's job is to SELECT relevant tokens;
# FFN's job is to TRANSFORM that selected info into something useful.
# They're complementary.
#
# LayerNorm keeps activations from drifting as the model scales. With
# more layers, this becomes critical.


# ============================================================================
# STEP 6 — Compare to pragma_mini.py
# ============================================================================
# What we just built is functionally identical to pragma_mini.py's PragmaMini
# class. The PyTorch version uses nn.TransformerEncoderLayer; ours uses our
# own from-scratch block. The numerical results should be very similar.

print("=" * 70)
print("STEP 6 — We just rebuilt pragma_mini.py from scratch")
print("=" * 70)
print()
print("pragma_mini.py's PragmaMini class is essentially:")
print()
print("  PragmaMini = Embedding + Position + nn.TransformerEncoderLayer x N + Linear")
print()
print("What we just built:")
print()
print("  FullModel = Embedding + Position + OurTransformerBlock(x N=1) + Linear")
print()
print("where OurTransformerBlock is the explicit attention + FFN + residual + LN.")
print("Same architecture. Same recipe. We just opened the box.")
print()


# ============================================================================
# STEP 7 — Things to try
# ============================================================================
# 1. Drop just ONE of {residual1, residual2, FFN, LayerNorm} and re-train.
#    Which piece, when removed, hurts the most?
#
# 2. Use 4 heads instead of 2. Does it help on this small task?
#
# 3. Stack TWO TransformerBlocks. Compare to one. How does it change the
#    parameter count, training time, and final accuracy?
#
# 4. Replace your TransformerBlock with PyTorch's nn.TransformerEncoderLayer.
#    Train. Do you get the same accuracy? (Should be within noise — they're
#    the same architecture.)
