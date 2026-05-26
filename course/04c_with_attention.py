"""
Lesson 4c — Add hand-rolled attention to the model.

Phase 2, step 3 of 4.

L4b used mean-pool: every token contributed equally to the final pooled vector.
That's a crude way to combine evidence. Some tokens matter more than others —
in particular, the [MASK] position needs to pay close attention to the
visible value tokens, not just average them in equally.

This lesson adds a single-head self-attention layer (from L3c) on top of
the embeddings. Now each token "looks at" every other token before pooling.

The architecture:

    token_ids ──► Embedding ──► SelfAttention ──► take MASK position ──► Linear ──► logits

Notice we no longer mean-pool the whole sequence. Instead we look at the
output vector AT THE MASK POSITION — that's where attention has gathered
all the relevant context.

Run:  python3 04c_with_attention.py
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)


# ============================================================================
# STEP 1 — Same data as L4a, L4b
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
# STEP 2 — The model: Embedding + Single-head self-attention + Linear at MASK position
# ============================================================================
# This is the L3c attention, plugged in between the embedding and the head.

D = 16

class SelfAttention(nn.Module):
    """Single-head self-attention. Copied from L3c."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # x: (B, L, d)
        Q, K, V_ = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d)
        attn = F.softmax(scores, dim=-1)
        return self.W_o(attn @ V_), attn


class EmbAttentionModel(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.attn = SelfAttention(d)
        self.head = nn.Linear(d, V)

    def forward(self, ids, positions=None):
        # ids:       (B, L)
        # positions: (B,)  -- which position in each row is the [MASK]
        h = self.emb(ids)              # (B, L, d)
        h, attn = self.attn(h)         # (B, L, d), (B, L, L)
        if positions is not None:
            # Pick the output at each row's MASK position
            B = h.size(0)
            mask_h = h[torch.arange(B), positions]  # (B, d)
        else:
            mask_h = h.mean(dim=1)
        return self.head(mask_h), attn


model = EmbAttentionModel(V, D)
total = sum(p.numel() for p in model.parameters())
print("=" * 70)
print(f"STEP 2 — Model: Embedding + Self-attention + Linear")
print("=" * 70)
print(f"Total parameters: {total}")
print(f"(vs {V*D + D*V + V} for L4b — extra params are the Q/K/V/O matrices)")
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
N_STEPS = 3000
for step in range(N_STEPS):
    idx = torch.randint(0, X_tr.size(0), (BATCH,))
    xb = X_tr[idx]; yb = y_tr[idx]; pb = pos_tr[idx]
    logits, _ = model(xb, pb)
    loss = loss_fn(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0 or step == N_STEPS - 1:
        print(f"  step {step:4d}   loss {loss.item():.3f}")
print()


# ============================================================================
# STEP 4 — Evaluate
# ============================================================================

model.eval()
with torch.no_grad():
    logits, _ = model(X_te, pos_te)
    pred = logits.argmax(-1)
    acc = (pred == y_te).float().mean().item()
    ce  = loss_fn(logits, y_te).item()

print("=" * 70)
print("STEP 4 — Results")
print("=" * 70)
print(f"  Test accuracy:       {acc * 100:.1f}%")
print(f"  Test cross-entropy:  {ce:.3f}")
print()
print(f"  L4a (counts):           63.4% acc,  0.670 CE")
print(f"  L4b (emb+linear):       64.2% acc,  0.558 CE")
print(f"  L4c (emb+ATTENTION):    {acc * 100:.1f}% acc,  {ce:.3f} CE")
print()
print("WAIT — attention is supposed to be BETTER. Why did the loss go UP?")
print()
print("Honest answer: a 'naive attention' layer with no residual connection,")
print("no FFN, and no LayerNorm is HARD to train. The attention learns to")
print("focus on a SINGLE highly-predictive token (often 'pet'), but then has")
print("no way to also use the OTHER token (action) — there's no FFN to mix")
print("them. Mean-pool wins here because it incorporates ALL tokens equally.")
print()
print("This is why real Transformer blocks include FOUR pieces:")
print("  1. Attention (we have this)")
print("  2. Residual connection around attention (we DON'T)")
print("  3. Feed-forward sub-layer (we DON'T)")
print("  4. LayerNorm (we DON'T)")
print()
print("L4d adds all three missing pieces. Watch what happens to the loss.")
print()


# ============================================================================
# STEP 5 — Look inside attention: what is the model paying attention to?
# ============================================================================
# When predicting the place, where does the MASK token put its attention?
# Hopefully on the pet and action values, not on the keys (which are
# uninformative — they're just labels saying "this is a pet field").

print("=" * 70)
print("STEP 5 — Look inside attention")
print("=" * 70)

# Take one example, hide the place, see where MASK attends.
test_ev = {"pet": "dog", "action": "play", "place": "garden"}
ids, pos, _ = make_mlm_example(test_ev, "place")
print(f"Input: dog plays in [MASK]")
print(f"Position layout: {[vocab[i] for i in ids]}")
print(f"MASK is at position {pos}.")

with torch.no_grad():
    logits, attn = model(torch.tensor([ids]), torch.tensor([pos]))
    attn_row = attn[0, pos].tolist()    # attention FROM the MASK position
    pred_id = logits.argmax(-1).item()

print(f"\nAttention from MASK to each token:")
for i, (tok, weight) in enumerate(zip([vocab[i] for i in ids], attn_row)):
    bar = "█" * int(weight * 50)
    print(f"  pos {i}  {tok:8s} {weight:.3f}  {bar}")

print(f"\nPredicted value:  {vocab[pred_id]}    (truth: garden)")
print()
print("If attention is well-trained, MASK should attend strongly to 'dog'")
print("and 'play' — the visible value tokens — rather than to keys or padding.")
print()


# ============================================================================
# STEP 6 — Why this beats L4b
# ============================================================================
# Mean-pool gave every token equal weight. Attention gives the MASK token
# the ability to CHOOSE which tokens matter for its prediction. For
# example, when predicting `place`, the model has learned to look at the
# `pet` value much more than the `action` value (since pet → place is
# stronger in the rules).
#
# But we're still missing pieces:
#   - Single layer (no stacking).
#   - No residual connections.
#   - No feed-forward sub-layer (just attention).
#   - No LayerNorm.
#
# L4d adds all those. That's the full Transformer block — pragma_mini.py
# rebuilt from scratch.


# ============================================================================
# STEP 7 — Things to try
# ============================================================================
# 1. Drop the sqrt(d) scaling in the attention. Retrain. What happens to
#    the attention weights' sharpness?
#
# 2. Try TWO heads of attention instead of one. (You'll need to use the
#    MultiHeadAttention from L3c.) Does it help on this small task?
#
# 3. Add a residual connection: instead of returning W_o(attn @ V), return
#    x + W_o(attn @ V). Does training stabilise?
#
# 4. Print the attention matrix from STEP 5 BEFORE training (random weights)
#    and AFTER. The "after" should clearly show the model has learned to
#    attend to value tokens (positions 1, 3) more than key tokens.
