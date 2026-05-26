"""
Lesson 4b — Embedding + linear: the first neural baseline.

Phase 2, step 2 of 4. We replace L4a's counts table with LEARNED embeddings
and a linear head. Same data, same task. Should beat counts (~63.4%, ~0.67 CE).

The architecture (no attention yet, no Transformer block):

    token_ids ──► nn.Embedding ──► mean-pool ──► nn.Linear ──► logits

That's it. We embed each visible token (key+value), average them into one
vector representing the whole event-so-far, then linearly project to a
distribution over the vocabulary.

This is the dumbest possible neural net for this task. But because the
embeddings ARE LEARNED, the model can capture combinations of features
in a way the counts model couldn't.

Run:  python3 04b_embedding_linear.py
"""

import math
import random
import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)


# ============================================================================
# STEP 1 — Reuse the same dataset as L4a
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
    """Flatten to [pet_key, pet_val, action_key, action_val, place_key, place_val]"""
    ids = []
    for k in KEYS:
        ids.append(tok2id[k])
        ids.append(tok2id[ev[k]])
    return ids

train = [random_event() for _ in range(5000)]
test  = [random_event() for _ in range(1000)]

print("=" * 70)
print("STEP 1 — Same data as L4a. Counts baseline: 63.4% acc, 0.670 CE.")
print("=" * 70)
print()


# ============================================================================
# STEP 2 — Make MLM training examples
# ============================================================================
# For each event we generate 3 training examples by masking each value in turn.

def make_mlm_example(ev, mask_field):
    """Mask the value of `mask_field`. Return (input_ids, target_idx, target_value_id)."""
    ids = encode_event(ev)
    # value of mask_field is at position 2*KEYS.index(mask_field) + 1
    pos = 2 * KEYS.index(mask_field) + 1
    target_value_id = ids[pos]
    ids[pos] = tok2id[MASK]
    return ids, pos, target_value_id


# Build tensors
def build_dataset(events):
    Xs, positions, ys = [], [], []
    for ev in events:
        for k in KEYS:
            ids, pos, tgt = make_mlm_example(ev, k)
            Xs.append(ids); positions.append(pos); ys.append(tgt)
    return (torch.tensor(Xs, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))

X_tr, pos_tr, y_tr = build_dataset(train)
X_te, pos_te, y_te = build_dataset(test)
print(f"Train tensor: X={tuple(X_tr.shape)}, y={tuple(y_tr.shape)}")
print(f"Test  tensor: X={tuple(X_te.shape)}, y={tuple(y_te.shape)}")
print()


# ============================================================================
# STEP 3 — The model: Embedding + mean-pool + Linear
# ============================================================================

D = 16   # embedding dim

class EmbLinearModel(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.emb  = nn.Embedding(V, d)
        self.head = nn.Linear(d, V)

    def forward(self, ids):
        # ids:  (B, L)
        h = self.emb(ids)            # (B, L, d)
        pooled = h.mean(dim=1)       # (B, d)  — average across positions
        return self.head(pooled)     # (B, V)  — logits over vocab


model = EmbLinearModel(V, D)
total = sum(p.numel() for p in model.parameters())
print("=" * 70)
print(f"STEP 3 — Model: Embedding({V}, {D}) + Linear({D}, {V})")
print("=" * 70)
print(f"Total trainable parameters: {total}")
print("(vs 0 for the counts model)")
print()


# ============================================================================
# STEP 4 — Train
# ============================================================================

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

print("=" * 70)
print("STEP 4 — Training")
print("=" * 70)
BATCH = 128
N_STEPS = 1000
for step in range(N_STEPS):
    idx = torch.randint(0, X_tr.size(0), (BATCH,))
    xb = X_tr[idx]; yb = y_tr[idx]
    logits = model(xb)
    loss = loss_fn(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0 or step == N_STEPS - 1:
        print(f"  step {step:4d}   loss {loss.item():.3f}")
print()


# ============================================================================
# STEP 5 — Evaluate on test set
# ============================================================================

model.eval()
with torch.no_grad():
    logits = model(X_te)
    pred = logits.argmax(-1)
    acc  = (pred == y_te).float().mean().item()
    ce   = loss_fn(logits, y_te).item()

print("=" * 70)
print("STEP 5 — Results")
print("=" * 70)
print(f"  Test accuracy:       {acc * 100:.1f}%")
print(f"  Test cross-entropy:  {ce:.3f}")
print()
print(f"  L4a (counts):        63.4% acc,  0.670 CE")
print(f"  L4b (this lesson):   {acc * 100:.1f}% acc,  {ce:.3f} CE")
print(f"  Improvement:         +{(acc*100) - 63.4:.1f}% acc,  −{0.670 - ce:.3f} CE")
print()


# ============================================================================
# STEP 6 — Why this model does better than counts
# ============================================================================
# The key advantage: the EMBEDDINGS are LEARNED. The model can place
# 'dog', 'eat', 'garden' at positions in the embedding space that make
# the linear head's job easy.
#
# Mean-pooling discards order, but for this 3-field task that's fine —
# every event has the same structure. The linear head learns to "look at"
# the combined evidence (mean of embeddings) and pick the most likely value.

print("=" * 70)
print("STEP 6 — Inspect the learned embeddings")
print("=" * 70)

emb_weight = model.emb.weight.detach()

def cos_sim(a, b):
    return (a @ b / (a.norm() * b.norm())).item()

print("Cosine similarity between learned embeddings:")
print(f"  dog · cat    = {cos_sim(emb_weight[tok2id['dog']],    emb_weight[tok2id['cat']]):.2f}")
print(f"  dog · fish   = {cos_sim(emb_weight[tok2id['dog']],    emb_weight[tok2id['fish']]):.2f}")
print(f"  eat · sleep  = {cos_sim(emb_weight[tok2id['eat']],    emb_weight[tok2id['sleep']]):.2f}")
print(f"  garden·couch = {cos_sim(emb_weight[tok2id['garden']], emb_weight[tok2id['couch']]):.2f}")
print(f"  bowl·garden  = {cos_sim(emb_weight[tok2id['bowl']],   emb_weight[tok2id['garden']]):.2f}")
print()
print("Even with mean-pooling + linear, the embeddings learn meaningful")
print("structure (pets cluster together, sounds cluster together, etc.)")
print()


# ============================================================================
# STEP 7 — What this model STILL can't do well
# ============================================================================
# Mean-pooling is a crude way to combine token features. It treats every
# token equally regardless of position or relevance. If we want one token
# to look more carefully at another (e.g., the [MASK] position needs to
# look hardest at the 'pet' value), we need ATTENTION.
#
# That's L4c. We'll replace mean-pool with a hand-rolled single-head
# attention layer.


# ============================================================================
# STEP 8 — Things to try
# ============================================================================
# 1. Reduce D from 16 to 2. Re-run. Does it still work? Why does a tiny
#    embedding dimension still suffice for this simple task?
#
# 2. Add a `padding_idx=0` argument to nn.Embedding. (Pad tokens shouldn't
#    contribute to mean-pool gradients.) Does it change anything?
#
# 3. Replace mean-pool with sum-pool. Does it learn the same patterns?
#    What's the relationship between the two?
#
# 4. Look at predictions for an "illegal" combination like (pet=dog,
#    action=sleep) → place=? The counts model handled this poorly. Does
#    embedding+linear do better?
