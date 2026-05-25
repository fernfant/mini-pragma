"""
pragma_mini.py — a tiny "hello world" version of PRAGMA.

PRAGMA (Revolut) is a Transformer that reads a person's banking events
(purchases, transfers, app taps) and learns to fill in missing pieces,
just like BERT does for words in a sentence. Once trained, the same
model can be reused for credit scoring, fraud detection, etc.

Here we shrink the idea down to a kid-friendly toy:
    each "user" is a pet, and the "events" are things the pet does.
    We hide some tokens and train the model to guess them back.

Run:  python3 pragma_mini.py
Needs: pip install torch
"""

import random
import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)

# ---------------------------------------------------------------------------
# 1. Vocabulary.
# Real PRAGMA has ~60 "keys" (Type, Amount, ...) and ~28k "values".
# Ours has just a handful.
# ---------------------------------------------------------------------------
KEYS   = ["pet", "action", "place"]
VALUES = ["dog", "cat", "fish",          # pets
          "eat", "sleep", "play",        # actions
          "garden", "couch", "bowl"]     # places

PAD, MASK = "<pad>", "<mask>"
vocab = [PAD, MASK] + KEYS + VALUES
tok2id = {t: i for i, t in enumerate(vocab)}
V = len(vocab)

# ---------------------------------------------------------------------------
# 2. Synthetic "events". Each pet does sensible things in sensible places.
# This is the pattern we want the model to discover on its own.
# ---------------------------------------------------------------------------
RULES = {
    "dog":  {"action": ["eat", "play"],   "place": ["garden", "bowl"]},
    "cat":  {"action": ["sleep", "play"], "place": ["couch", "bowl"]},
    "fish": {"action": ["eat", "sleep"],  "place": ["bowl"]},
}

def random_event():
    pet = random.choice(list(RULES))
    act = random.choice(RULES[pet]["action"])
    plc = random.choice(RULES[pet]["place"])
    return [("pet", pet), ("action", act), ("place", plc)]

# ---------------------------------------------------------------------------
# 3. Tokenise. PRAGMA decomposes each record into (key, value) pairs —
# we do the same. An event becomes [pet, dog, action, eat, place, garden].
# ---------------------------------------------------------------------------
def encode(event):
    ids = []
    for k, v in event:
        ids.append(tok2id[k])
        ids.append(tok2id[v])
    return ids

# ---------------------------------------------------------------------------
# 4. Masking — the BERT-style training game from §2.3.5 of the paper.
# Hide some value tokens, remember the answer, ask the model to guess.
# ---------------------------------------------------------------------------
def mask(ids, p=0.25):
    ids = list(ids)
    labels = [-100] * len(ids)              # -100 = "don't score this position"
    for i, t in enumerate(ids):
        if vocab[t] in KEYS:                # keep field names visible
            continue
        if random.random() < p:
            labels[i] = t                   # remember the truth
            ids[i] = tok2id[MASK]           # hide the value
    return ids, labels

# ---------------------------------------------------------------------------
# 5. Tiny Transformer. This is the "backbone" — like PRAGMA's encoders,
# but with 32-d embeddings and 2 layers instead of 1024-d and 18 layers.
# ---------------------------------------------------------------------------
class PragmaMini(nn.Module):
    def __init__(self, V, d=32, heads=2, layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(64, d)
        layer = nn.TransformerEncoderLayer(d, heads, 64, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, V)         # the MLM head: predict any token
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.enc(self.emb(x) + self.pos(positions))
        return self.head(h)

# ---------------------------------------------------------------------------
# 6. Train.
# ---------------------------------------------------------------------------
model   = PragmaMini(V)
opt     = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

print("Training...")
for step in range(2000):
    batch = [random_event() for _ in range(32)]
    masked, labels = zip(*[mask(encode(e)) for e in batch])
    x = torch.tensor(masked)
    y = torch.tensor(labels)
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"  step {step:4d}   loss {loss.item():.3f}")

# ---------------------------------------------------------------------------
# 7. Play with it. Hide one field, ask the model to fill the blank.
# ---------------------------------------------------------------------------
def guess(event, hide_key):
    ids = encode(event)
    pos = None
    for i in range(0, len(ids), 2):
        if vocab[ids[i]] == hide_key:
            pos = i + 1
            ids[pos] = tok2id[MASK]
    with torch.no_grad():
        logits = model(torch.tensor([ids]))
    return vocab[logits[0, pos].argmax().item()]

print("\nFill in the blank:")
print("  dog is playing in the ____ ->",
      guess([("pet","dog"), ("action","play"), ("place","garden")], "place"))
print("  cat is sleeping on the ____ ->",
      guess([("pet","cat"), ("action","sleep"), ("place","couch")], "place"))
print("  fish is eating in the ____ ->",
      guess([("pet","fish"), ("action","eat"), ("place","bowl")], "place"))
print("  ____ is sleeping on the couch ->",
      guess([("pet","dog"), ("action","sleep"), ("place","couch")], "pet"))
print("  dog is ____ in the garden ->",
      guess([("pet","dog"), ("action","eat"), ("place","garden")], "action"))
