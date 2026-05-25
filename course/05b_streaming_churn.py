"""
Lesson 5b — A more realistic worked example: streaming churn prediction.

A streaming service (think Netflix) wants to predict which users are about
to cancel. Each user has a history of viewing events. We apply the same
PRAGMA recipe as pragma_mini.py:

  1. Generate synthetic event data — "engaged" users vs "churning" users.
  2. PRE-TRAIN a Transformer on ALL events with fill-in-the-blank (no labels).
  3. FREEZE the encoder, add a tiny classifier head on top.
  4. Train ONLY the head on a small labelled set.
  5. Compare to a baseline that doesn't pre-train.

This is the exact "foundation model" workflow PRAGMA runs at billion-event
scale, just shrunk to a laptop.

Same recipe as pragma_mini.py — but now applied to a more realistic domain
and showing the FULL downstream-task pipeline.

Run:  python3 05b_streaming_churn.py
"""

import random
import torch
import torch.nn as nn

torch.manual_seed(7)
random.seed(7)
torch.set_printoptions(precision=3, sci_mode=False)

# ============================================================================
# PART 1 — Synthetic data
# ============================================================================
# Each event has 4 (key, value) pairs:
#   action   — what the user did
#   genre    — what genre they were watching
#   time     — when in the day
#   duration — how long they watched (bucket)
#
# Two user types:
#   ENGAGED  — finishes most episodes, varied genres, watches in long sessions
#   CHURNING — skips a lot, narrow genre, short sessions, browsing without committing

KEYS     = ["action", "genre", "time", "duration"]
ACTIONS  = ["start", "finish", "skip", "pause", "browse"]
GENRES   = ["comedy", "drama", "action", "documentary", "kids"]
TIMES    = ["morning", "afternoon", "evening", "night"]
DURATION = ["short", "medium", "long"]

PAD, MASK = "<pad>", "<mask>"
vocab  = [PAD, MASK] + KEYS + ACTIONS + GENRES + TIMES + DURATION
tok2id = {t: i for i, t in enumerate(vocab)}
V      = len(vocab)

EVENTS_PER_USER = 15
N_USERS         = 2000
CHURN_RATE      = 0.10


def engaged_event():
    a = random.choices(ACTIONS, weights=[2, 8, 1, 1, 1])[0]   # mostly finishes
    g = random.choice(GENRES)                                  # varied tastes
    t = random.choices(TIMES,   weights=[1, 2, 5, 2])[0]       # mostly evening
    d = random.choices(DURATION, weights=[1, 3, 6])[0]         # long sessions
    return [("action", a), ("genre", g), ("time", t), ("duration", d)]


def churning_event(narrow_genre):
    a = random.choices(ACTIONS, weights=[3, 1, 6, 1, 4])[0]   # skips a lot
    g = random.choices([narrow_genre] * 6 + GENRES, k=1)[0]    # narrow taste
    t = random.choices(TIMES,   weights=[2, 3, 2, 5])[0]       # mostly night
    d = random.choices(DURATION, weights=[7, 2, 1])[0]         # short sessions
    return [("action", a), ("genre", g), ("time", t), ("duration", d)]


def make_user(churning):
    if churning:
        narrow = random.choice(GENRES)
        return [churning_event(narrow) for _ in range(EVENTS_PER_USER)]
    return [engaged_event() for _ in range(EVENTS_PER_USER)]


def encode_event(event):
    """Flatten an event into [key, value, key, value, ...] token IDs."""
    ids = []
    for k, v in event:
        ids.append(tok2id[k])
        ids.append(tok2id[v])
    return ids


def encode_user(events):
    """Flatten a user's whole event history into a single token sequence."""
    return [tok for e in events for tok in encode_event(e)]


# Build the dataset
print("Generating 2000 synthetic users...")
users, labels = [], []
for _ in range(N_USERS):
    churn = random.random() < CHURN_RATE
    users.append(make_user(churn))
    labels.append(1 if churn else 0)

X = torch.tensor([encode_user(u) for u in users], dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)
seq_len = X.size(1)
print(f"  shape X: {tuple(X.shape)}  ({seq_len} tokens per user)")
print(f"  labels:  {int(y.sum())} churning,  {int((y == 0).sum())} engaged\n")


# ============================================================================
# PART 2 — Architecture (same shape as pragma_mini.py)
# ============================================================================

D_MODEL  = 32
N_HEADS  = 2
N_LAYERS = 2


class Encoder(nn.Module):
    """The backbone: embedding + position + Transformer encoder.
    Same architecture as pragma_mini.py. (L2 + L3.)"""
    def __init__(self, V, d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(max_len, d)
        layer    = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.enc(self.emb(x) + self.pos(positions))


class MLMHead(nn.Module):
    """Output head used during pre-training: vector -> vocab scores."""
    def __init__(self, V, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d, V)
    def forward(self, h):
        return self.proj(h)


class ChurnHead(nn.Module):
    """Downstream head: pool the encoder output, predict churn (2 classes)."""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d, 2)
    def forward(self, h):
        pooled = h.mean(dim=1)   # average over all positions
        return self.proj(pooled)


# ============================================================================
# PART 3 — Pre-train via masked language modelling (L4)
# ============================================================================

KEY_IDS = torch.tensor([tok2id[k] for k in KEYS])

def mlm_mask(X_batch, p=0.20):
    """Hide random VALUE tokens (never key tokens) and remember the answers."""
    X = X_batch.clone()
    y = torch.full_like(X, -100)
    is_value = ~torch.isin(X, KEY_IDS)
    pick = (torch.rand_like(X, dtype=torch.float) < p) & is_value
    y[pick] = X[pick]
    X[pick] = tok2id[MASK]
    return X, y

encoder  = Encoder(V)
mlm_head = MLMHead(V)
opt      = torch.optim.AdamW(list(encoder.parameters()) + list(mlm_head.parameters()), lr=3e-3)
loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)

print(f"Pre-training encoder via MLM for 2000 steps...")
print(f"  encoder + MLM head knobs: {sum(p.numel() for p in list(encoder.parameters()) + list(mlm_head.parameters())):,}")
for step in range(2000):
    idx       = torch.randint(0, N_USERS, (64,))      # mini-batch
    xb, yb    = mlm_mask(X[idx])
    h         = encoder(xb)
    logits    = mlm_head(h)
    loss      = loss_fn(logits.reshape(-1, V), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        print(f"  step {step:4d}   MLM loss {loss.item():.3f}")

# Save the pre-trained encoder so we can compare against the baseline
import copy
pretrained_encoder = copy.deepcopy(encoder)
print()


# ============================================================================
# PART 4 — Downstream task: predict CHURN
# ============================================================================
# Split data into train / test
perm   = torch.randperm(N_USERS)
split  = int(N_USERS * 0.8)
tr_idx, te_idx = perm[:split], perm[split:]
X_te, y_te     = X[te_idx], y[te_idx]
print(f"Test set: {len(te_idx)} users  ({int(y_te.sum())} churning)\n")


def freeze(mod):
    for p in mod.parameters(): p.requires_grad = False
    mod.eval()


def train_classifier(encoder, X_tr, y_tr, epochs=200, freeze_encoder=True):
    head = ChurnHead()
    if freeze_encoder:
        freeze(encoder)
        params = list(head.parameters())
    else:
        params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        h      = encoder(X_tr)
        logits = head(h)
        loss   = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        logits = head(encoder(X_te))
        pred   = logits.argmax(-1)
        acc    = (pred == y_te).float().mean().item()
        churn  = (y_te == 1)
        recall = (pred[churn] == 1).float().mean().item() if churn.sum() > 0 else float("nan")
    return acc, recall


print(f"{'labels':>7} | {'pretrained acc':>14}  {'pretrained recall':>17} | "
      f"{'baseline acc':>12}  {'baseline recall':>15}")
print("-" * 84)
for n_labels in [20, 50, 200, 1000]:
    sub = tr_idx[:n_labels]
    X_tr, y_tr = X[sub], y[sub]

    # A. Frozen pre-trained encoder + classifier head only
    enc_a = copy.deepcopy(pretrained_encoder)
    acc_a, rec_a = train_classifier(enc_a, X_tr, y_tr, freeze_encoder=True)

    # B. Random-init encoder, train everything end-to-end
    torch.manual_seed(n_labels)
    enc_b = Encoder(V)
    acc_b, rec_b = train_classifier(enc_b, X_tr, y_tr, freeze_encoder=False)

    print(f"{n_labels:>7} | {acc_a:>14.3f}  {rec_a:>17.3f} | {acc_b:>12.3f}  {rec_b:>15.3f}")

print()
print("Pre-trained encoder + linear probe wins big when labels are scarce.")
print("That's the whole foundation-model pitch in miniature.")
