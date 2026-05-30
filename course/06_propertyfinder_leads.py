"""
Capstone — Predicting viewing-request leads on a property portal (PropertyFinder-style).

A property site wants to know, mid-session, whether a visitor is a serious buyer
who will request a home viewing (a "lead") or just window-shopping. Each visit is
a CLICKSTREAM: a sequence of events, where every event carries what the user did
AND the attributes of the listing they did it on. We apply the exact PRAGMA recipe
from 05b_streaming_churn.py:

  1. Generate synthetic sessions — "hot" buyers vs "cold" browsers.
  2. PRE-TRAIN a Transformer on ALL clickstream events with fill-in-the-blank (no labels).
  3. FREEZE the encoder, add a tiny classifier head.
  4. Train ONLY the head on a small labelled set.
  5. Compare to a from-scratch baseline.

The "contact agent / request a viewing" action is the OUTCOME (the label) — it is
NOT in the event sequence. The model must read INTENT from the browsing behaviour
that came before it: how focused the search is, and how deep the engagement goes.

Run:  python3 propertyfinder_leads.py
"""

import random, copy
import torch
import torch.nn as nn

torch.manual_seed(7)
random.seed(7)
torch.set_printoptions(precision=3, sci_mode=False)

# ============================================================================
# PART 1 — Synthetic clickstream data
# ============================================================================
# Each event = 4 (key, value) pairs:
#   action — what the user did on this step
#   price  — price band of the listing in view
#   beds   — bedroom count of the listing
#   area   — neighbourhood of the listing
#
# Two visitor types:
#   HOT  — a real buyer: a focused search (consistent price/beds/area) and deep
#          engagement (photos, floorplan, map, save, mortgage calc, re-views).
#   COLD — window-shopping: scattered listings (price/beds/area jump around) and
#          shallow actions (search, glance, back, share). Never converts.

KEYS    = ["action", "price", "beds", "area"]
# NOTE: no "contact"/"request_viewing" here — that is the label, not a predictor.
ACTIONS = ["search", "view", "photos", "floorplan", "map", "save", "mortgage", "share", "back"]
PRICE   = ["budget", "mid", "premium", "luxury"]
BEDS    = ["studio", "1bed", "2bed", "3bed", "4plus"]
AREAS   = ["marina", "downtown", "jbr", "businessbay", "jvc", "palm"]

PAD, MASK = "<pad>", "<mask>"
vocab  = [PAD, MASK] + KEYS + ACTIONS + PRICE + BEDS + AREAS
tok2id = {t: i for i, t in enumerate(vocab)}
id2tok = {i: t for t, i in tok2id.items()}
V      = len(vocab)

EVENTS_PER_SESSION = 12
N_SESSIONS         = 2500
LEAD_RATE          = 0.12


def hot_event(target):
    """A serious buyer: deep-engagement actions, listing usually matches their target."""
    a = random.choices(ACTIONS, weights=[1, 6, 7, 4, 3, 5, 3, 1, 1])[0]  # photos/view/save heavy
    # 80% of the time they're looking at something matching their real need
    focused = random.random() < 0.80
    p = target["price"] if focused else random.choice(PRICE)
    b = target["beds"]  if focused else random.choice(BEDS)
    ar = target["area"] if focused else random.choice(AREAS)
    return [("action", a), ("price", p), ("beds", b), ("area", ar)]


def cold_event():
    """Window-shopper: shallow actions, listings jump all over the place."""
    a = random.choices(ACTIONS, weights=[7, 5, 1, 1, 1, 1, 1, 4, 6])[0]  # search/back/share heavy
    return [("action", a),
            ("price", random.choice(PRICE)),
            ("beds",  random.choice(BEDS)),
            ("area",  random.choice(AREAS))]


def make_session(is_lead):
    if is_lead:
        target = {"price": random.choice(PRICE), "beds": random.choice(BEDS), "area": random.choice(AREAS)}
        return [hot_event(target) for _ in range(EVENTS_PER_SESSION)]
    return [cold_event() for _ in range(EVENTS_PER_SESSION)]


def encode_event(event):
    ids = []
    for k, v in event:
        ids.append(tok2id[k]); ids.append(tok2id[v])
    return ids


def encode_session(events):
    return [tok for e in events for tok in encode_event(e)]


print("Generating", N_SESSIONS, "synthetic browsing sessions...")
sessions, labels = [], []
for _ in range(N_SESSIONS):
    lead = random.random() < LEAD_RATE
    sessions.append(make_session(lead))
    labels.append(1 if lead else 0)

X = torch.tensor([encode_session(s) for s in sessions], dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)
seq_len = X.size(1)
print(f"  X shape: {tuple(X.shape)}  ({seq_len} tokens per session = {EVENTS_PER_SESSION} events x 4 pairs x 2)")
print(f"  labels:  {int(y.sum())} leads,  {int((y == 0).sum())} no-lead\n")

# Show one of each so the dataset is legible
def show(events, tag):
    print(f"  [{tag}]")
    for e in events[:4]:
        print("     " + "  ".join(f"{k}={v}" for k, v in e))
    print("     ...")
lead_i  = labels.index(1)
cold_i  = labels.index(0)
print("Sample sessions (first 4 events):")
show(sessions[lead_i], "HOT  -> lead")
show(sessions[cold_i], "COLD -> no lead")
print()


# ============================================================================
# PART 2 — Architecture (same shape as pragma_mini.py / 05b)
# ============================================================================
D_MODEL, N_HEADS, N_LAYERS = 32, 2, 2


class Encoder(nn.Module):
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
    def __init__(self, V, d=D_MODEL):
        super().__init__(); self.proj = nn.Linear(d, V)
    def forward(self, h): return self.proj(h)


class LeadHead(nn.Module):
    """Pool the session into one vector, predict lead / no-lead."""
    def __init__(self, d=D_MODEL):
        super().__init__(); self.proj = nn.Linear(d, 2)
    def forward(self, h): return self.proj(h.mean(dim=1))


# ============================================================================
# PART 3 — Pre-train via masked language modelling (no labels)
# ============================================================================
KEY_IDS = torch.tensor([tok2id[k] for k in KEYS])

def mlm_mask(X_batch, p=0.20):
    X = X_batch.clone()
    y = torch.full_like(X, -100)
    is_value = ~torch.isin(X, KEY_IDS)            # only ever hide VALUE tokens
    pick = (torch.rand_like(X, dtype=torch.float) < p) & is_value
    y[pick] = X[pick]; X[pick] = tok2id[MASK]
    return X, y

encoder  = Encoder(V)
mlm_head = MLMHead(V)
opt      = torch.optim.AdamW(list(encoder.parameters()) + list(mlm_head.parameters()), lr=3e-3)
loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)

print("Pre-training encoder via masked-event modelling for 2000 steps...")
print(f"  encoder + MLM head knobs: {sum(p.numel() for p in list(encoder.parameters()) + list(mlm_head.parameters())):,}")
for step in range(2000):
    idx    = torch.randint(0, N_SESSIONS, (64,))
    xb, yb = mlm_mask(X[idx])
    logits = mlm_head(encoder(xb))
    loss   = loss_fn(logits.reshape(-1, V), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        print(f"  step {step:4d}   MLM loss {loss.item():.3f}")
pretrained_encoder = copy.deepcopy(encoder)
print()


# ============================================================================
# PART 4 — Downstream task: predict the LEAD
# ============================================================================
perm  = torch.randperm(N_SESSIONS)
split = int(N_SESSIONS * 0.8)
tr_idx, te_idx = perm[:split], perm[split:]
X_te, y_te = X[te_idx], y[te_idx]
print(f"Test set: {len(te_idx)} sessions  ({int(y_te.sum())} leads)\n")


def freeze(mod):
    for p in mod.parameters(): p.requires_grad = False
    mod.eval()


def train_classifier(encoder, X_tr, y_tr, epochs=200, freeze_encoder=True):
    head = LeadHead()
    if freeze_encoder:
        freeze(encoder); params = list(head.parameters())
    else:
        params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        logits = head(encoder(X_tr))
        loss   = loss_fn(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred  = head(encoder(X_te)).argmax(-1)
        acc   = (pred == y_te).float().mean().item()
        leads = (y_te == 1)
        rec   = (pred[leads] == 1).float().mean().item() if leads.sum() > 0 else float("nan")
        flagged = (pred == 1)
        prec  = (y_te[flagged] == 1).float().mean().item() if flagged.sum() > 0 else float("nan")
    return acc, rec, prec


print(f"{'labels':>7} | {'pretrained: acc':>15} {'recall':>7} {'prec':>6} | {'baseline: acc':>14} {'recall':>7} {'prec':>6}")
print("-" * 80)
for n_labels in [20, 50, 200, 1000]:
    sub = tr_idx[:n_labels]
    X_tr, y_tr = X[sub], y[sub]
    enc_a = copy.deepcopy(pretrained_encoder)
    acc_a, rec_a, prec_a = train_classifier(enc_a, X_tr, y_tr, freeze_encoder=True)
    torch.manual_seed(n_labels)
    enc_b = Encoder(V)
    acc_b, rec_b, prec_b = train_classifier(enc_b, X_tr, y_tr, freeze_encoder=False)
    print(f"{n_labels:>7} | {acc_a:>15.3f} {rec_a:>7.3f} {prec_a:>6.3f} | {acc_b:>14.3f} {rec_b:>7.3f} {prec_b:>6.3f}")

print()
print("With few labels, the pre-trained encoder + a tiny probe spots serious buyers")
print("the from-scratch model misses — the foundation-model pitch, on clickstream.")
