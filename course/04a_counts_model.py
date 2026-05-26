"""
Lesson 4a — Counts model: a baseline with NO neural network at all.

Phase 2 begins. Over the next four lessons (L4a → L4d) we'll build PRAGMA
up one piece at a time. Same data, same task throughout. Each version is
a small delta from the previous, and each version's loss should be lower
than the one before.

This is Karpathy's "build the dumbest version first" move. Before any
embeddings, attention, or even a single trainable parameter, we'll see
how far we can get with simple bookkeeping.

The task (same throughout the series):
  - Take a pet event of (pet, action, place) — 6 tokens with keys + values.
  - Mask one value token (e.g., the place).
  - Predict it.
  - The "rules" are: dogs play in gardens, cats sleep on couches, etc.

Run:  python3 04a_counts_model.py
"""

import math
import random
from collections import Counter, defaultdict

random.seed(0)


# ============================================================================
# STEP 1 — The dataset (we'll reuse this for L4a, L4b, L4c, L4d)
# ============================================================================

KEYS   = ["pet", "action", "place"]
VALUES = ["dog", "cat", "fish",          # pets
          "eat", "sleep", "play",        # actions
          "garden", "couch", "bowl"]     # places

# The "secret rules" — model doesn't see them
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


# Generate train + test splits
def make_dataset(n):
    return [random_event() for _ in range(n)]


train = make_dataset(5000)
test  = make_dataset(1000)
print("=" * 70)
print("STEP 1 — Dataset")
print("=" * 70)
print(f"Train: {len(train)} events")
print(f"Test:  {len(test)} events")
print(f"\nFirst 5 train events:")
for e in train[:5]:
    print(f"  {e}")
print()


# ============================================================================
# STEP 2 — The counts model
# ============================================================================
# Idea: for each (visible_key, visible_value, target_key) combination,
# count how often each target value appeared in the training data.
# At test time, pick the most-frequent target value (or sample from a
# probability distribution).
#
# Concretely: given pet=dog and we're asked to predict the place,
# we look up the empirical distribution of `place` values across all
# training events where pet=dog. The model's "prediction" is that
# distribution.
#
# This is a NON-NEURAL model. It's just bookkeeping over the training set.
# No trainable parameters at all.

print("=" * 70)
print("STEP 2 — Build the counts model")
print("=" * 70)

# counts[(visible_key, visible_value, target_key)] -> Counter of target values
counts = defaultdict(Counter)

# For each event, for each pair of (visible_field, target_field), record
# what the target field's value was.
for ev in train:
    for vis_k in KEYS:
        for tgt_k in KEYS:
            if vis_k == tgt_k:
                continue
            counts[(vis_k, ev[vis_k], tgt_k)][ev[tgt_k]] += 1

# Convert each Counter to a probability distribution
def probs_for(vis_k, vis_v, tgt_k):
    c = counts[(vis_k, vis_v, tgt_k)]
    total = sum(c.values())
    if total == 0:
        # never seen this — fall back to uniform
        return {v: 1.0 / len(VALUES) for v in VALUES}
    return {v: cnt / total for v, cnt in c.items()}


print("Example probability distributions learned from the data:")
print()
print(f"  P(place | pet=dog):    {probs_for('pet', 'dog',  'place')}")
print(f"  P(place | pet=cat):    {probs_for('pet', 'cat',  'place')}")
print(f"  P(place | pet=fish):   {probs_for('pet', 'fish', 'place')}")
print(f"  P(action | pet=dog):   {probs_for('pet', 'dog',  'action')}")
print()
print("This model has ZERO trainable parameters — just a lookup table.")
print()


# ============================================================================
# STEP 3 — Evaluate: cross-entropy loss + accuracy on test set
# ============================================================================
# For each test event, we hide ONE of its values and ask the model to
# predict it from the OTHER two. We score:
#   - accuracy: was the most-likely guess correct?
#   - cross-entropy: -log(probability the model assigned to the correct answer)

print("=" * 70)
print("STEP 3 — Evaluate the counts model")
print("=" * 70)


def evaluate(model_predict, dataset):
    """model_predict(visible_dict, target_key) -> {value: prob, ...}"""
    correct = 0
    total = 0
    losses = []
    for ev in dataset:
        # Try hiding each field in turn
        for tgt_k in KEYS:
            visible = {k: v for k, v in ev.items() if k != tgt_k}
            probs = model_predict(visible, tgt_k)
            true_v = ev[tgt_k]
            pred_v = max(probs.items(), key=lambda kv: kv[1])[0]
            correct += int(pred_v == true_v)
            total += 1
            # cross-entropy = -log(prob of true label)
            p_true = probs.get(true_v, 1e-9)
            losses.append(-math.log(max(p_true, 1e-9)))
    return correct / total, sum(losses) / len(losses)


def counts_predict(visible, target_key):
    """Merge predictions from each visible field via simple averaging."""
    distribs = []
    for vis_k, vis_v in visible.items():
        distribs.append(probs_for(vis_k, vis_v, target_key))
    # Average the probability distributions
    avg = defaultdict(float)
    for d in distribs:
        for k, v in d.items():
            avg[k] += v / len(distribs)
    return dict(avg)


acc, loss = evaluate(counts_predict, test)
print(f"  Test accuracy:        {acc * 100:.1f}%")
print(f"  Test cross-entropy:   {loss:.3f}")
print()
print("This is our BASELINE. Every subsequent lesson should beat it.")
print()


# ============================================================================
# STEP 4 — Why we can't do much better with counts
# ============================================================================
# The counts model handles each (visible_field, value) pair INDEPENDENTLY,
# then averages. It can't combine signals — e.g., it knows P(place | pet=cat)
# and P(place | action=sleep) separately, but it can't reason about
# "place GIVEN pet=cat AND action=sleep TOGETHER".
#
# For richer reasoning we need a model that can MIX features. That's
# Lesson 4b (embeddings + linear), where each pet/action/place gets a
# vector and the linear head combines them.

print("=" * 70)
print("STEP 4 — What this model CAN'T do")
print("=" * 70)
print("Look at this case:")
print("  visible: pet=dog, action=sleep   →   predict place?")
print(f"  counts_predict: {dict(sorted(counts_predict({'pet':'dog','action':'sleep'}, 'place').items(), key=lambda kv: -kv[1])[:3])}")
print()
print("'dog' rules say places=[garden, bowl]. But dog NEVER sleeps in the")
print("rules — so the dog-sleep combo never appears in training. The counts")
print("model has no way to express 'this combination is illegal'.")
print()
print("A model with mixable features can learn: 'dog + sleep is impossible →")
print("any place is equally unlikely → maybe predict uniform'. Or even better,")
print("'fall back to what cats do, since cats DO sleep.'")
print()


# ============================================================================
# STEP 5 — What's coming next
# ============================================================================
# L4b: replace the counts table with LEARNED embeddings + a linear head.
#      Each pet/action/place value gets a vector. The model learns those
#      vectors AND a linear function from (concatenated embeddings) → output
#      probability. This combines features in a way counts can't.
#
# L4c: add ATTENTION on top of the embeddings. Each token "looks at" the
#      other tokens before predicting. Even more expressive.
#
# L4d: full Transformer block — attention + feed-forward + residual +
#      LayerNorm. Now you've recreated pragma_mini.py from scratch.
#
# At each step we'll re-evaluate on the same test set and watch the loss
# drop.


# ============================================================================
# STEP 6 — Things to try
# ============================================================================
# 1. Modify counts_predict to take the PRODUCT of the visible distributions
#    instead of the average. (Hint: that's the "naive Bayes" approach.) Does
#    it work better or worse?
#
# 2. Add "smoothing": when a count is zero, add a small constant before
#    dividing. This avoids assigning probability zero to anything (which
#    would give infinite cross-entropy loss on rare cases).
#
# 3. Compute the THEORETICAL BEST loss the counts model can achieve. Each
#    pet has 2 valid actions and 1-2 valid places (per RULES), so the model
#    can never get below H(uniform over valid options) per masked value.
