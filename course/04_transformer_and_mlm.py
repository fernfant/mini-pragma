"""
Lesson 4 — A Transformer block + the fill-in-the-blank game

You now know:
  - Lesson 1: how a model "learns" by adjusting numbers.
  - Lesson 2: how words become vectors (tokens + embeddings).
  - Lesson 3: how attention lets words look at each other.

A TRANSFORMER BLOCK is just:
  attention  ->  feed-forward network  ->  (with normalisation + skip connections)

You stack a few of these blocks on top of each other and you get an ENCODER.

But we still need a TRAINING GAME — something the model can practice on,
without any humans having to label data. The trick BERT (and PRAGMA) use:
the FILL-IN-THE-BLANK game, also called MASKED LANGUAGE MODELLING (MLM).

  1. Take a sequence.
  2. Hide a few tokens with <mask>.
  3. Ask the model to guess what was there.
  4. Score it. Nudge the weights. Repeat.

We'll train a tiny Transformer to learn three rules:
  dog -> bark, cat -> meow, fish -> swim.

Run:  python3 04_transformer_and_mlm.py
"""
import random
import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)

# ----------------------------------------------------------------------------
# Vocab + data (same setup as Lesson 2)
# ----------------------------------------------------------------------------
vocab = ["<pad>", "<mask>", "dog", "cat", "fish", "bark", "meow", "swim"]
tok2id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

PAIRS = [("dog", "bark"), ("cat", "meow"), ("fish", "swim")]

# ----------------------------------------------------------------------------
# Make a training example: pick a pair, hide one of the two tokens
# ----------------------------------------------------------------------------
def make_example():
    animal, sound = random.choice(PAIRS)
    ids = [tok2id[animal], tok2id[sound]]
    labels = [-100, -100]                       # -100 means "don't score"
    hide_position = random.choice([0, 1])
    labels[hide_position] = ids[hide_position]  # remember the real answer
    ids[hide_position] = tok2id["<mask>"]       # hide it
    return ids, labels

# ----------------------------------------------------------------------------
# A tiny Transformer
# ----------------------------------------------------------------------------
# Three parts:
#   emb  — embedding lookup (Lesson 2)
#   enc  — one Transformer encoder layer (attention + feed-forward; Lesson 3)
#   head — a linear layer that maps the model's output vectors back to vocab
#          scores (so we can pick which word it thinks goes in the blank).
class TinyMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(V, 16)
        self.enc  = nn.TransformerEncoderLayer(
            d_model=16,            # vector dimension
            nhead=2,               # multi-head attention (2 heads)
            dim_feedforward=32,    # size of the internal feed-forward layer
            batch_first=True,
        )
        self.head = nn.Linear(16, V)

    def forward(self, x):
        h = self.emb(x)            # tokens   -> vectors
        h = self.enc(h)            # vectors  -> contextualised vectors
        return self.head(h)        # vectors  -> scores for each vocab word

model = TinyMLM()

# ----------------------------------------------------------------------------
# The training loop (same shape as Lesson 1!)
# ----------------------------------------------------------------------------
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

# CrossEntropyLoss is the standard "how wrong is your guess?" for picking
# one out of many classes. ignore_index=-100 means "skip positions we
# weren't asking the model to predict".
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

print("Training the model to learn animal -> sound...\n")
for step in range(500):
    batch = [make_example() for _ in range(32)]
    ids, labels = zip(*batch)
    x = torch.tensor(ids)
    y = torch.tensor(labels)

    logits = model(x)                                       # forward pass
    loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))    # how wrong?
    opt.zero_grad()
    loss.backward()                                         # which way?
    opt.step()                                              # nudge

    if step % 100 == 0:
        print(f"  step {step:3d}   loss {loss.item():.3f}")

# ----------------------------------------------------------------------------
# Play with the trained model
# ----------------------------------------------------------------------------
def fill_blank(animal_or_sound, position):
    ids = [tok2id["<mask>"], tok2id["<mask>"]]
    ids[1 - position] = tok2id[animal_or_sound]
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x)
    return vocab[logits[0, position].argmax().item()]

print("\nFill in the blank:")
for animal, _ in PAIRS:
    print(f"  {animal:5s} -> {fill_blank(animal, 1)}")
print("And going the other way:")
for _, sound in PAIRS:
    print(f"  ?     -> {sound:5s}   (model guesses: {fill_blank(sound, 0)})")

# That's BERT, in about 30 lines of model code.
# `pragma_mini.py` is the same recipe — just with a bigger vocabulary, longer
# sequences, and a richer training game.

# ----------------------------------------------------------------------------
# Exercises
# ----------------------------------------------------------------------------
# 1. Add a fourth pair, e.g. ("bird", "tweet"). Update the vocab, retrain,
#    and check that the model picks it up.
#
# 2. Reduce training to 20 steps. The loss is still high — what do the
#    fill-in-the-blank predictions look like now? (Proof that the learning
#    is real, not built in.)
#
# 3. The model has a few hundred parameters. Print
#       sum(p.numel() for p in model.parameters())
#    Now compare to PRAGMA-Large from the paper: 1,000,000,000 parameters.
#    A factor of how much?
#
# 4. (Stretch) The encoder layer has a `nhead=2` argument. Try `nhead=1` and
#    `nhead=4`. Does it train faster or slower? Better or worse?
