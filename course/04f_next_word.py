"""
Lesson 4f — Predict the next word (the ChatGPT loop)

Runnable companion to the "data journey" widget in the Lesson 4 webpage.

Lesson 4's TinyMLM plays FILL-IN-THE-BLANK (BERT-style): it sees a <mask>
in the middle of a sentence and can peek at words on BOTH sides.

This file builds the OTHER flavour — PREDICT-THE-NEXT-WORD (GPT-style). It
only sees the words BEFORE the gap, so it can keep going forever: predict a
word, add it, predict again. That's the loop that makes ChatGPT write.

Same pipeline (embed -> attention -> feed-forward -> head -> softmax). The
only change is WHERE the blank is: middle (BERT) vs always-at-the-end (GPT),
enforced by a CAUSAL MASK that stops each token looking to its right.

Run:  python3 04f_next_word.py
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

# ----------------------------------------------------------------------------
# A tiny, overlapping corpus so a small model can learn the patterns
# ----------------------------------------------------------------------------
sentences = [
    "the cat sat on the mat", "the cat sat on the couch", "the dog sat on the mat",
    "the dog likes to run", "the cat likes to sleep", "the dog likes to play",
    "the cat likes to play", "the dog likes to sleep",
    "she opened the door", "she opened the box", "he opened the door", "he opened the box",
    "i went to the park", "i went to the store", "we went to the park", "we went to the store",
    "the dog ran to the park", "the cat ran to the door", "the dog ran to the mat",
    "the boy likes to play", "the girl likes to run", "the boy likes to run", "the girl likes to play",
    "she went to the park", "he went to the store", "the cat sat on the couch",
    "the dog plays in the park", "the cat sleeps on the couch", "the boy went to the store",
]

# ----------------------------------------------------------------------------
# Vocabulary + special tokens
#   <pad> filler   <bos> start-of-sentence   <eos> end (lets the model stop)
# ----------------------------------------------------------------------------
words = sorted({w for s in sentences for w in s.split()})
vocab = ["<pad>", "<bos>", "<eos>"] + words
tok2id = {w: i for i, w in enumerate(vocab)}
id2tok = {i: w for w, i in tok2id.items()}
V = len(vocab)
PAD, BOS, EOS = tok2id["<pad>"], tok2id["<bos>"], tok2id["<eos>"]
MAXLEN = 16  # room for <bos> + prompt + generated words


def encode(s):
    return [BOS] + [tok2id[w] for w in s.split()] + [EOS]


# ----------------------------------------------------------------------------
# The model — a next-word Transformer
#   tok  : word id  -> vector              (L2)
#   pos  : position -> vector              (so order matters)
#   enc  : attention + feed-forward        (L3 + L1.5), with a CAUSAL mask
#   head : vector   -> score per word
# ----------------------------------------------------------------------------
class NextWord(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64, batch_first=True)
        self.head = nn.Linear(d, V)

    def hidden(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)  # no peeking right
        return self.enc(h, src_mask=mask)

    def forward(self, x):
        return self.head(self.hidden(x))


model = NextWord()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

# pad all sentences to one batch
rows = [encode(s) for s in sentences]
L = max(len(r) for r in rows)
X = torch.full((len(rows), L), PAD)
for i, r in enumerate(rows):
    X[i, :len(r)] = torch.tensor(r)

# ----------------------------------------------------------------------------
# Train: predict each token from the ones before it (labels shifted by one)
# ----------------------------------------------------------------------------
print("Training a next-word model...\n")
for step in range(800):
    inp, tgt = X[:, :-1], X[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"  step {step:3d}   loss {loss.item():.3f}")

model.eval()  # inference mode: dropout off, predictions clean & repeatable


# ----------------------------------------------------------------------------
# Generate — the actual ChatGPT loop: predict next word, append, repeat
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(prompt, max_new=10, greedy=True):
    ids = [BOS] + [tok2id[w] for w in prompt.split()]
    for _ in range(max_new):
        if len(ids) >= MAXLEN:
            break
        logits = model(torch.tensor([ids]))[0, -1]
        probs = F.softmax(logits, dim=-1)
        nxt = int(probs.argmax()) if greedy else int(torch.multinomial(probs, 1))
        if nxt in (PAD, EOS):
            break
        ids.append(nxt)
    return " ".join(id2tok[i] for i in ids[1:])


print("\nGeneration (greedy):")
for p in ["the cat", "the dog likes to", "she opened the", "i went to the"]:
    print(f"  {p:22s} -> {generate(p)}")

print("\nGeneration (sampled — different each time, like ChatGPT's randomness):")
for _ in range(4):
    print("  ", generate("the dog", greedy=False))


# ----------------------------------------------------------------------------
# The data journey — every stage, for one sentence (all numbers are REAL)
# ----------------------------------------------------------------------------
@torch.no_grad()
def journey(prompt, k=5):
    ids = [BOS] + [tok2id[w] for w in prompt.split()]
    x = torch.tensor([ids])
    T = x.size(1)
    toks = [id2tok[i] for i in ids]

    print(f"\nDATA JOURNEY for: {prompt!r}\n")
    print("Stage 1 — tokenise")
    print("   words:", toks)
    print("   ids:  ", ids)

    pos = torch.arange(T).unsqueeze(0)
    h = model.tok(x) + model.pos(pos)
    print("\nStage 2 — embed (first 4 of 32 dims)")
    for i, t in enumerate(toks):
        print(f"   {t:6s}", [round(float(v), 2) for v in h[0, i, :4]])

    mask = nn.Transformer.generate_square_subsequent_mask(T)
    _, attn = model.enc.self_attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=True)
    print("\nStage 3 — attention: how much the LAST word looks at each word")
    for i, t in enumerate(toks):
        w = float(attn[0, -1, i]) * 100
        print(f"   {t:6s} {w:5.1f}%  {'#' * int(w / 3)}")

    logits = model(x)[0, -1]
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    print("\nStages 4-5 — head -> scores -> softmax: top next-word predictions")
    for p, idx in zip(top.values, top.indices):
        pct = float(p) * 100
        print(f"   {id2tok[int(idx)]:6s} {pct:5.1f}%  {'=' * int(pct / 3)}")


journey("the cat sat on the")
print("\nparameters:", sum(p.numel() for p in model.parameters()))

# ----------------------------------------------------------------------------
# Exercises
# ----------------------------------------------------------------------------
# 1. Add sentences using a new word (e.g. "the bird likes to fly"), retrain,
#    and check generate("the bird").
# 2. Shrink the model: NextWord(d=8). Still coherent? How low can you go?
# 3. Pass src_mask=None in hidden() (no causal mask). Generation breaks — why?
# 4. Call generate(..., greedy=False) ten times — count the distinct outputs.
# 5. Run journey("the") and watch the probabilities spread out (genuine doubt).
