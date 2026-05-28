"""
Lesson 4g — Spelling from scratch: a CHARACTER-level model (nursery rhymes)

Runnable companion to the "from memorizer to general-purpose" section of the
Lesson 4f webpage.

Lesson 4f built a next-WORD model: it knew 31 whole words and predicted the next
word. Neat sentences — but it can only ever use those 31 words. Type "puppy" and
it has no box for it. That is NOT how ChatGPT is general purpose.

This file makes the real leap: a next-CHARACTER model. Its whole alphabet is ~25
letters and it predicts the next letter, one at a time. Out of a handful of
letters it can spell ANY word — including ones it never saw in training. That
open vocabulary is exactly what "general purpose" means.

Same pipeline as everywhere else (embed -> attention -> feed-forward -> head ->
softmax, causal mask). The ONLY change vs 4f is the size of a token:
word -> character.

Run:  python3 04g_char_rhymes.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# 1. The corpus — nursery rhymes as one long stream of text (not a word list)
# ----------------------------------------------------------------------------
CORPUS = """twinkle twinkle little star
how i wonder what you are
up above the world so high
like a diamond in the sky
twinkle twinkle little star
how i wonder what you are

humpty dumpty sat on a wall
humpty dumpty had a great fall
all the kings horses and all the kings men
could not put humpty together again

baa baa black sheep have you any wool
yes sir yes sir three bags full
one for the master and one for the dame
and one for the little boy who lives down the lane

jack and jill went up the hill
to fetch a pail of water
jack fell down and broke his crown
and jill came tumbling after

hickory dickory dock
the mouse ran up the clock
the clock struck one the mouse ran down
hickory dickory dock

the itsy bitsy spider climbed up the water spout
down came the rain and washed the spider out
out came the sun and dried up all the rain
and the itsy bitsy spider climbed up the spout again

mary had a little lamb its fleece was white as snow
and everywhere that mary went the lamb was sure to go
it followed her to school one day which was against the rule
it made the children laugh and play to see a lamb at school

little miss muffet sat on a tuffet
eating her curds and whey
along came a spider who sat down beside her
and frightened miss muffet away

row row row your boat gently down the stream
merrily merrily merrily merrily life is but a dream

old macdonald had a farm
and on his farm he had a cow
with a moo moo here and a moo moo there
here a moo there a moo everywhere a moo moo

the wheels on the bus go round and round
round and round round and round
the wheels on the bus go round and round
all through the town

rain rain go away come again another day
little children want to play rain rain go away

hey diddle diddle the cat and the fiddle
the cow jumped over the moon
the little dog laughed to see such sport
and the dish ran away with the spoon
"""

# ----------------------------------------------------------------------------
# 2. Vocabulary — CHARACTERS, not words (the key change vs 4f)
#    Every distinct character gets an id. No <bos>/<eos>/<pad>: we slide a
#    window over the raw text, so there is nothing to pad.
# ----------------------------------------------------------------------------
chars = sorted(set(CORPUS))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
T = 64  # context window: how many characters the model sees at once

print(f"corpus: {len(CORPUS)} chars, {CORPUS.count(chr(10))} lines")
print(f"alphabet: V = {V} characters -> {chars}\n")


# ----------------------------------------------------------------------------
# 3. The model — a character Transformer (same as 4f, stackable layers)
# ----------------------------------------------------------------------------
class CharGPT(nn.Module):
    def __init__(self, d=64, nhead=4, ff=128, nlayers=2):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, nhead, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        return self.head(self.enc(h, mask=mask))


# ----------------------------------------------------------------------------
# 4. Training data — slide a window, predict the next character
#    input  = chars[i : i+T]
#    target = chars[i+1 : i+T+1]   (same window, shifted by one)
# ----------------------------------------------------------------------------
def get_batch(bs=32):
    ix = torch.randint(0, len(data) - T - 1, (bs,))
    xb = torch.stack([data[i:i + T] for i in ix])
    yb = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return xb, yb


@torch.no_grad()
def sample(model, seed="the ", n=110, temp=0.7):
    model.eval()
    ids = [stoi[c] for c in seed if c in stoi] or [stoi[" "]]
    for _ in range(n):
        logits = model(torch.tensor([ids[-T:]]))[0, -1] / temp
        p = F.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(p, 1)))
    return "".join(itos[i] for i in ids)


def train(nlayers=2, steps=4000, show=False):
    torch.manual_seed(0)
    model = CharGPT(nlayers=nlayers)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    checkpoints = {0, 150, 400, 1000, 2500, steps}
    for step in range(steps + 1):
        if show and step in checkpoints:
            print(f"--- step {step:>4} ---")
            print(sample(model)[:110], "\n")
            model.train()
        if step == steps:
            break
        xb, yb = get_batch(32)
        loss = F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        xb, yb = get_batch(128)
        final = F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1)).item()
    return model, sum(p.numel() for p in model.parameters()), final


# ----------------------------------------------------------------------------
# 5. Train and WATCH IT LEARN TO SPELL: noise -> words -> whole rhymes
# ----------------------------------------------------------------------------
print("Training a character model (watch the samples improve)...\n")
model, params_2L, loss_2L = train(nlayers=2, steps=4000, show=True)
print(f"2-layer model: {params_2L:,} parameters, final loss {loss_2L:.3f}\n")

# ----------------------------------------------------------------------------
# 6. Generate — give it the start of any rhyme, character by character
# ----------------------------------------------------------------------------
print("Generation (seed -> continuation):")
for seed in ["twinkle twinkle ", "the cat ", "jack and jill ", "humpty "]:
    print(f"  {seed!r} ->")
    print("   ", sample(model, seed, n=90).replace("\n", "\n    "))
    print()

# ----------------------------------------------------------------------------
# 7. Why this is "general purpose": the out-of-vocabulary wall
# ----------------------------------------------------------------------------
word_vocab = ["box", "boy", "cat", "couch", "dog", "door", "girl", "he", "i", "in",
              "likes", "mat", "on", "opened", "park", "play", "plays", "ran", "run",
              "sat", "she", "sleep", "sleeps", "store", "the", "to", "we", "went"]
print("Out-of-vocabulary wall (4f's 31-word model vs this char model):")
for w in ["cat", "puppy", "banana", "dinosaur"]:
    a = "OK" if w in word_vocab else "CANT READ IT"
    b = "can spell it" if all(c in stoi for c in w) else "missing a letter"
    print(f"  {w:9s} whole-word: {a:14s} character: {b}")

# ----------------------------------------------------------------------------
# 8. Now stacking layers finally helps (it did nothing on the 4f toy)
# ----------------------------------------------------------------------------
print("\nDepth experiment (1 vs 2 layers on this harder task):")
_, p1, l1 = train(nlayers=1, steps=4000)
print(f"  1 layer : {p1:,} params, final loss {l1:.3f}")
print(f"  2 layers: {params_2L:,} params, final loss {loss_2L:.3f}")
print("  -> more params AND lower loss: depth helps when the task is hard enough.")

# ----------------------------------------------------------------------------
# Exercises
# ----------------------------------------------------------------------------
# 1. Replace CORPUS with your own songs/story (lowercase), re-run, learn that style.
# 2. In sample(), try temp=0.3 (safe, repetitive) vs temp=1.2 (wild, more typos).
# 3. Make it bigger: CharGPT(d=96, nlayers=3). Lower loss? Does it overfit?
# 4. Go smaller: CharGPT(d=16, nlayers=1). How garbled does spelling get?
# 5. Add capitals/punctuation to CORPUS and watch V grow — each symbol is a new id.
