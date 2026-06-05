"""
mini.py — the whole course in one file.

It starts (Lesson 1) as nothing but `w * x`, and by the end it is a tiny but REAL
transformer that plays fill-in-the-blank: show it "dog ___" and it answers "bark",
having learned the rule from nothing but examples. No numpy, no torch — every
number is a `Value` that remembers how it was computed, so we can see the gradient.

This is the END-STATE. The course reveals it one rung at a time (see
course/agent/plans/mini_py_growth_ladder.md). Lesson tags below show where each
piece is introduced.

Run it:   python3 mini.py
"""
import random, math

# ───────────────────────── the engine (L1.5) ─────────────────────────
# A Value is a number that remembers how it was built, so it can hand back a
# gradient: "nudge me this way to make the loss smaller." This is all of autograd.
class Value:
    def __init__(self, data, _children=()):
        self.data = data; self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data + o.data, (self, o))
        def _b(): self.grad += out.grad; o.grad += out.grad
        out._backward = _b; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data * o.data, (self, o))
        def _b(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = _b; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _b(): self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _b; return out
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _b(): self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _b; return out
    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def _b(): self.grad += out.data * out.grad
        out._backward = _b; return out
    def log(self):
        out = Value(math.log(self.data), (self,))
        def _b(): self.grad += (1.0 / self.data) * out.grad
        out._backward = _b; return out
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (o * -1 if isinstance(o, Value) else Value(-o))
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()

# ─────────────────── small linear-algebra helpers (L1.5b) ───────────────────
def dot(a, b):                                   # a·b for two vectors of Values
    s = a[0] * b[0]
    for i in range(1, len(a)): s = s + a[i] * b[i]
    return s
def linear(x, W):                                # one matrix-vector product (no bias)
    return [dot(row, x) for row in W]
def linear_b(x, W, b):                            # with a bias
    return [dot(W[i], x) + b[i] for i in range(len(W))]
def softmax(zs):                                  # scores -> a 100% budget (L3)
    m = max(z.data for z in zs)
    es = [(z - m).exp() for z in zs]
    s = es[0]
    for e in es[1:]: s = s + e
    return [e * (s ** -1) for e in es]
def cross_entropy(logits, target):                # how surprised we are at the truth (L4)
    m = max(z.data for z in logits)               # log-sum-exp form: stable, no log(0)
    es = [(z - m).exp() for z in logits]
    S = es[0]
    for e in es[1:]: S = S + e
    return (S.log() + m) - logits[target]         # = -log P(target)

# ──────────────────────────── the corpus (L2) ────────────────────────────
# One vocabulary, used from L2 all the way to L4. Two special tokens, three
# animals, three sounds — and three hidden rules to rediscover from blanks alone.
VOCAB = ["<pad>", "<mask>", "dog", "cat", "fish", "bark", "meow", "swim"]
TOK = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
PAIRS = [("dog", "bark"), ("cat", "meow"), ("fish", "swim")]

# ──────────────────────────── the model ────────────────────────────
D = 8                                             # embedding width (L2)
SCALE = 1.0 / math.sqrt(D)                        # the 1/sqrt(d) in attention (L3a)

def _init(seed=1):
    random.seed(seed)
    rv = lambda: Value(random.uniform(-0.3, 0.3))
    mat = lambda r, c: [[rv() for _ in range(c)] for _ in range(r)]
    return {
        "emb": mat(V, D),          # L2  — one vector per word
        "pos": mat(2, D),          # L4  — "which slot am I in" (animal vs sound)
        "Wq": mat(D, D), "Wk": mat(D, D), "Wv": mat(D, D),   # L3a — query / key / value
        "Wh": mat(V, D), "bh": [Value(0.0) for _ in range(V)],  # L4 — head: vector -> word scores
    }

def params(P):
    out = []
    for k in ("emb", "pos", "Wq", "Wk", "Wv", "Wh"):
        for row in P[k]: out += row
    return out + P["bh"]

def forward(P, tokens, mask_pos):
    # 1. embed each token and add its position  (L2 + L4)
    X = [[P["emb"][TOK[t]][k] + P["pos"][i][k] for k in range(D)] for i, t in enumerate(tokens)]
    # 2. attention: every slot makes a query/key/value, the masked slot looks around (L3/L3a)
    Q = [linear(x, P["Wq"]) for x in X]
    K = [linear(x, P["Wk"]) for x in X]
    Vv = [linear(x, P["Wv"]) for x in X]
    i = mask_pos
    w = softmax([dot(Q[i], K[j]) * SCALE for j in range(len(X))])
    ctx = [sum((w[j] * Vv[j][k] for j in range(1, len(X))), w[0] * Vv[0][k]) for k in range(D)]
    # 3. head: turn the blended vector into one score per vocab word  (L4)
    return linear_b(ctx, P["Wh"], P["bh"])

def examples():
    # fill-in-the-blank: hide one slot of [animal, sound], predict what was there
    data = []
    for a, b in PAIRS:
        data.append((["<mask>", b], 0, TOK[a]))   # ___ bark  -> dog
        data.append(([a, "<mask>"], 1, TOK[b]))   # dog  ___  -> bark
    return data

def train(P, steps=400, lr=0.1):
    ps = params(P)
    data = examples()
    for _ in range(steps):
        for toks, mpos, target in data:
            loss = cross_entropy(forward(P, toks, mpos), target)
            for p in ps: p.grad = 0.0
            loss.backward()
            for p in ps: p.data -= lr * p.grad
    return P

def predict(P, tokens, mask_pos):
    p = softmax(forward(P, tokens, mask_pos))
    return VOCAB[max(range(V), key=lambda i: p[i].data)]

if __name__ == "__main__":
    P = _init()
    print("BEFORE:", [f"{a} ___ -> {predict(P, [a, '<mask>'], 1)}" for a, _ in PAIRS])
    train(P)
    print("AFTER :", [f"{a} ___ -> {predict(P, [a, '<mask>'], 1)}" for a, _ in PAIRS])
    print("       ", [f"___ {b} -> {predict(P, ['<mask>', b], 0)}" for _, b in PAIRS])
    print(f"params: {len(params(P))}")
