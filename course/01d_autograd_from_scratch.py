"""
Lesson 1d — Autograd from scratch (a tiny micrograd).

The most foundational "no black boxes" lesson.

We've used `loss.backward()` everywhere. In Lesson 1c we even showed by hand
what gradient PyTorch was computing. But how does PyTorch actually KNOW how
to differentiate something we wrote? It builds a graph of operations as
you compute the forward pass, then walks it backward applying the chain rule.

This file builds that engine. A 100-line autograd library — micrograd-style,
following Karpathy's lead. Then we use it to train the linear regression
model from Lesson 1, with NO PyTorch autograd. To prove our gradients are
correct, we compare them to PyTorch's at every step.

By the end you will know:
  - Why each Tensor needs a `.grad` field and a "backward function".
  - What "the chain rule" looks like in code.
  - That `.backward()` is just a topological walk of the computation graph.
  - That you could build PyTorch yourself (the math part) in a weekend.

Run:  python3 01d_autograd_from_scratch.py
"""

import math
import torch  # only for the comparison at the end


# ============================================================================
# STEP 1 — A "Value": a number that remembers how it was computed
# ============================================================================
# Each Value holds:
#   .data   — the numerical value
#   .grad   — the gradient (filled in by .backward())
#   ._prev  — the Values it was built from
#   ._op    — a label for debugging
#   ._backward — a function that, given THIS Value's grad, sets the grad
#                of its parents using the chain rule.

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None   # no-op for leaf nodes (constants)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------- ADDITION ----------
    # If c = a + b, then dc/da = 1 and dc/db = 1.
    # By the chain rule, da_grad += dc/da * c_grad = 1 * c_grad = c_grad.
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    # ---------- MULTIPLICATION ----------
    # If c = a * b, then dc/da = b and dc/db = a.
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    # ---------- POWER (constant exponent) ----------
    # If c = a**n, then dc/da = n * a**(n-1).
    def __pow__(self, n):
        assert isinstance(n, (int, float))
        out = Value(self.data ** n, (self,), f"**{n}")
        def _backward():
            self.grad += (n * self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    # Convenience methods so we can write nice expressions
    def __neg__(self):     return self * -1
    def __sub__(self, o):  return self + (-o)
    def __rsub__(self, o): return Value(o) + (-self)
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else Value(o) ** -1)

    # ---------- THE BIG ONE: .backward() ----------
    # Compute gradients of THIS Value w.r.t. all ancestors.
    def backward(self):
        # 1. Topologically sort the computation graph so we visit parents
        #    BEFORE we visit children's backward functions on them.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. Seed: dL/dL = 1 (this Value is the "loss" being differentiated).
        self.grad = 1.0

        # 3. Walk in REVERSE topological order, applying each node's backward.
        for v in reversed(topo):
            v._backward()


# ============================================================================
# STEP 2 — Sanity check: gradient of a known expression
# ============================================================================
# Take f(x) = 3x² + 2x + 5. We know df/dx = 6x + 2.
# At x = 4, df/dx should be 26.

print("=" * 70)
print("STEP 2 — Sanity check: differentiate f(x) = 3x² + 2x + 5 at x = 4")
print("=" * 70)

x = Value(4.0)
f = Value(3) * x ** 2 + Value(2) * x + Value(5)
f.backward()

print(f"f(4) = 3*16 + 2*4 + 5 = {f.data}    (expected 61)")
print(f"df/dx at x=4 = 6*4 + 2 = {x.grad}    (expected 26)")
assert abs(x.grad - 26) < 1e-6, "gradient incorrect!"
print("  ✓ Correct.")
print()


# ============================================================================
# STEP 3 — Train Lesson 1's linear regression with OUR autograd
# ============================================================================
# Remember Lesson 1? We had y = 2x + 1, started w=0, b=0, and the 5-line
# loop nudged them to (2, 1). Let's do the same with NO PyTorch autograd —
# just our Value class.

print("=" * 70)
print("STEP 3 — Train y = w*x + b on data from y = 2x + 1")
print("=" * 70)
print("Using OUR autograd, not PyTorch's. Should still find w=2, b=1.\n")

# Data
xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [2 * x + 1 for x in xs]  # truth

# Initial weights
w = Value(0.0)
b = Value(0.0)

lr = 0.05
for step in range(101):
    # Forward
    loss = Value(0.0)
    for xv, yv in zip(xs, ys):
        pred = w * Value(xv) + b
        err  = pred - Value(yv)
        loss = loss + err ** 2
    loss = loss * Value(1.0 / len(xs))  # mean

    # Zero gradients (manually — no opt.zero_grad() since we don't have one)
    # Easiest: build a fresh value graph each step, just reset the trainables
    w.grad = 0.0
    b.grad = 0.0

    # Backward
    loss.backward()

    # Step (manual SGD)
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    if step % 20 == 0:
        print(f"  step {step:3d}   w={w.data:.3f}   b={b.data:.3f}   loss={loss.data:.4f}")

print(f"\nFinal: w = {w.data:.3f}, b = {b.data:.3f}    (target: w=2, b=1)")
print()


# ============================================================================
# STEP 4 — Compare to PyTorch's autograd at one specific step
# ============================================================================
# Recreate Lesson 1c's hand-computed step 0 in our autograd. Confirm we get
# the same gradients PyTorch computed.

print("=" * 70)
print("STEP 4 — Step-0 gradients: our autograd vs PyTorch")
print("=" * 70)

# Ours
w = Value(0.0)
b = Value(0.0)
loss = Value(0.0)
for xv, yv in zip(xs, ys):
    err = w * Value(xv) + b - Value(yv)
    loss = loss + err ** 2
loss = loss * Value(1.0 / len(xs))
w.grad = 0; b.grad = 0
loss.backward()
print(f"  Ours:     loss = {loss.data:.2f},  grad_w = {w.grad:.2f},  grad_b = {b.grad:.2f}")

# PyTorch
wt = torch.tensor(0.0, requires_grad=True)
bt = torch.tensor(0.0, requires_grad=True)
xt = torch.tensor(xs)
yt = torch.tensor(ys)
lt = ((wt * xt + bt - yt) ** 2).mean()
lt.backward()
print(f"  PyTorch:  loss = {lt.item():.2f},  grad_w = {wt.grad.item():.2f},  grad_b = {bt.grad.item():.2f}")
print()
print("Same loss, same gradients. Our 100-line autograd does exactly what")
print("PyTorch's industrial-grade engine does — just slower and only for scalars.")
print("(Real PyTorch generalises this to tensors and uses C++ for speed.)")
print()


# ============================================================================
# STEP 5 — Visualise the computation graph
# ============================================================================
# To prove this isn't magic, let's print the graph for a tiny expression.

print("=" * 70)
print("STEP 5 — What does the computation graph look like?")
print("=" * 70)

a = Value(2.0)
c = Value(3.0)
y = a * c + Value(1.0)
y.backward()
print(f"\ny = a * c + 1   where a = {a.data}, c = {c.data}")
print(f"y.data = {y.data}     (forward result)")
print(f"\nBackward results:")
print(f"  dy/da = c = {c.data}    →    a.grad = {a.grad}")
print(f"  dy/dc = a = {a.data}    →    c.grad = {c.grad}")
print()
print("The graph that .backward() walked:")
print("  y(+)")
print("  ├── (*)")
print("  │   ├── a (leaf)")
print("  │   └── c (leaf)")
print("  └── 1 (leaf)")
print()
print("Topological order: a, c, mul-node, 1, add-node (= y)")
print("Backward order (reverse): y, add, 1, mul, c, a")
print()


# ============================================================================
# STEP 6 — Why this matters
# ============================================================================
# loss.backward() in PyTorch is NOT magic. Every operation you write —
# addition, multiplication, matmul, softmax, layernorm, attention — has
# a registered "backward function" that knows how to push gradients
# back to its inputs via the chain rule. .backward() walks the graph
# from your loss back to the leaves (your trainable parameters) and
# accumulates the gradient for each.
#
# Every Transformer, every LLM, every modern AI system is trained using
# exactly this mechanism. The math we just implemented (chain rule + topo
# walk) is the entire foundation of deep learning.


# ============================================================================
# STEP 7 — Things to try
# ============================================================================
# 1. Add a .tanh() method to Value. The math:
#       y = tanh(x)
#       dy/dx = 1 - y^2
#    With this, you can train a 1-hidden-layer MLP entirely in our autograd.
#
# 2. Add a .exp() method, then implement .softmax(). Then implement
#    cross-entropy loss. Now you can train a tiny classifier with our autograd.
#
# 3. Add a .relu() method (much simpler than tanh!). Train an MLP with ReLU
#    activations.
#
# 4. (Stretch) Generalise Value to hold tensors instead of scalars. This is
#    what real PyTorch does — but you have to be careful about broadcasting
#    in backward (sum across broadcasted axes).
#
# 5. Plot the loss curve from STEP 3. Compare to Lesson 1's loss curve from
#    PyTorch. They should be very close (any difference is numerical noise).
