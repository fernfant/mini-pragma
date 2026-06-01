"""
Lesson 1 — How a model learns

Before transformers, before any of the fancy stuff, let's see what
"learning" actually means.

A model is just:
    1. A bag of numbers (the "weights").
    2. A rule for combining those numbers with input to produce an output.

Training means: show it examples, see how wrong it is, nudge the numbers
to be a little less wrong, repeat thousands of times.

That's it. That's the whole game. The Transformer in pragma_mini.py is
doing the same thing — just with millions of numbers instead of two, and
a much fancier "rule".

Run:  python3 01_how_models_learn.py
"""
import torch

torch.manual_seed(0)

# The secret rule that the model doesn't know:  y = 2*x + 1
# In real life this rule could be "given these events, will the user default
# on a loan?" — but the principle is identical.
x      = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y_true = 2 * x + 1

# Our model is two numbers. We start them at 0 — totally wrong.
# The `requires_grad=True` part tells PyTorch: "these are the parameters to learn".
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# An optimiser nudges w and b based on how wrong we are.
# SGD = Stochastic Gradient Descent. "Walk downhill on the loss landscape."
opt = torch.optim.SGD([w, b], lr=0.01)

print(f"Before training:  y ≈ {w.item():.2f}*x + {b.item():.2f}")

for step in range(1000):
    y_pred = w * x + b                          # the model's guess
    loss   = ((y_pred - y_true) ** 2).mean()    # how wrong are we?
    opt.zero_grad()
    loss.backward()                             # which way is "less wrong"?
    opt.step()                                  # take one tiny step that way
    if step % 200 == 0:
        print(f"  step {step:4d}   w={w.item():.3f}   b={b.item():.3f}   loss={loss.item():.4f}")

print(f"After training:   y ≈ {w.item():.2f}*x + {b.item():.2f}")
print("(The secret rule was y = 2*x + 1. We re-discovered it.)")

# Why this matters:
#   The Transformer you'll see in later lessons has MILLIONS of numbers
#   instead of just `w` and `b`. But the training loop is exactly the same:
#       1. Make a guess.
#       2. Compute the loss.
#       3. loss.backward()  — figure out which way to nudge every number.
#       4. opt.step()       — nudge them.
#       5. Repeat.

# ----------------------------------------------------------------------------
# Exercises
# ----------------------------------------------------------------------------
# 1. Change the secret rule to y = 3*x - 2 (just edit the line above).
#    Does the model still figure it out?
#
# 2. Set lr=0.0001 (a tiny step size). How many steps until it learns?
#    Now try lr=1.0 (huge step). What happens? (You'll see why "learning rate"
#    is one of the most important things to tune.)
#
# 3. Start with w = torch.tensor(100.0, requires_grad=True). Can it still
#    find the right answer? How many steps does it take?
