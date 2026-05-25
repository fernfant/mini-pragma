"""
Model A — Linear regression (Lesson 1, with parameter tracking).

ARCHITECTURE: y = w*x + b
  KNOBS: 2  (just w and b)

This is the simplest possible "model" — two numbers and a multiplication.
Watch how training nudges each knob over time.
"""
import torch

torch.manual_seed(0)

# The data
x      = torch.tensor([1., 2., 3., 4., 5.])
y_true = 2 * x + 1

# THE ARCHITECTURE: y = w*x + b. Two knobs.
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
knobs = [w, b]

# THE TRAINING RECIPE: identical to Model B (only the architecture differs)
opt = torch.optim.SGD(knobs, lr=0.05)

print("=" * 60)
print("MODEL A — Linear regression")
print("  Architecture: y = w*x + b")
print(f"  Total knobs:  {sum(k.numel() for k in knobs)}  (w, b)")
print("=" * 60)
print(f"\nINITIAL KNOBS:   w = {w.item():.3f}    b = {b.item():.3f}\n")

print(f"{'step':>5} | {'w':>7} | {'b':>7} | {'loss':>8}")
print("-" * 40)
for step in range(101):
    y_pred = w * x + b                       # 1. guess
    loss   = ((y_pred - y_true) ** 2).mean() # 2. measure wrongness
    opt.zero_grad()                          # 3. clear notes
    loss.backward()                          # 4. compute gradients
    opt.step()                               # 5. nudge knobs
    if step % 20 == 0:
        print(f"{step:>5} | {w.item():>7.3f} | {b.item():>7.3f} | {loss.item():>8.4f}")

print(f"\nFINAL KNOBS:     w = {w.item():.3f}    b = {b.item():.3f}")
print(f"(The secret rule was y = 2x + 1.  We learned: y ≈ {w.item():.2f}x + {b.item():.2f}.)")
