"""
Model A — same linear regression as before, but now we also print the
GRADIENTS at every step.

The gradient for each knob = "if I increase this knob by 1, how does loss
change?" PyTorch computes these for us when we call loss.backward().

The optimiser then nudges each knob in the OPPOSITE direction of its
gradient (because we want loss to go DOWN, not up).
"""
import torch

torch.manual_seed(0)

x      = torch.tensor([1., 2., 3., 4., 5.])
y_true = 2 * x + 1

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.05
opt = torch.optim.SGD([w, b], lr=lr)

print(f"{'step':>4} | {'w':>7} | {'b':>7} | {'loss':>7} | {'grad_w':>8} | {'grad_b':>8} | {'update_w':>10} | {'update_b':>10}")
print("-" * 90)

for step in range(11):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    opt.zero_grad()
    loss.backward()

    # PEEK at the gradients (computed by loss.backward()) BEFORE we step
    grad_w = w.grad.item()
    grad_b = b.grad.item()

    # The update rule: new = old - lr * gradient
    update_w = -lr * grad_w
    update_b = -lr * grad_b

    # Print the state BEFORE the step happens
    print(f"{step:>4} | {w.item():>7.3f} | {b.item():>7.3f} | {loss.item():>7.3f} | "
          f"{grad_w:>+8.2f} | {grad_b:>+8.2f} | {update_w:>+10.3f} | {update_b:>+10.3f}")

    # Now apply the step
    opt.step()

print(f"\nFinal: w = {w.item():.3f}, b = {b.item():.3f}")
print(f"(Each row's 'update_w' = -lr * grad_w. New w next step = current w + update_w.)")
