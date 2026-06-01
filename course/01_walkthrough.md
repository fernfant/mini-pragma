# Lesson 1 — line-by-line walkthrough

Companion to `01_how_models_learn.py`. Read it side-by-side with the code.

---

### Line 20: `import torch`

```python
import torch
```

**What it does:** loads the PyTorch library so we can use it.

**What is PyTorch?** A toolbox for doing math on lots of numbers at once, and — more importantly — for *automatically figuring out how to make those numbers better*. Every modern AI model you've heard of (ChatGPT, image generators, PRAGMA) is built on top of either PyTorch or its cousin TensorFlow. Today we'll use the two most important superpowers it gives us:

1. A "tensor" — a fancy list of numbers that's fast to do math on.
2. "Autograd" — a way of asking "if I want this number to go down, which way should I nudge all the other numbers?" *automatically*. We'll see this in action below.

---

### Line 22: `torch.manual_seed(0)`

```python
torch.manual_seed(0)
```

**What it does:** locks the random number generator to a specific starting point.

**Why we care:** anything random in PyTorch (random starting weights, random shuffles, etc.) becomes **reproducible**. If you run this script today and your son runs it tomorrow, you'll both see the *exact same* numbers. That's huge when debugging: if results differ, it's because of something you changed, not random luck.

**Mental model:** like saying "deal the cards using deck pattern #0." Anyone who uses deck pattern #0 will get the same hands.

---

### Lines 27–28: the data

```python
x      = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y_true = 2 * x + 1
```

**What it does:**
- `x` is a tensor of 8 numbers: 1 through 8.
- `y_true` is what we expect the model to learn to output for each x. Since we wrote `y_true = 2 * x + 1`, PyTorch computes:

| x | y_true |
|---|---|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |
| 6 | 13 |
| 7 | 15 |
| 8 | 17 |

**The setup:** the model will be shown the `x` column and the `y_true` column. It will *not* be told the rule `y = 2x + 1`. Its job is to figure out the rule from the examples.

**Why this is the simplest possible "learning":** in real life the "rule" could be hugely complicated — "given this customer's last 200 banking events, what's the chance they default on a loan?" But the *idea* is the same: you have examples of `(input, correct answer)` pairs, and you want the model to generalise the underlying rule.

---

### Lines 32–33: the model itself

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
```

**What it does:** creates two numbers, both starting at zero. These two numbers ARE the model. That's it.

**The model's "rule"** (we'll write it below in line 42) is going to be `y_pred = w * x + b`. So:
- `w` is the slope of the line.
- `b` is the y-intercept (where it crosses the y-axis).

Right now both are 0, so the model's "rule" is `y_pred = 0 * x + 0 = 0` — it always predicts 0, for any input. Totally wrong. That's fine. It's about to learn.

**`requires_grad=True` — the magic flag.** This tells PyTorch: *"These are the weights I want to tune. Please remember how to adjust them later."* PyTorch quietly starts tracking every calculation that involves `w` or `b`, so it can later answer the question: "If I want the loss to go down, which direction should I push w? Which direction should I push b?"

This is **autograd** — automatic differentiation. It's the engine that makes deep learning possible. You don't have to do any calculus yourself. PyTorch does it for you.

**Mental model:** `w` and `b` are dials on a radio. `requires_grad=True` means "these are the dials you're allowed to turn." Everything else in the program is fixed.

---

### Line 37: the optimiser

```python
opt = torch.optim.SGD([w, b], lr=0.01)
```

**What it does:** creates an "optimiser" — the object that will actually turn the dials.

Three things to notice:

1. **`SGD`** = Stochastic Gradient Descent. The name sounds scary but the idea is plain: *go downhill*. If you're standing on a hillside and want to reach the bottom, take a small step in the steepest-downhill direction, then look around, then take another step. Repeat until you can't go down anymore.

2. **`[w, b]`** — we're telling the optimiser which weights it's allowed to turn. In bigger models this list might have a billion entries; here we have two.

3. **`lr=0.01`** — the **learning rate**. This is the SIZE of each step. 0.01 means "take a small step." Try `lr=10` and the model will overshoot the answer like a drunk person trying to walk in a straight line. Try `lr=0.0000001` and it'll take forever. Picking the right learning rate is one of the most important — and most annoying — parts of training real models.

**Mental model:** the optimiser is a coach. Every step you tell it "here's how wrong I was", and the coach pushes the dials a little in the better direction.

---

### Line 39: a sanity check

```python
print(f"Before training:  y ≈ {w.item():.2f}*x + {b.item():.2f}")
```

**What it does:** prints the model's current "rule" before any training happens. It says `y ≈ 0.00*x + 0.00`. Useless — but useful to print so we can compare it to the trained version at the end.

**Tiny Python detail:** `w.item()` pulls the actual number out of a tensor (tensors can hold lots of numbers; `.item()` works when there's exactly one). The `:.2f` is just formatting: "show 2 decimal places."

---

### Line 41: the training loop begins

```python
for step in range(1000):
```

We're going to do the next four steps **one thousand times**. Each repetition nudges the weights a tiny bit. After 1000 nudges, the model is usually fully trained on this kind of problem.

---

### Line 42: the model makes a guess

```python
y_pred = w * x + b
```

**What it does:** for each of the 8 numbers in `x`, calculate what the model would predict using its current rule.

The first time through the loop, with `w=0` and `b=0`, this gives `y_pred = [0, 0, 0, 0, 0, 0, 0, 0]`. Totally wrong! But that's fine — that's why we're about to measure the wrongness.

**Note:** because `x` is a tensor of 8 numbers, `w * x + b` is computed for all 8 in a single shot. This is one of the superpowers of tensors — math happens to the whole list at once. (In a real model this might be billions of numbers in parallel on a GPU.)

---

### Line 43: how wrong is the guess?

```python
loss = ((y_pred - y_true) ** 2).mean()
```

**What it does:** computes a single number, called the *loss*, that summarises how wrong the model is.

Let's decompose it:

| Step | What it computes |
|---|---|
| `y_pred - y_true` | For each example, how far off was the guess? (Some positive, some negative.) |
| `** 2` | Square it. This makes all errors positive AND penalises big mistakes more than small ones. |
| `.mean()` | Average all 8 squared errors into ONE number. |

This is called **Mean Squared Error**, or MSE. It's the most common loss function for "predict a number" tasks.

**Why we need a single number:** the optimiser needs ONE thing to minimise. "Make this number go down." Without collapsing the 8 errors into one, the question "are we doing better?" is ambiguous.

---

### Lines 44–46: the three lines that ARE machine learning

```python
opt.zero_grad()
loss.backward()
opt.step()
```

If you only memorise three lines of this entire course, memorise these three. They appear in every deep learning model ever written, including PRAGMA.

**Line 44: `opt.zero_grad()`**
"Clear last step's notes." PyTorch keeps track of which way to nudge `w` and `b` in something called `.grad` on each tensor. If we don't clear those notes, they keep accumulating across steps — which makes the math wrong. So: zero them out at the start of every step.

**Line 45: `loss.backward()`**
This is the most magical line in the whole field. PyTorch looks at the chain of math we did to compute `loss`, runs it *backwards*, and computes:

- "If I increase `w` by a tiny amount, does the loss go up or down? By how much?"
- "If I increase `b` by a tiny amount, does the loss go up or down? By how much?"

These answers (the "gradients") get stored inside `w` and `b`. We didn't write any calculus. PyTorch did it for us, automatically, by tracing every `+`, `*`, `**`, and `.mean()` we used.

In a Transformer with a billion parameters, `loss.backward()` computes a billion gradients in one line. This is why GPUs were invented.

**Line 46: `opt.step()`**
"Take a step downhill." The optimiser reads the gradients that `loss.backward()` just computed, and updates the values:

```
w  ←  w  -  lr × (gradient of loss w.r.t. w)
```

That subtraction is the "descent" in "gradient descent" — we move *against* the gradient, because the gradient points uphill (toward more loss) and we want to go downhill (toward less loss).

After this line, `w` and `b` are slightly better than they were a microsecond ago. Repeat 1000 times.

---

### Lines 47–48: print progress every now and then

```python
if step % 200 == 0:
    print(f"  step {step:4d}   w={w.item():.3f}   b={b.item():.3f}   loss={loss.item():.4f}")
```

**What it does:** every 200 steps, print where we are. (`step % 200 == 0` is "step is a multiple of 200" — Python's modulo operator.)

Looking at the output:

```
  step    0   w=1.110   b=0.200   loss=121.0000
  step  200   w=2.050   b=0.721   loss=0.0162
  step  400   w=2.022   b=0.874   loss=0.0033
  step  600   w=2.010   b=0.944   loss=0.0007
  step  800   w=2.005   b=0.975   loss=0.0001
```

Notice the story this tells:
- **Step 0** (after just ONE training step): `w` already jumped from 0 to 1.11. The first step was a big one because the loss was huge (121), so the gradient was huge.
- **Step 200**: `w` is already at 2.05 — *basically the right answer* for the slope. `b` lags behind (0.72 vs the real 1.0). This is normal: different parameters often learn at different speeds.
- **Step 800**: loss is 0.0001 — essentially zero. The model has nailed it.

The loss dropped by a factor of about a million. That's not a typo. From 121 → 0.0001.

---

### Lines 50–51: the result

```python
print(f"After training:   y ≈ {w.item():.2f}*x + {b.item():.2f}")
print("(The secret rule was y = 2*x + 1. We re-discovered it.)")
```

**Final output:** `y ≈ 2.00*x + 0.99`. The true rule was `y = 2*x + 1`. Off by 0.01 on the intercept — close enough for government work.

**The big claim again:** the model was never told the rule. It was only given examples. It nudged its way to the answer by playing "guess, get told you're wrong, adjust" a thousand times.

This is **every machine learning model**, ever. PRAGMA's billion-parameter Transformer does *exactly* this. The "rule" it's discovering is unimaginably more complex ("given these 100 banking events, what's the next likely event?"), and it has a billion parameters instead of two — but the loop is the same.

---

## TL;DR

Three sentences. Memorise these:

> 1. **A model is a bag of numbers.**
> 2. **Training is nudging those numbers to make guesses more correct.**
> 3. **PyTorch computes which way to nudge them, automatically, for free, no matter how many numbers there are.**

That's it. Everything else in the course is decoration.

---

## 🪜 Optional next steps before Lesson 2

If you want to go deeper before moving on:

- **[Lesson 1b — Architecture vs. Training](01b_architecture_vs_training.md)** — separates two ideas that are easy to confuse: what the model computes vs. how its weights get tuned. Shows a side-by-side comparison of linear regression (2 parameters) and an embedding-based classifier (12 parameters), both trained with the same loop.
- **[Lesson 1c — Inside the training loop](01c_gradient_descent.md)** — opens up `loss.backward()` and `opt.step()`. Shows by hand how the gradient is computed, and includes an [interactive visualisation](visuals/gradient_descent_interactive.html) you can play with.

Both are optional. They're there in case "PyTorch handles it" felt too hand-wavy.
