# Mini Transformer Course

A six-lesson course that teaches how the PRAGMA model (and Transformers in
general) work, starting from zero. Each lesson is a small runnable program
with a companion **line-by-line walkthrough** explaining every concept in
plain English.

By the end you'll understand every line of `../pragma_mini.py` and be ready
to build your own version from scratch in the capstone.

## Lessons

| # | Code | Walkthrough |
|---|------|-------------|
| 1 | [`01_how_models_learn.py`](01_how_models_learn.py) | [`01_walkthrough.md`](01_walkthrough.md) — what "learning" actually means |
| 1b | [`aside/model_A_linear.py`](aside/model_A_linear.py)<br>[`aside/model_B_embedding.py`](aside/model_B_embedding.py) | [`01b_architecture_vs_training.md`](01b_architecture_vs_training.md) — architecture vs. training (the key distinction) |
| 1c | [`aside/model_A_with_gradients.py`](aside/model_A_with_gradients.py) | [`01c_gradient_descent.md`](01c_gradient_descent.md) — inside the training loop: loss and gradient descent (with an [interactive visualisation](visuals/gradient_descent_interactive.html)) |
| 2 | [`02_tokens_and_embeddings.py`](02_tokens_and_embeddings.py) | [`02_walkthrough.md`](02_walkthrough.md) — how text becomes numbers |
| 3 | [`03_attention.py`](03_attention.py) | [`03_walkthrough.md`](03_walkthrough.md) — the heart of the Transformer |
| 3b | — | [`03b_why_transformers_won.md`](03b_why_transformers_won.md) — why attention beat RNNs (speed, memory, scale) |
| 4 | [`04_transformer_and_mlm.py`](04_transformer_and_mlm.py) | [`04_walkthrough.md`](04_walkthrough.md) — putting it all together with the fill-in-the-blank training game |
| 5 | [`../pragma_mini.py`](../pragma_mini.py) | [`05_putting_it_together.md`](05_putting_it_together.md) — line-by-line walkthrough of `pragma_mini.py`, plus a deeper look at what real PRAGMA adds (key-value tokenisation, encoder depth, …) |
| 5b | [`05b_streaming_churn.py`](05b_streaming_churn.py) | [`05b_walkthrough.md`](05b_walkthrough.md) — same recipe applied to a realistic problem: streaming-service churn prediction (with the full pre-train → freeze → probe pipeline) |
| 6 | — | [`06_capstone.md`](06_capstone.md) — build your own mini fraud detector |

> **Lessons 1b, 1c, and 3b are optional but recommended.** They build the
> mental model that makes the core lessons stick. Skip them on a first read
> and come back when something feels hand-wavy.

## How to use the course

For each numbered lesson:

1. **Read** the `.py` file top to bottom. Don't worry if some lines are unclear.
2. **Run it**: `python3 0X_lesson.py`. See the output.
3. **Read** the matching `_walkthrough.md` for a careful, line-by-line explanation.
4. **Re-run** the `.py` and now read it again — every line should make sense.
5. **Try the exercises** at the bottom of the `.py` file. Edit, run, observe.

Then move on to the next lesson.

## How to run a lesson

```bash
cd course
python3 01_how_models_learn.py
```

## What you should already know

- Basic Python (variables, functions, lists, loops).
- That's it. No math, no machine learning needed.

## What you'll learn

- How a model "learns" by nudging its numbers.
- How text (or any data) becomes vectors.
- What attention is and why it matters.
- How a Transformer is built out of these pieces.
- How the BERT/PRAGMA fill-in-the-blank training game works.
- How to take a pre-trained model and use it for a new task — the foundation-model idea.
