# Mini Transformer Course

A six-lesson course that teaches how the PRAGMA model (and Transformers in
general) work, starting from zero. Each lesson is a small runnable program
with a companion **line-by-line walkthrough** explaining every concept in
plain English.

By the end you'll understand every line of `../pragma_mini.py` and be ready
to build your own version from scratch in the capstone.

## Lessons (main track)

| # | Code | Walkthrough |
|---|------|-------------|
| 1 | [`01_how_models_learn.py`](01_how_models_learn.py) | [`01_walkthrough.md`](01_walkthrough.md) — what "learning" actually means |
| 1b | [`aside/model_A_linear.py`](aside/model_A_linear.py)<br>[`aside/model_B_embedding.py`](aside/model_B_embedding.py) | [`01b_architecture_vs_training.md`](01b_architecture_vs_training.md) — architecture vs. training (the key distinction) |
| 1c | [`aside/model_A_with_gradients.py`](aside/model_A_with_gradients.py) | [`01c_gradient_descent.md`](01c_gradient_descent.md) — inside the training loop: loss and gradient descent (with an [interactive visualisation](visuals/gradient_descent_interactive.html)) |
| 2 | [`02_tokens_and_embeddings.py`](02_tokens_and_embeddings.py) | [`02_walkthrough.md`](02_walkthrough.md) — how text becomes numbers |
| 3 | [`03_attention.py`](03_attention.py) | [`03_walkthrough.md`](03_walkthrough.md) — the heart of the Transformer |
| 3b | — | [`03b_why_transformers_won.md`](03b_why_transformers_won.md) — why attention beat RNNs (speed, memory, scale) |
| 4 | [`04_transformer_and_mlm.py`](04_transformer_and_mlm.py) | [`04_walkthrough.md`](04_walkthrough.md) — putting it all together with the fill-in-the-blank training game |
| 5 | [`../pragma_mini.py`](../pragma_mini.py)<br>[notebook 📓](notebooks/lesson_05_pragma_mini.ipynb) | [`05_putting_it_together.md`](05_putting_it_together.md) — line-by-line walkthrough of `pragma_mini.py`, plus a deeper look at what real PRAGMA adds (key-value tokenisation, encoder depth, …) |
| 5a | [`05a_house_predictor.py`](05a_house_predictor.py)<br>[notebook 📓](notebooks/lesson_05a_house_predictor.ipynb) | [`05a_walkthrough.md`](05a_walkthrough.md) — same recipe on **tabular data with realistic correlations**. 10 attributes per house, 5 price classes. Bridge between the toy and streaming. |
| 5b | [`05b_streaming_churn.py`](05b_streaming_churn.py)<br>[notebook 📓](notebooks/lesson_05b_streaming_churn.ipynb) | [`05b_walkthrough.md`](05b_walkthrough.md) — same recipe applied to a realistic problem: streaming-service churn prediction (with the full pre-train → freeze → probe pipeline) |
| 6 | — | [`06_capstone.md`](06_capstone.md) — build your own mini fraud detector |

> **Lessons 1b, 1c, and 3b are optional but recommended.** They build the
> mental model that makes the core lessons stick. Skip them on a first read
> and come back when something feels hand-wavy.

## 🛠 Karpathy track (first-principles, no black boxes)

Inspired by Andrej Karpathy's "Zero to Hero" series. These lessons either
**re-implement PyTorch primitives from scratch** (so you see exactly what
nn.Embedding, attention, and TransformerEncoderLayer are doing), or
**build PRAGMA up incrementally** (so you watch the architecture grow
piece by piece on the same dataset).

### Internals — re-implement primitives from scratch

| # | What it re-implements | Why |
|---|---|---|
| 1d | [`01d_autograd_from_scratch.py`](01d_autograd_from_scratch.py) | A 100-line autograd engine (micrograd-style). Trains Lesson 1's regression with NO PyTorch autograd. Matches PyTorch's gradients exactly. |
| 2c | [`02c_embedding_from_scratch.py`](02c_embedding_from_scratch.py) | `nn.Embedding` in 4 lines. Verifies bit-exact outputs and gradients vs PyTorch. |
| 3c | [`03c_attention_from_scratch.py`](03c_attention_from_scratch.py) | Multi-head self-attention from primitives. Numerically identical to `nn.MultiheadAttention`. |
| 4e | [`04e_encoder_layer_from_scratch.py`](04e_encoder_layer_from_scratch.py) | `nn.TransformerEncoderLayer` decomposed: LayerNorm, FFN, MHA, residuals. Numerically identical to PyTorch's version. |

### Incremental build-up — same data, growing model

The same pets MLM task across four lessons. Each lesson adds one piece and
re-evaluates on the same test set. You watch loss go down (or, in one
case, up — and learn why).

| # | What it adds | Result |
|---|---|---|
| 4a | [`04a_counts_model.py`](04a_counts_model.py) — no neural net at all | 63.4% acc, 0.670 CE |
| 4b | [`04b_embedding_linear.py`](04b_embedding_linear.py) — embedding + mean-pool + linear | 64.2% acc, 0.558 CE |
| 4c | [`04c_with_attention.py`](04c_with_attention.py) — add self-attention | **53.6% acc, 0.902 CE (gets WORSE)** |
| 4d | [`04d_full_block.py`](04d_full_block.py) — full Transformer block | **64.5% acc, 0.533 CE (best)** |

The L4c regression is intentional — it shows attention alone isn't enough.
The fix (residuals + FFN + LayerNorm) is exactly what makes L4d win.

## 🌐 Interactive HTML version

Every lesson is also published as a self-contained interactive HTML page.
Start at **[`html/index.html`](html/index.html)** and click through. Lessons 1,
2, 3, 3b, and 4 have hands-on interactive widgets (sliders, draggable
embeddings, live attention matrices, animated RNN vs Transformer).

## How to use the course

For each numbered lesson:

1. **Read** the `.py` file top to bottom. Don't worry if some lines are unclear.
2. **Run it**: `python3 0X_lesson.py`. See the output.
3. **Read** the matching `_walkthrough.md` for a careful, line-by-line explanation.
4. **Re-run** the `.py` and now read it again — every line should make sense.
5. **Try the exercises** at the bottom of the `.py` file. Edit, run, observe.

Then move on to the next lesson.

## How to run a lesson

Three ways:

```bash
# 1. Run a script straight through
cd course
python3 01_how_models_learn.py
```

```bash
# 2. Open a Jupyter notebook (Lessons 5 and 5b)
pip3 install jupyter   # if you don't have it
jupyter notebook course/notebooks/
# then click into lesson_05_pragma_mini.ipynb and run cells one by one
```

```bash
# 3. Just READ on GitHub
# Both the .py files and the .ipynb notebooks render directly on GitHub —
# you can read the line-by-line explanations without installing anything.
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
