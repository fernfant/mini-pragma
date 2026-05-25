# Mini PRAGMA — a hello-world Transformer for kids (and curious adults)

A small, runnable, kid-friendly recreation of [**PRAGMA**](https://arxiv.org/abs/2604.08649),
Revolut's foundation model for banking event streams — plus a complete
6-lesson course teaching how it works, from zero.

## What's in here

- **[`pragma_mini.py`](pragma_mini.py)** — the toy. A ~150-line PyTorch program
  that takes the PRAGMA recipe (BERT-style masked modelling on key/value event
  records) and applies it to a kid-friendly dataset: pets doing things in places.
  Trains in seconds on CPU.

- **[`course/`](course/)** — a 6-lesson mini-course that builds up to
  understanding `pragma_mini.py` (and the real paper). Each lesson is a runnable
  Python file plus a line-by-line walkthrough in plain English. The final
  lesson is a capstone project: a mini fraud detector built using the PRAGMA
  recipe.

- **[`course/html/`](course/html/)** — interactive HTML versions of every
  lesson with sliders, live attention matrices, animated comparisons, and a
  roll-the-ball gradient-descent demo. Open `course/html/index.html` in any
  browser.

- **[fernfant/mini-pragma-solutions](https://github.com/fernfant/mini-pragma-solutions)** —
  a separate repo with the reference solution to the capstone. Try the capstone
  yourself before peeking.

## Quick start

```bash
pip3 install torch
python3 pragma_mini.py
```

You should see the model train on synthetic pet events and then correctly fill
in blanks like `cat sleeping on the ____` → `couch`.

Then open [`course/README.md`](course/README.md) and start with Lesson 1.

## The story behind it

PRAGMA is a foundation model Revolut published in 2026. It's BERT-but-for-banking:
masked modelling on multi-source customer event streams (transactions, app
events, communications), tokenised as `(key, value, time)` triples. After
pre-training, the same model is reused — via embedding probe or LoRA — for
credit scoring, fraud detection, lifetime-value prediction, and more.

The big idea is the foundation-model idea: pre-train one model on lots of
unlabelled data, then adapt to many downstream tasks. PRAGMA shows this works
not just for text or images but for structured financial event streams too.

This repo is a hands-on way to understand the recipe.

## License

[MIT](LICENSE) — use it, fork it, remix it, teach with it.

## Author

Built by [@fernfant](https://github.com/fernfant) as a way to teach
machine-learning fundamentals to my son. If you use it to teach anyone in
your life, I'd love to hear about it.
