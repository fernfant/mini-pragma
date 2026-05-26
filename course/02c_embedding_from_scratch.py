"""
Lesson 2c — nn.Embedding from scratch.

The first lesson in the "no black boxes" track.

In Lesson 2 we used PyTorch's nn.Embedding as if it were magic. This file
shows that it's not. nn.Embedding is literally a 2-D weight matrix and an
indexing operation. We re-implement it from scratch in ~15 lines and prove
it gives the same forward result, the same gradients, and trains to the
same answer as PyTorch's built-in.

By the end you will know:
  - Exactly what nn.Embedding stores (a single weight tensor).
  - Why "looking up an ID" IS the forward pass — no math involved.
  - How gradients flow back through the lookup (index_add).
  - That every "layer" in PyTorch is just (parameters + forward function).

Run:  python3 02c_embedding_from_scratch.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


# ============================================================================
# STEP 1 — Look at nn.Embedding as a black box
# ============================================================================
# Here's what we used in Lesson 2:

V, d = 5, 3
ref = nn.Embedding(V, d)
print("=" * 70)
print("STEP 1 — nn.Embedding as PyTorch gives it to us")
print("=" * 70)
print(f"nn.Embedding({V}, {d}) — V words in vocab, each represented as a d={d}-dim vector")
print(f"\nWhat's inside? Let's look:")
for name, p in ref.named_parameters():
    print(f"  parameter '{name}' has shape {tuple(p.shape)}:")
    print(p.data)
print()

# Use it
ids = torch.tensor([0, 2, 4])
vectors = ref(ids)
print(f"Calling ref({ids.tolist()}) returns:")
print(vectors)
print(f"\nshape: {tuple(vectors.shape)}  (3 rows, each is a 3-dim vector)")
print()


# ============================================================================
# STEP 2 — Open the box. What is nn.Embedding actually doing?
# ============================================================================
# It is a SINGLE WEIGHT MATRIX of shape (V, d).
# When you call emb(ids), it returns weight[ids] — i.e., indexing.
# That's it. No math. Just memory lookup.

print("=" * 70)
print("STEP 2 — Prove it's just indexing")
print("=" * 70)
print("These two should be identical:")
print(f"  ref(ids):              {ref(ids).flatten().tolist()}")
print(f"  ref.weight[ids]:       {ref.weight[ids].flatten().tolist()}")
print(f"  ref.weight.data[ids]:  {ref.weight.data[ids].flatten().tolist()}")
print()
print("If your son writes `ref.weight[ids]`, he reimplemented nn.Embedding.")
print("That's the whole magic.\n")


# ============================================================================
# STEP 3 — Re-implement nn.Embedding ourselves
# ============================================================================
# An nn.Module is just two things:
#   1. Parameters (tensors that will be trained).
#   2. A `forward` method (what to do when called).

print("=" * 70)
print("STEP 3 — Our own Embedding class, from scratch")
print("=" * 70)


class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        # The ONE parameter: a (vocab_size, dim) matrix. Random init.
        # nn.Parameter tells PyTorch "this is trainable" — same as
        # requires_grad=True from Lesson 1, but for tensors inside a module.
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))

    def forward(self, ids):
        # The forward pass IS the lookup. That's it.
        return self.weight[ids]


# Make ours and copy the reference's weights so we can compare exactly
mine = MyEmbedding(V, d)
mine.weight.data.copy_(ref.weight.data)

print("Same input, same weights, same output:")
print(f"  ref(ids):   {ref(ids).flatten().tolist()}")
print(f"  mine(ids):  {mine(ids).flatten().tolist()}")
assert torch.allclose(ref(ids), mine(ids)), "outputs differ!"
print("\n  ✓ Outputs identical.\n")


# ============================================================================
# STEP 4 — What does the GRADIENT look like?
# ============================================================================
# Here's a great question: when we call loss.backward(), what gradient does
# nn.Embedding's weight receive?
#
# Answer: only the rows we LOOKED UP get a gradient. All other rows are
# untouched (their gradient is zero). Within the looked-up rows, the
# gradient is the SUM of the upstream gradient at every position where that
# row was looked up.
#
# Let's prove it.

print("=" * 70)
print("STEP 4 — How gradients flow through embedding lookup")
print("=" * 70)

# Set up a tiny training step on both versions, with identical weights
mine2 = MyEmbedding(V, d)
ref2  = nn.Embedding(V, d)
ref2.weight.data.copy_(mine2.weight.data)

ids = torch.tensor([0, 2, 2, 4])  # note: id=2 appears TWICE
# Use a dummy "loss" so we can call backward
out_mine = mine2(ids).sum()
out_ref  = ref2(ids).sum()

out_mine.backward()
out_ref.backward()

print("Lookup ids:", ids.tolist())
print("Note id=2 appears twice, so its gradient should be 2× the gradient at id=0 or id=4.\n")
print("Reference (nn.Embedding) gradient:")
print(ref2.weight.grad)
print("\nOur version gradient:")
print(mine2.weight.grad)
print("\nNotice:")
print("  - Rows 1 and 3 have gradient 0 (we never looked them up).")
print("  - Rows 0 and 4 have gradient [1, 1, 1] (looked up once, summing ones).")
print("  - Row 2 has gradient [2, 2, 2] (looked up TWICE, summed).")
assert torch.allclose(ref2.weight.grad, mine2.weight.grad), "gradients differ!"
print("\n  ✓ Gradients identical.\n")


# ============================================================================
# STEP 5 — End-to-end training comparison
# ============================================================================
# To be absolutely sure, let's train both versions on the same fake task
# (predict a label from a token ID) and verify they converge to the same
# weights.

print("=" * 70)
print("STEP 5 — Train both versions side by side on the same task")
print("=" * 70)


def make_dataset(V, n=200):
    """Each token id maps to a fixed target label (one of 2 classes).
    The model has to learn the mapping."""
    ids = torch.randint(0, V, (n,))
    labels = (ids % 2).long()
    return ids, labels


def train_one(emb_module, V, d=8, n_classes=2, steps=300):
    torch.manual_seed(123)
    ids, y = make_dataset(V, n=200)

    # Tiny model: embedding -> mean pool (trivial here) -> linear classifier
    head = nn.Linear(d, n_classes)
    opt = torch.optim.AdamW(list(emb_module.parameters()) + list(head.parameters()), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        h = emb_module(ids)
        logits = head(h)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), (head(emb_module(ids)).argmax(-1) == y).float().mean().item()


V_big, d_big = 20, 8
torch.manual_seed(42); ref_emb = nn.Embedding(V_big, d_big)
torch.manual_seed(42); mine_emb = MyEmbedding(V_big, d_big)

loss_ref,  acc_ref  = train_one(ref_emb,  V_big, d_big)
loss_mine, acc_mine = train_one(mine_emb, V_big, d_big)

print(f"nn.Embedding (PyTorch):  final loss {loss_ref:.4f}   accuracy {acc_ref*100:.1f}%")
print(f"MyEmbedding (ours):      final loss {loss_mine:.4f}   accuracy {acc_mine*100:.1f}%")
print()
print("Trained to the same (or near-identical) result. Our 4-line class")
print("does everything nn.Embedding does. The framework just gives us a")
print("better-named, slightly faster version with some bells and whistles")
print("(weight init schemes, padding_idx, etc.) — but the math is identical.")
print()


# ============================================================================
# STEP 6 — Things to try
# ============================================================================
# 1. In MyEmbedding, change the initialization from `torch.randn(...)` to
#    `torch.zeros(...)`. Re-run STEP 5. Does training still work? Why not?
#    (Hint: if every embedding is identical, the model can't distinguish
#     between tokens.)
#
# 2. Print `mine.weight.grad` right after one .backward() call with a single
#    repeated id (e.g., torch.tensor([3, 3, 3])). Predict what you'll see
#    before running it.
#
# 3. nn.Embedding has a `padding_idx` argument. When you set padding_idx=0,
#    the gradient for row 0 is forced to zero. Try implementing this in
#    MyEmbedding by zeroing the gradient yourself after backward().
