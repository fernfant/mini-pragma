"""
Model B — Tiny neural net with an embedding table.

ARCHITECTURE:    word_id  ──►  embedding lookup (2 numbers)  ──►  linear  ──►  3 scores
  PARAMS: 12  (6 weights in the embedding table + 6 in the linear layer)

This model learns to predict an animal's sound from its name:
  dog  → bark
  cat  → meow
  fish → swim

The SAME 5-line training loop as Model A — just nudging more parameters.
Watch the embedding table evolve from random to meaningful.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

ANIMALS = ["dog", "cat", "fish"]
SOUNDS  = ["bark", "meow", "swim"]

# Training data: input = animal id, target = sound id
inputs  = torch.tensor([0, 1, 2])    # dog, cat, fish
targets = torch.tensor([0, 1, 2])    # bark, meow, swim

# THE ARCHITECTURE: embedding lookup + linear projection
emb  = nn.Embedding(3, 2)            # 3 animals × 2 dims = 6 weights
head = nn.Linear(2, 3, bias=False)   # 2 × 3 = 6 weights

# THE TRAINING RECIPE: identical to Model A
all_params = list(emb.parameters()) + list(head.parameters())
opt = torch.optim.SGD(all_params, lr=0.5)
loss_fn = nn.CrossEntropyLoss()

print("=" * 60)
print("MODEL B — Embedding lookup + linear classifier")
print("  Architecture: id ─► embedding (2 nums) ─► linear ─► 3 scores")
print(f"  Total params:  {sum(k.numel() for k in all_params)}  (6 in embedding + 6 in head)")
print("=" * 60)
print("\nINITIAL EMBEDDING TABLE (random numbers — meaningless):")
for i, animal in enumerate(ANIMALS):
    print(f"  {animal:5s} -> {emb.weight[i].detach().tolist()}")

print(f"\n{'step':>5} | {'dog vec':>16} | {'cat vec':>16} | {'fish vec':>16} | {'loss':>6}")
print("-" * 80)
for step in range(401):
    logits = head(emb(inputs))                # 1. guess scores for each sound
    loss = loss_fn(logits, targets)           # 2. measure wrongness
    opt.zero_grad()                           # 3. clear notes
    loss.backward()                           # 4. compute gradients for ALL 12 parameters
    opt.step()                                # 5. nudge them all
    if step % 80 == 0:
        d = emb.weight[0].detach().tolist()
        c = emb.weight[1].detach().tolist()
        f = emb.weight[2].detach().tolist()
        print(f"{step:>5} | [{d[0]:+.2f}, {d[1]:+.2f}]    | [{c[0]:+.2f}, {c[1]:+.2f}]    | [{f[0]:+.2f}, {f[1]:+.2f}]    | {loss.item():>6.3f}")

print("\nFINAL EMBEDDING TABLE (numbers have evolved to encode meaning):")
for i, animal in enumerate(ANIMALS):
    print(f"  {animal:5s} -> {emb.weight[i].detach().tolist()}")

# Verify the model learned correctly
with torch.no_grad():
    preds = head(emb(inputs)).argmax(-1)
print("\nPredictions:")
for i, animal in enumerate(ANIMALS):
    predicted_sound = SOUNDS[preds[i].item()]
    correct = "✓" if predicted_sound == SOUNDS[i] else "✗"
    print(f"  {animal:5s} -> {predicted_sound:5s}  {correct}")
