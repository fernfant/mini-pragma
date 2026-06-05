# `mini.py` — the one file that grows into a transformer

**The bet (Karpathy #1):** the course stops being 25 separate widgets and becomes
**one program the kid grows, line by line**, from `tip = 0.2 * bill` (L1) into a
working tiny transformer that fills in `dog → bark` (L4) — and then gets *re-pointed*
at bank transactions (L5) without changing the model. Same variable names, cradle
to grave. At the end they scroll to the top and the first line they ever wrote is
still there. *"I didn't read about a transformer. I built one, and I can run it."*

This doc is the **plan to react to**, not built code. But the keystone is
**verified working** (see §3) so the ladder isn't fiction.

---

## 1. Substrate decision: micrograd-style scalar autograd (pure Python)

One ~30-line `Value` class (Karpathy's micrograd): scalars that remember how they
were computed and can `.backward()`. Why this and not numpy/torch:

- **Pure Python, zero deps** → runs in the browser via the existing `.runpy`
  Pyodide cells, stays offline-capable, matches the course's substrate.
- **The gradient becomes visible** (this is Karpathy bet #3 for free): every
  number knows its `.grad`; the kid can drag a weight, call `.backward()`, and
  *see* which way training pushes it. No black-box `loss.backward()`.
- **Canonical**: this is literally the Zero-to-Hero arc (micrograd → makemore →
  GPT), compressed for a 13-year-old.

The whole course becomes: **build micrograd, then build a transformer on it.**

---

## 2. The growth ladder (what each spine step ADDS to the one file)

Variable names carry forward. Each rung is the *new* lines; everything above stays.

| Step | Lesson | New lines added to `mini.py` | Runs live? |
|---|---|---|---|
| 1 | L1 | `w=0.2; model=lambda x: w*x` + the nudge loop `w -= lr*err*x` | ✅ instant |
| 2 | L1.5 | reveal the `Value` autograd engine; `neuron = relu(w*x+b)`; `loss.backward()` | ✅ instant |
| 3 | L1.5b | `x` becomes a vector `[Value,…]`; `linear(x,W,b)` = dot per row | ✅ instant |
| 4 | L1.6 | `mlp(x)=linear(relu_all(linear(x,W1,b1)),W2,b2)` — "this block = the FFN" | ✅ instant |
| 5 | L2 | `vocab`, `tok`, `emb=[[Value]*D per word]`; `embed(word)=emb[tok[word]]` | ✅ instant |
| 6 | L2a | train `emb` so co-occurring words cluster (the CBOW objective) | ✅ ~1s (tiny) |
| 7 | L3 | `attention(X)`: score→softmax→mix (the exact L3 "bark" cell, generalised) | ✅ forward instant |
| 8 | L3a | add `Q,K,V` projections + `1/√d` scaling → real routing | ✅ forward instant |
| 9 | L4 | `pos` embeddings + `head` + `cross_entropy`; **the train loop** → learns `dog→bark` | ⚠️ see §4 |
| 10 | L4f | one rule change: causal mask + shifted labels → **generate** text | ✅ inference |
| 11 | L4h | no new lines — comment: "bigger `D`, more blocks, more data = GPT. Same file." | — |
| 12 | L5 | `import mini`; feed **key-value event tokens** instead of words | ✅ inference |
| 13 | L5a | re-point `mini` at house-price fields (tabular) | ✅ inference |
| 14 | L5b | re-point `mini` at churn events (streaming) | ✅ inference |
| 15 | L6 | run the whole `mini.py` top to bottom — "you built this" | ✅ |

**The core growing program is steps 1–10 (L1 → L4f): one transformer.**
Steps 12–14 are the *payoff* — the model never changes, only the tokens do. That's
the "same machine, new domain" reveal, made literal: `import mini`.

---

## 3. Keystone VERIFIED (the riskiest rung, proven before planning)

Built the micrograd engine + a tiny self-attention masked-LM on the real corpus
(`dog↔bark / cat↔meow / fish↔swim`) and trained it:

- Engine works; **gradients flow; it trains.** 72–80 params, 200–300 steps,
  **0.2–0.4 s native.**
- **Two real bugs surfaced by running it** (exactly why we verify first):
  1. With no **positional encoding**, it can't tell the "animal slot" from the
     "sound slot" → the two mask directions fight → collapses. *Fix: add `pos`
     embeddings (every real transformer has them).* → step 9 must include `pos`.
  2. Plain dot-product attention on raw embeddings under-fits (learned `dog→bark`
     but not `cat→meow`). *Fix: Q/K/V routing + an FFN + a bit more `D`* → which is
     why steps 8 (Q/K/V) and 4 (FFN) exist. The toy **needs the real pieces** to
     learn — a great honest lesson in itself.

**Net:** substrate confirmed for the readable artifact and small/forward cases;
the full *training* rung has a speed constraint (§4).

---

## 4. The one hard constraint + the speed strategy

Scalar autograd is perfect for **seeing** and for **tiny** training, but training
the *full* QKV+FFN block live to convergence is **seconds for the toy, likely
minutes once it's big enough to reliably learn all rules** — too slow for a cell a
kid waits on.

**Strategy, per step:**
- **Steps 1–8 + the cross-entropy-by-hand**: run **live** on micrograd. Small,
  fast, gradients visible. This is the bulk of the "build it yourself" magic and
  it all runs in <1 s.
- **Step 9 (train the MLM)** — the climax: **live-train a deliberately tiny case**
  (1–2 pairs, `D=4`, ~100 steps → a few seconds, real, bounded) so the kid watches
  the loss actually fall, **and ship the full 3-pair run's loss curve + final
  weights as data** so the "it learned all the rules" result is honest and instant.
- **Steps 10–14**: **inference only** on the shipped/just-trained weights → fast.
- The existing torch **notebook** remains the "train the big one for real" path.

This keeps every claim honest: what the kid runs live is real; what's too slow to
train live is shipped as real pre-computed numbers and clearly labelled.

---

## 5. How the existing course re-organises around it

`mini.py` becomes the **spine artifact**; today's pages become **windows** into its
current state:

- The `.runpy` cells already added (L2 similarity/train, L3 bark, L4 train) are the
  first windows — they get re-pointed at the shared `mini.py` so they're literally
  the same file at different stages.
- The JS "live sketch" playgrounds (badged last pass) get **replaced** by real
  `.runpy` windows over `mini.py` (Karpathy bet #2: one real substrate, no
  simulations).
- A persistent **"your `mini.py` so far"** panel could show the accumulated file,
  growing each lesson (ownership + continuity).
- Nav/structure unchanged — this is additive to the 15-step spine we just cleaned.

---

## 6. Open decisions for you

1. **Scope of the live-train climax (step 9):** tiny-case-live + shipped-full
   (recommended), or pre-trained-weights + "watch 20 steps" only?
2. **Replace the JS playgrounds now, or leave them badged and add `mini.py`
   windows alongside first?** (Lower risk to add alongside, then retire.)
3. **Where `mini.py` physically lives:** `course/mini.py` (importable by L5+) and
   surfaced read-only in pages, vs inlined per-cell. Recommend a real file +
   windows.
4. **Build order:** I'd build the *verified end-state* `mini.py` first (engine +
   tiny transformer that learns, ship its weights), then carve the rungs backward
   from it — so every lesson's window is guaranteed to match a working whole.

---

## 7. Verified artifacts backing this plan

- `/tmp/mini_proof.py`, `/tmp/mini_proof2.py` — the micrograd engine + tiny MLM
  used to confirm the substrate and surface the pos-encoding / QKV findings.
  (Throwaway; the real `course/mini.py` would be built clean per §6.4.)
