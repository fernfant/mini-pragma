# Lesson 5a — House price predictor (walkthrough)

Companion to [`05a_house_predictor.py`](05a_house_predictor.py).

We're going to apply the PRAGMA recipe to houses. **Step by step. Slow.** First we look at the data, then build intuition for how the model sees it, then run two experiments and compare.

## 🧰 Lesson reference legend

- **L1** — the 5-line training loop
- **L1b** — architecture vs. training
- **L1c** — gradient descent details
- **L2** — tokens & embeddings
- **L3** — attention
- **L4** — masked language modelling
- **L5** — `pragma_mini.py`

---

## STEP 1 — Look at the data (build your eye first)

Before any model, just look at three houses. Each house has 10 attributes.

| Attribute      | 🏡 Rural cottage  | 🏘 Suburban family | 🏖 Luxury beach |
|----------------|------------------:|-------------------:|---------------:|
| bedrooms       | 2bed              | 3bed               | 5+bed          |
| bathrooms      | 1bath             | 2bath              | 3+bath         |
| size           | small             | medium             | **huge**       |
| age            | **vintage**       | modern             | new            |
| neighborhood   | rural             | suburb             | **beach**      |
| garage         | 1car              | 2car               | 2car           |
| pool           | nopool            | nopool             | **haspool**    |
| garden         | largegarden       | smallgarden        | largegarden    |
| schools        | avgschool         | goodschool         | excschool      |
| condition      | faircond          | goodcond           | exccond        |
| **price**      | **$280k (cheap)** | **$580k (avg)**    | **$1.4M (luxury)** |

> **What do you notice?** Look at the columns top-to-bottom.
>
> - The **luxury beach** house has BIG values everywhere — huge size, beach location, pool, excellent everything. They go together.
> - The **rural cottage** has SMALL values everywhere — small size, rural, no pool, vintage, fair condition. They go together too.
> - The **suburban family** sits in the middle.

This is **correlation**: the attributes aren't independent. Knowing a house is on the *beach* tells you it probably *has a pool*. Knowing it's *rural* tells you it probably has *no pool*.

> 🔑 **The whole reason MLM (Lesson 4) works here:** correlations like these are what the fill-in-the-blank game can learn. Without them, the attributes would be unrelated and there'd be nothing context-dependent to predict.

---

## STEP 2 — A wider view: 10 actual rows from the dataset

Now zoom out from the 3 archetypal examples and look at 10 actual rows from the full 8,000-house dataset. Same data shown two ways:

### (a) As pre-training sees them — features only, NO labels

```
  row | bedrooms  bathrooms size      age       neighborh garage    pool      garden    schools   condition
  ---------------------------------------------------------------------------------------------------------
    1 | 1bed      2bath     small     older     downtown  nogarage  nopool    smGarden  avgSch    faircond
    2 | 2bed      2bath     medium    vintage   suburb    1car      nopool    nogarden  avgSch    exccond
    3 | 2bed      2bath     small     vintage   downtown  nogarage  nopool    nogarden  avgSch    goodcond
    4 | 3bed      2bath     large     modern    suburb    nogarage  haspool   lgGarden  excSch    exccond
    5 | 3bed      3+bath    medium    older     rural     2car      nopool    lgGarden  goodSch   goodcond
    6 | 3bed      3+bath    medium    new       downtown  nogarage  nopool    smGarden  avgSch    goodcond
    7 | 4bed      3+bath    large     new       beach     1car      haspool   lgGarden  excSch    exccond
    8 | 2bed      2bath     medium    older     rural     2car      nopool    nogarden  goodSch   poorcond
    9 | 5+bed     2bath     large     new       beach     2car      haspool   lgGarden  goodSch   exccond
   10 | 1bed      3+bath    small     older     rural     2car      haspool   smGarden  avgSch    exccond
```

> Pre-training plays fill-in-the-blank on these features alone. **Uses ALL 8,000 houses. No labels needed.**

### (b) As the downstream task sees them — same rows, with price class

```
  row | bedrooms  bathrooms size      age       neighborh garage    pool      garden    schools   condition | price class
  -----------------------------------------------------------------------------------------------------------------------
    1 | 1bed      2bath     small     older     downtown  nogarage  nopool    smGarden  avgSch    faircond  | cheap
    2 | 2bed      2bath     medium    vintage   suburb    1car      nopool    nogarden  avgSch    exccond   | average
    3 | 2bed      2bath     small     vintage   downtown  nogarage  nopool    nogarden  avgSch    goodcond  | cheap
    4 | 3bed      2bath     large     modern    suburb    nogarage  haspool   lgGarden  excSch    exccond   | expensive
    5 | 3bed      3+bath    medium    older     rural     2car      nopool    lgGarden  goodSch   goodcond  | average
    6 | 3bed      3+bath    medium    new       downtown  nogarage  nopool    smGarden  avgSch    goodcond  | average
    7 | 4bed      3+bath    large     new       beach     1car      haspool   lgGarden  excSch    exccond   | luxury
    8 | 2bed      2bath     medium    older     rural     2car      nopool    nogarden  goodSch   poorcond  | cheap
    9 | 5+bed     2bath     large     new       beach     2car      haspool   lgGarden  goodSch   exccond   | luxury
   10 | 1bed      3+bath    small     older     rural     2car      haspool   smGarden  avgSch    exccond   | cheap
```

> Downstream task uses the full row, **including the price class**. Uses only a small subset of houses (50, 100, 500, or 4000).

**Read the rows carefully:**

- Row 7: 4-bed, large, new, beach, pool, excellent schools → **luxury**. Of course.
- Row 9: 5+bed, large, new, beach, pool → **luxury** again. Beach + pool + size = luxury, every time.
- Row 1: 1-bed, small, downtown, no pool, no garage → **cheap**. Tiny urban apartment.
- Row 8: 2-bed, medium, rural, poor condition → **cheap**. Rural fixer-upper.

You can practically read the price class by eye. **That's the signal the model has to learn from the token sequences alone.**

---

## STEP 3 — How the computer sees each house

The computer doesn't see those nice rows. It sees a **flat list of token IDs** (L2 + L5):

```
event:  [("bedrooms", "2bed"),
         ("bathrooms", "1bath"),
         ("size",  "small"),
         ...
         ("condition", "faircond")]

ids:    [bedrooms_id, 2bed_id, bathrooms_id, 1bath_id, size_id, small_id, ...]
        ↳ 20 token IDs total (10 keys × 2 each)
```

That's it. The model never sees "rural cottage" as a label. Just a 20-element list of integers. Its whole job is to find structure in those integer sequences.

---

## STEP 4 — The TWO games the model plays

> **This is the part that needs to be crystal clear.** Pre-training and the downstream task are **two separate training sessions** on the same data.

### Game 1: Pre-training (a.k.a. self-supervised learning)

```
Task:  fill-in-the-blank on house attributes
       (mask some values, predict them)

Data:  ALL 8,000 houses
Labels needed?  NO — the data labels itself
                (we just hide tokens we already have)
Goal:  teach the encoder the correlation structure
```

This is the L4 game from earlier. The model never sees prices here.

### Game 2: Downstream task (the thing we actually care about)

```
Task:  predict price class (bargain / cheap / avg / expensive / luxury)
Data:  a SMALL labelled set (50, 100, 500, or 4000 houses)
Labels needed?  YES — we need the price for each training house
Goal:  classify houses by price
```

> 🔑 **The big PRAGMA bet:** if Game 1 has taught the encoder something useful about house structure, Game 2 can be done with very few labels — because the encoder already knows the patterns. We just need to teach a tiny "translator" layer (the price head) that maps encoder output → price class.

---

## STEP 5 — Pre-training, walked through with one concrete house

Take the rural cottage from Step 1. Encode it as token IDs. Now randomly mask ~30% of the value tokens (never key tokens — those are the "prompt" telling the model what KIND of thing to predict):

```
ORIGINAL:
  bedrooms→2bed  bathrooms→1bath  size→small  age→vintage  neighborhood→rural
  garage→1car    pool→nopool      garden→largegarden       schools→avgschool
  condition→faircond

AFTER MASKING (random 30%):
  bedrooms→2bed  bathrooms→1bath  size→<MASK>  age→vintage  neighborhood→rural
  garage→1car    pool→<MASK>      garden→largegarden       schools→avgschool
  condition→<MASK>
```

The model has to predict the 3 masked values. It looks at the visible context — *rural, vintage, 1car, largegarden, avgschool* — and uses **attention** (L3) to figure out:

- `size→<MASK>` — context says rural cottage, so probably `small` or `medium`.
- `pool→<MASK>` — rural + vintage + avgschool = probably `nopool`.
- `condition→<MASK>` — vintage + fair-ish profile = probably `faircond`.

The model's guesses (random at first) get compared to the actual masked values. The loss is high → backprop → nudge knobs → guesses get a little better next time. Run this 3,000 times across random batches and the model learns:

- "beach" embeddings drift close to "haspool", "largegarden", "exccond"
- "rural" embeddings drift close to "small", "vintage", "largegarden"
- "downtown" embeddings drift close to "small", "nogarden", "nopool"

The encoder has now learned the **archetype structure** without ever being told there are archetypes.

---

## STEP 6 — How training works in this lesson (link back to L1, L1c)

Same 5-line loop as Lesson 1, just with thousands more knobs to nudge:

```python
for step in range(3000):
    xb, yb = mlm_mask(X[idx])                   # take a batch, mask some values
    logits = mlm_head(encoder(xb))              # 1. guess
    loss   = loss_fn(logits, yb)                # 2. measure wrongness
    opt.zero_grad()                             # 3. clear notes
    loss.backward()                             # 4. compute gradients for ALL knobs
    opt.step()                                  # 5. nudge them
```

`loss.backward()` is the magic — it computes gradients for every embedding number, every position embedding, every Q/K/V matrix in attention, every feed-forward weight, and the MLM head — all in one line. We saw exactly how this works in [L1c](01c_gradient_descent.md).

After 3,000 steps, the encoder has rich learned representations. **We then freeze it** and move to Game 2.

---

## STEP 7 — The downstream task: predict price class

> 🧰 **Concept bridge — pieces and their origin**
>
> The encoder used in this lesson is exactly the one from `pragma_mini.py`:
>
> | Component | Origin |
> |---|---|
> | Encoder (Embedding + position + Transformer × 2) | **[L5 / `pragma_mini.py`](05_putting_it_together.md)** — same backbone |
> | The pre-training recipe (mask + cross-entropy) | **[L4](04_walkthrough.md)** — masked language modelling |
> | `PriceHead` (mean-pool + Linear) | **[L1](01_walkthrough.md)** — linear projection |
> | The "freeze + probe" downstream pattern | **[PRAGMA paper §3.1.1](../Pragna_Model.pdf)** — embedding probe |
> | The 5-line training loop | **[L1](01_walkthrough.md)** |

Stack a tiny new layer on top of the (frozen) encoder:

```python
class PriceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 5)   # 32-d encoder output → 5 price classes
    def forward(self, h):
        pooled = h.mean(dim=1)          # average across positions
        return self.proj(pooled)
```

Then train *only* this head on labelled houses, using the same 5-line loop again. The encoder is frozen — `opt` only knows about the head's parameters.

That's the full **embedding probe** recipe from PRAGMA §3.1.1.

---

## STEP 8 — The big experiment (PRE-TRAINING vs BASELINE, defined clearly)

This is the comparison the lesson is built around. **It's not "pre-trained encoder vs no encoder"** — both have an encoder. The difference is what happens BEFORE we train the price head.

```
  PRE-TRAINING RECIPE                       BASELINE RECIPE
  ─────────────────────                     ───────────────────
                                            (skip pre-training entirely)
  Step 1: Train encoder via MLM
          on all 8,000 houses
          (no price labels needed)
                                            Step 1: Random-init encoder
                                                    (knobs start as noise)

  Step 2: FREEZE encoder
          (don't nudge its knobs anymore)

  Step 3: Add price head                    Step 2: Add price head

  Step 4: Train price head ONLY             Step 3: Train EVERYTHING
          on N labelled houses                       (encoder + head together)
                                                     on N labelled houses
```

**Why we compare them.** We want to know: *does the pre-training step actually help?* If the baseline (skip pre-training, train end-to-end on labels) does just as well, then pre-training was wasted effort. If the pre-trained recipe wins — especially when labels are scarce — that's the whole foundation-model pitch validated.

Both recipes use the same data, the same architecture, and the same downstream training loop. The only difference is whether we do the self-supervised pre-training step first.

---

## STEP 9 — Results, interpreted

```
 labels | pretrained acc  pretrained recall | baseline acc  baseline recall
─────────────────────────────────────────────────────────────────────────────
     50 |       0.66 ✓              0.64    |      0.59           0.62
    100 |       0.70 ✓              0.59    |      0.67           0.60
    500 |       0.72                0.61    |      0.79 ✓         0.76 ✓
   4000 |       0.73                0.61    |      0.86 ✓         0.84 ✓
```

### What the numbers say

- **50 labels**: pre-training wins by 7 accuracy points. With so few labels, training the encoder from scratch can't find the price-relevant patterns. The pre-trained encoder *already knows* the house structure — the price head just has to learn the small mapping from encoder output → price class.
- **100 labels**: pre-training still ahead, smaller margin.
- **500 labels**: **baseline overtakes**. With more labels, training end-to-end can find encoder representations specifically optimised for price prediction. The frozen pre-trained encoder is stuck with generic representations.
- **4000 labels**: baseline wins clearly. End-to-end has enough signal to dominate.

### Why isn't pre-training a clear winner everywhere?

Two reasons specific to this lesson:

1. **Tabular data has limited contextual richness.** Yes, attributes are correlated — but only 10 of them. Compare to streaming data in L5b: 15 events × 4 attributes = 60 correlated pieces of info, plus temporal ordering. More structure = more for pre-training to learn.
2. **A frozen encoder is a hard constraint.** It can't be tuned for the price task. Once we hit ~500 labels, the supervised signal is rich enough that the baseline's flexibility wins.

**The real-world fix:** instead of a fully frozen probe, **LoRA fine-tuning** (PRAGMA §3.1.2). Unfreeze a small fraction (~2-4%) of the encoder's weights during downstream training. You get pre-training's warm start AND the ability to tune for the task. Best of both worlds. We'll see this in the capstone.

---

## STEP 10 — What you should walk away with

1. **The PRAGMA recipe (pre-train → freeze → probe) is most valuable when labels are scarce AND the data has rich contextual structure.**
2. **Pre-training and downstream are two separate training sessions on the same data.** Different goals, different losses, different things being nudged.
3. **The "baseline" comparison is what proves pre-training mattered.** Without it, you can't tell if your shiny pre-trained model is actually helping vs just doing what end-to-end training would have done anyway.
4. **The recipe doesn't ALWAYS win.** As you'll see in L5b, with richer sequential data it wins much more decisively — because there's much more structure for MLM to capture.

---

## Code walkthrough (line by line, for reference)

If you want to dig into the code in detail:

### Section A — Vocabulary <span>L2</span> <span>L5</span>

```python
KEYS = ["bedrooms", "bathrooms", "size", "age", "neighborhood",
        "garage", "pool", "garden", "schools", "condition"]
VALUE_BUCKETS = {...}
vocab  = [PAD, MASK] + KEYS + VALUES
tok2id = {t: i for i, t in enumerate(vocab)}
V      = len(vocab)
```

Standard key-value vocab. 10 keys + ~36 values + 2 special tokens = 48 token IDs.

### Section B — Archetypes (the correlation structure)

```python
ARCHETYPES = {
    "rural_cottage":   {"size": {"small":5, "medium":4, "large":1, "huge":0}, ...},
    "luxury_beach":    {"size": {"small":0, "medium":1, "large":4, "huge":5}, ...},
    "urban_apartment": {...},
    "suburban_family": {...},
    "old_townhouse":   {...},
}
```

Each archetype is a weighted distribution over attribute values. Generating a house = pick an archetype, then sample each attribute from its distribution. This is what creates the correlations.

### Section C — Architecture <span>L1b</span> <span>L2</span> <span>L3</span>

```python
class Encoder(nn.Module):
    def __init__(self, V, d=32, heads=2, layers=2):
        self.emb = nn.Embedding(V, d)             # L2
        self.pos = nn.Embedding(64, d)
        layer    = nn.TransformerEncoderLayer(d, heads, d*2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)   # L3, stacked
    def forward(self, x):
        pos = torch.arange(x.size(1))
        return self.enc(self.emb(x) + self.pos(pos))
```

Same backbone as `pragma_mini.py`. Embedding + position + 2 attention layers.

### Section D — Pre-training loop <span>L1</span> <span>L1c</span> <span>L4</span>

```python
for step in range(3000):
    idx       = torch.randint(0, N_HOUSES, (128,))
    xb, yb    = mlm_mask(X[idx])                  # 30% masking
    logits    = mlm_head(encoder(xb))
    loss      = loss_fn(logits.reshape(-1, V), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
```

The 5-line loop. Nothing new — but the model is now learning house correlation structure.

### Section E — Downstream comparison

```python
for n_labels in [50, 100, 500, 4000]:
    enc_a = copy.deepcopy(pretrained_encoder)
    acc_a, rec_a, _ = train_classifier(enc_a, X_tr, y_tr, freeze_encoder=True)
    enc_b = Encoder(V)
    acc_b, rec_b, _ = train_classifier(enc_b, X_tr, y_tr, freeze_encoder=False)
```

Two recipes, side by side. `freeze_encoder=True` is the pre-training recipe (Step 7 above, left column). `freeze_encoder=False` is the baseline (right column).

---

## Bridge to L5b

Same recipe, but with sequential event data — where pre-training wins much more decisively. Onward to [Lesson 5b](05b_walkthrough.md).
