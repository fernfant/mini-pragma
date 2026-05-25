# Lesson 6 — Capstone: Mini Fraud Detector

You've now learned everything you need to build your own version of PRAGMA
from scratch, and use it for something real.

## The challenge

You work at a tiny digital bank. Each user generates a stream of events
(transactions, app actions, transfers). Most users are normal — but a few
are fraudsters who behave very differently.

You'll build a system using the exact PRAGMA recipe:

1. **Generate** synthetic event data for a few thousand users.
2. **Pre-train** a tiny Transformer on ALL events with fill-in-the-blank
   (the model never sees the labels at this stage — this is *self-supervised*).
3. **Freeze** the encoder. Add a tiny classifier head on top.
4. **Train** the classifier on a small labelled set (normal vs. fraud).
5. **Compare** to a baseline that doesn't pre-train.

This is the **exact same workflow** Revolut runs at billion-event scale, just
smaller. By the end you'll have hands-on proof of why foundation models work.

Target effort: 4–8 hours, spread over a few sessions. No new concepts —
only what you've learned in Lessons 1–5.

---

## Stage 1: Generate synthetic data

Create a file `capstone_data.py`. Decide on:

- **Vocabulary**. Some keys (e.g. `type`, `amount_bucket`, `time_of_day`) and
  values for each (e.g. `type ∈ {purchase, transfer, login, signup}`,
  `amount_bucket ∈ {tiny, small, medium, large}`, `time_of_day ∈ {morning,
  afternoon, evening, night}`).
- **Two user profiles**:
  - `NORMAL_RULES` — common patterns. E.g., mostly `purchase` events,
    mostly `small`/`medium` amounts, mostly `afternoon`/`evening`.
  - `FRAUD_RULES`  — different patterns. E.g., bursts of `tiny` test
    transactions late at `night`, very few `login` events, etc. Be creative.
- **2000 users total**: 95% normal, 5% fraudsters. Each user has 20 events.
- Save: a list of `(events, label)` where `label = 0` (normal) or `1` (fraud).

**Hint** — start from the `random_event()` function in `../pragma_mini.py`
and generalise it.

**Success check**: print a few sample sequences for a normal user and a
fraud user. Eyeball them — can you spot the difference? If not, make the
rules more distinct.

---

## Stage 2: Pre-train the encoder

Create `capstone_pretrain.py`.

Reuse the `PragmaMini` class (or write your own — your call). Run the same
masked-token training game from Lesson 4 / `pragma_mini.py`.

**Key rule**: during pre-training, **ignore the labels completely**.
The pre-training game cares nothing about whether a user is fraud — it's
just trying to learn the structure of events.

When training finishes, save the encoder weights with `torch.save(...)`.

**Success check**: pre-training loss should drop substantially (e.g., from
~3.0 to under 0.5). Pick a few random sequences, mask one token, and confirm
the predictions look reasonable.

---

## Stage 3: Build the fraud classifier

Create `capstone_classify.py`.

1. **Load** the pre-trained encoder.
2. **Freeze** it — `for p in encoder.parameters(): p.requires_grad = False`.
3. **Encode** each user's full event sequence. To get a single vector per
   user, take the mean of the output vectors across the sequence (or use a
   special `[USR]` token like PRAGMA does — your call).
4. **Add a tiny classifier head**: one linear layer mapping the encoder's
   output dimension to 2 logits (normal vs. fraud).
5. **Split** the data: 80% train, 20% test.
6. **Train** only the new head on the train set. Use `CrossEntropyLoss`
   with the user labels.
7. **Evaluate** on the test set. Report:
   - Overall accuracy.
   - **Recall on fraud** (what fraction of fraudsters did you catch?). This
     is the metric that actually matters — accuracy is misleading with
     class imbalance.

**Success check**: fraud recall well above the 5% you'd get from random
guessing.

---

## Stage 4: The crucial comparison

A model that does well doesn't prove that PRE-TRAINING helped. Maybe the
classifier could have learned everything by itself.

To prove the pre-training mattered, build a **baseline**:

- Same architecture, same classifier head, same training data and labels.
- But start the encoder with **random weights** (skip Stage 2).
- Train end-to-end on the labelled data only.

Compare:
- Pre-trained encoder + classifier (Stage 3).
- Random encoder + classifier (baseline).

If your synthetic data is rich enough, pre-training should win — especially
when you reduce the size of the labelled training set (try training with
only 100, 50, then 20 labelled users). This is exactly the regime where
foundation models shine: lots of unlabelled data, few labels.

**Write down what you find.** Keep it short:
- Did pre-training help?
- By how much?
- Did it help more when there were fewer labels?
- Anything that surprised you?

---

## Stretch goals (optional)

In rough order of difficulty:

1. **Add real time encoding**. Give each event a timestamp (just seconds
   from 0). Use the trick from PRAGMA §2.2: `8 * log(1 + t/8)`. Add it as
   a learned embedding alongside the value embeddings.

2. **Profile state**. Give each user some static attributes (country,
   account_age_bucket, plan). Encode them as a separate small sequence and
   include them in the input.

3. **LoRA fine-tuning instead of freeze + probe**. Instead of freezing,
   add small low-rank adapters to the attention layers and tune those.
   The paper used rank=8.

4. **Multi-task**. Add a second downstream task on top of the same frozen
   encoder — e.g., predict the user's most-used `type`. Train two heads
   independently. Does the same encoder serve both?

5. **Scale**. Increase your toy to 20k users with 100 events each, then
   100k users. Where does it start to break? What did you have to change?

---

## What you should turn in (to yourself)

A folder with:
- `capstone_data.py`
- `capstone_pretrain.py`
- `capstone_classify.py`
- A short `findings.md` (a paragraph or two — what you saw, what surprised you).

If you ever publish a paper based on this, send it to dad.

Have fun.

---

> **Stuck? Want to compare?** A full reference solution lives at
> [fernfant/mini-pragma-solutions](https://github.com/fernfant/mini-pragma-solutions).
> Try it yourself first — peek only when you've given it a real shot.
