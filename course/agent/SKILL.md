---
name: course-principles
description: Principles and rating rubric for the Mini PRAGMA interactive HTML course (kid-friendly intro to Transformers/BERT/GPT/PRAGMA). Use this whenever writing, reviewing, improving, or rating any course lesson page. Defines the audience, voice, pedagogy, the 13-step spine, structural/markup conventions, the shared component library, JS conventions, anti-patterns, and a scored rubric.
---

# Mini PRAGMA — course principles

The canonical guide for what a good lesson in this course looks like. Source of
truth lives at `course/agent/SKILL.md`. Lives next to the course it describes so
it travels with the repo.

## 1. What this course is

A self-contained, interactive, static-HTML course that teaches a curious
12–14-year-old how Transformers — BERT, GPT, and Revolut's PRAGMA — actually
work, starting from "a computer can't read, it only does math on numbers." The
HTML pages live in `course/html/`. They share `styles.css` and `course.js` and
have NO build step and NO external runtime dependencies. Every page must open
correctly by double-clicking the file.

The course also has a Python/notebook track (`course/*.py`, `course/notebooks/`),
but **these principles are about the HTML lesson pages.**

## 2. Audience & voice

- **Reader:** a bright 12–14-year-old who knows basic Python and nothing about ML.
  No calculus, no linear algebra, no jargon assumed.
- **Voice:** warm, plain, second-person ("you"), short sentences. Concrete before
  abstract. A friendly teacher, not a textbook.
- **Every new term is earned.** Introduce the plain idea first, name it second.
  Anything in the glossary (`course.js` GLOSSARY) auto-links on first use — don't
  fight it, and add genuinely new terms to the glossary rather than leaving them
  undefined.
- **Analogies must be honest.** A good analogy clarifies the real mechanism; it
  must not require un-learning later. If an analogy leaks, flag the leak.
- **No hype, no hand-waving.** When something is approximate or simplified, say so
  in one clause. Numbers shown should be real (run the code) or clearly labelled
  "illustrative."

## 3. Pedagogical principles

1. **Learn by doing beats reading.** Every page should have something to click,
   drag, slide, or run. "You learn more in 30 seconds of playing than 5 minutes of
   reading" is the house style. A page that is all prose is a failing page.
2. **Predict before reveal.** Before showing a result, ask the reader to guess
   (use `predict(...)`). The "🤔 good guess — here's what actually happens" moment
   is where learning sticks.
3. **One idea per step.** Each `<h2 class="step">` advances exactly one concept.
   If a step teaches two things, split it.
4. **Show the same data transforming.** The spine is one running story (text →
   ids → vectors → attention → guess). Re-use the same example sentences/shapes
   across lessons so the reader sees continuity, not a new toy each time.
5. **Honest framing of results.** When accuracy is low, or a change makes things
   *worse* (e.g. the L4c attention regression), teach *why* — don't hide it. The
   "it got worse, here's the fix" arc is a feature.
6. **Concrete → general.** Start with a 2-word worked example with real numbers,
   then generalise. Never open with the general formula.
7. **Spaced reinforcement.** End each lesson with a `recap` ("You can now say…")
   and a `renderQuiz` self-check. Inline `check(...)` mid-lesson for quick pulse
   checks.
8. **Respect the reader's time.** A spine page is ~10–15 min. Side-quests are
   optional and labelled as such. Never make an optional digression block the
   main path.
9. **Every claim is verifiable.** Prefer numbers that come from the companion
   `.py`/notebook. If you cite a figure (params, accuracy, layer count), it must
   match the code or be labelled illustrative.
10. **Every widget works without a mouse, every figure has words.** Click/drag/slide
   controls must be keyboard-operable and focus-visible; an interaction a kid can't
   reach with Tab/Enter isn't "learn-by-doing" for everyone. Every meaningful
   SVG/figure carries a `<figcaption>` or text equivalent so the idea survives if
   the image doesn't render.

## 4. Structure: the 13-step spine

The main path is a single linear spine. Pagination must be reciprocal (each
page's Next ↔ the next page's Prev).

| Step | File | Title role |
|---|---|---|
| 1 | lesson_01.html | What is a model? (learning) |
| 2 | lesson_01_5.html | The single neuron / MLP |
| 3 | lesson_02.html | Tokens & embeddings |
| 4 | lesson_02a.html | Scaling up embeddings |
| 5 | lesson_03.html | Attention |
| 6 | lesson_04.html | Training a tiny BERT (MLM) |
| 7 | lesson_04f.html | Next-word / generation |
| 8 | lesson_04g.html | Bonus deep-dives (brain to scale) |
| 9 | lesson_04h.html | minigpt |
| 10 | lesson_05.html | Putting it together (PRAGMA) |
| 11 | lesson_05a.html | Tabular (house predictor) |
| 12 | lesson_05b.html | Streaming churn (foundation-model pipeline) |
| 13 | lesson_06.html | Capstone |

**Off-spine branches** (reachable but not part of the 13 steps):
- `opt` side-quests: lesson_01b, lesson_01c, lesson_03b
- `deep` code walkthroughs: lesson_04_code, lesson_04h_code
- index.html (roadmap), glossary.html

## 5. Markup & component conventions

Every spine page has, in order:
1. `<nav class="topbar">` with `.brand` + `.lesson-pills` — **21 pills, identical
   on every page** (Index + 19 lessons/branches + glossary), only the `current`
   marker differs. Side-quests carry class
   `opt`, code walkthroughs `deep`. (When current: `opt current` / `deep current`
   / `current`.)
2. `<div class="spinebar">` with `.sb-home` (🗺 → index), `.sb-part` (Part label),
   `.sb-track`>`.sb-fill` (width %), `.sb-step` ("Step N of 13"). Branch pages use
   "Side-quest"/"Code walkthrough" labels and **no `.sb-fill`**.
   - 13-step fill widths: 7.7, 15.4, 23.1, 30.8, 38.5, 46.2, 53.8, 61.5, 69.2,
     76.9, 84.6, 92.3, 100 (%).
3. `<div class="wrap">` — content. `<h1>` + `<p class="subtitle">`, optional
   `.lesson-links`, optional `<details class="lesson-toc">`, then `<h2 class="step"
   id="...">` sections each opening with a `<span class="step-num">` chip.
4. `.recap` card → `<div id="quiz-..">` (renderQuiz) → `.pagination` (Prev/Next).
5. `<footer class="site">`.

**Component library** (all in `course.js`, no deps):
- `predict(id, {q, options, correct, reveal})` — predict-before-reveal MCQ.
- `check(id, {q, options, answer, explain})` — slim inline checkpoint MCQ.
- `renderQuiz(id, [{q, options, answer, explain}])` — end-of-lesson self-check.
- Glossary auto-linking runs on DOMContentLoaded; `<span class="term"
  data-def="...">` is the inline tooltip form.
- Callouts: `.callout` (neutral), `.callout.info`, `.callout.good`, `.callout.bad`.
- Figures: `<figure class="figure-wide">` with `<figcaption>`; SVGs live in
  `course/visuals/`.

**Interactive-widget data pattern:** put data in an inline
`<script id="X-data" type="application/json">…</script>` block; read it from a
self-contained IIFE with graceful guards (`const el = document.getElementById(...);
if (!el) return;`). Widgets must degrade silently if their host node is absent.

## 6. JS / code conventions

- Vanilla JS only. No frameworks, no CDN, no build. Self-contained IIFEs.
- Guard every DOM lookup; never throw on a page that lacks the widget.
- Inline data as `type="application/json"` (parsed once), logic separate.
- **Smoke test before declaring done** — extract every inline `<script>` (exclude
  `src=` and `application/json`) and `node --check` each. Zero errors required.
- Keep diffs small and focused. Match the surrounding style. Minimal comments.

## 7. Anti-patterns (a lesson FAILS if it does these)

- Wall of prose with no interaction.
- Formula or jargon before the concrete example / plain-English idea.
- A new undefined term with no glossary entry and no inline definition.
- Fake or unlabelled numbers; figures that contradict the companion code.
- An analogy that the reader must later un-learn, presented as fact.
- Broken nav: wrong pill count, missing `current`, wrong "Step N of 13", or
  non-reciprocal pagination.
- A widget that throws (no guard), has malformed JSON data, or can only be operated
  with a mouse (no keyboard path, no visible focus).
- Optional digressions that block the main path, or duplicated content that
  drifts between pages.

## 8. Rating rubric (score a version against the principles)

Score each criterion 0–5 (0 absent/violated, 3 acceptable, 5 exemplary), then a
weighted total out of 100. Always justify each score with a concrete page
reference (section id / line).

| # | Criterion | Weight | What 5/5 looks like |
|---|---|---|---|
| 1 | **Interactivity** | 12 | Multiple meaningful click/drag/slide/run widgets; predict-before-reveal used at the key moment. |
| 2 | **Age-appropriate clarity** | 9 | Plain second-person voice; every term earned; short sentences; a 12–14-yo could follow unaided. |
| 3 | **Concrete-first pedagogy** | 9 | Opens with a real worked example/numbers; one idea per step; generalises only after. |
| 4a | **Numeric/factual accuracy** | 8 | Every cited figure (params, accuracy, layer counts, demo numbers) matches the companion code or is explicitly labelled "illustrative"; zero unverified assertions. |
| 4b | **Honest framing** | 6 | Simplifications flagged in-clause; no analogy sold as fact that the reader must later un-learn; "it got worse, here's why" arcs kept honest. |
| 5 | **Narrative continuity** | 8 | Re-uses the running example; connects back/forward; fits the spine story. |
| 6 | **Reinforcement** | 8 | Inline checks + recap + quiz; questions test understanding, not recall. |
| 7 | **Structural correctness** | 8 | Correct pills (21, right `current`), spinebar step/width, reciprocal pagination, recap→quiz→pager order. |
| 8 | **Technical soundness** | 8 | All inline JS passes `node --check`; widgets guarded; JSON valid; no external deps. |
| 9 | **Accessibility** | 8 | Every interactive widget is keyboard-operable with visible focus; every figure/SVG has a text equivalent (`<figcaption>` or aria-label); colour is never the sole signal (callout `good`/`bad` also carry a word/icon). |
| 10 | **Flow / sequencing** | 8 | A 12–14-yo can read top-to-bottom *once*, never backtracking: every term is defined before first use, each step says how it builds on the last, one new idea per step, order goes concrete→abstract / simple→hard. Distinct from #2 (per-sentence pitch) and #5 (cross-lesson threading) — this is *within-page* logical flow. |
| 11 | **Pacing / gradual build-up** | 8 | Every genuinely new, load-bearing concept gets real runway: plain idea → name → micro-example → use. A concept is never named-and-used inside a single clause or parenthetical. Distinct from #2 (per-sentence pitch), #3 (concrete *opening*) and #10 (ordering / used-before-defined) — this is *how much runway each new idea gets*. |

**Scoring #11 (Pacing).** For each genuinely new, load-bearing concept on the page, check the four-beat runway: (1) plain-idea sentence, (2) name introduced, (3) a micro-example or worked instance, (4) use. Score = how consistently concepts get all four: every load-bearing concept gets full runway → 5; one concept named-and-used in a single clause/parenthetical with no micro-example → 4; several rushed → 3; most concepts compressed → ≤2. Justify by citing the rushed concept and its section id (e.g. "`#train`: cross-entropy named and used in one line"). A concept that is a deliberate *callback* to an earlier full treatment does not need re-runway — a one-line re-gloss suffices.

**Scoring #10 (Flow).** Do a **cold read**: simulate a 13-year-old reading the page once, top-to-bottom, *no scrolling back*. Log every point where they'd get confused or have to re-read. The score is the inverse of confusion points: 0 points → 5; 1–2 minor stumbles → 4; one place you genuinely *must* backtrack → 3; several backtracks → 2; the order fights the reader throughout → ≤1. Justify the score by citing the **checklist probes** that fail (section id / line): (a) a term used before it's defined; (b) a step with no transition saying how it builds on the prior one; (c) a step introducing more than one new idea; (d) a current step that depends on an "as we'll see later" hand-wave; (e) abstract-before-concrete ordering. Optionally report an unweighted Flesch-Kincaid grade (target 6–8) as a sanity floor — it does **not** set the score (it's blind to ordering).

**Total /100.** Bands: 90–100 exemplary · 75–89 strong · 60–74 acceptable ·
40–59 weak · <40 failing. Report per-criterion score, weighted subtotal, total,
band, and the top 3 highest-leverage fixes.

A score of 5/5 on any criterion is reserved for pages that are exemplary *and*
could not be improved on that axis — if you can name a concrete improvement, the
ceiling is 4. "Acceptable, nothing wrong" is 3, not 4. When two pages tie within a
band, rank the one with a genuine predict-before-reveal at the key moment higher
(the course's signature move).
