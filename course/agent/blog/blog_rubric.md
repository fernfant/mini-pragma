# Blog scorecard — the adapted rubric for CxAI build-story posts

The course has its own 11-criterion rubric (`course/agent/SKILL.md` §8) built for an
**interactive HTML lesson** aimed at a 12–14-year-old. Four of those criteria don't
transfer to a static executive blog post — Interactivity (click/drag widgets),
keyboard Accessibility, the 20-pill/spinebar Structural correctness, and `node --check`
Technical soundness. This rubric keeps the transferable criteria and swaps the
course-only ones for blog analogues. Same shape: each criterion scored **0–5**
(0 absent · 3 acceptable · 5 exemplary), weighted, summed to **/100**, same bands
(90–100 exemplary · 75–89 strong · 60–74 acceptable · 40–59 weak · <40 failing).
A 5 is reserved for "couldn't improve it on this axis" — if you can name a concrete
fix, the ceiling is 4.

| # | Criterion | Weight | What 5/5 looks like | Course analogue |
|---|-----------|--------|---------------------|-----------------|
| B1 | **Controlling idea & narrative** | 12 | One thesis; every section serves it; callbacks and a clear arc. | C5 Narrative |
| B2 | **Concrete / worked examples** | 12 | Opens concrete; real worked numbers; examples *before* abstraction; at least one step-by-step box. | C3 Concrete-first |
| B3 | **Visuals & diagrams** | 12 | Meaningful diagrams that carry the argument (not decoration); each has a caption / text equivalent. | C1 Interactivity |
| B4 | **Accuracy / verifiable data** | 12 | Every figure traces to a named artifact (report, scorecard, code); nothing invented. | C4a Accuracy |
| B6 | **Clarity (exec reader)** | 10 | Plain second person; jargon earned; short sentences; a busy executive follows unaided. | C2 Clarity |
| B5 | **Honest framing** | 8 | Includes a genuine wrinkle (e.g. a number going *down*); no hype; simplifications flagged. | C4b Honesty |
| B7 | **Flow / sequencing** | 8 | Reads top-to-bottom once, no backtracking; each section transitions from the last. | C10 Flow |
| B9 | **Skimmability / structure** | 8 | Scannable; declarative argumentative headings; purposeful bold; lists where they earn it. | C7 Structure |
| B8 | **Pacing / runway** | 6 | Each new concept gets plain-idea → name → example; nothing named-and-used in one breath. | C11 Pacing |
| B10 | **CxAI voice-match** | 6 | Subtitle with no terminal period; "From X to Y" title; framework-by-naming; the sign-off. | (house) |
| B11 | **Takeaways / actionability** | 6 | A concrete imperative list; coined frameworks the reader can repeat. | C6 Reinforcement |

Weights sum to **100**. Per-point multiplier = weight ÷ 5.

## Scoring `build-story_method-not-model_2026-05-29` (this very post)

Ran through the **score → fix → re-score loop**, same as the course.

| # | Criterion | W | Draft | v2 | v3 | Note |
|---|-----------|---|-------|----|----|------|
| B1 | Controlling idea & narrative | 12 | 5 | 5 | 5 | "method, not model" holds throughout. |
| B2 | Concrete / worked examples | 12 | 3 | 5 | 5 | **v2 fix:** added before/after worked-example boxes (softmax pacing fix, a scored rubric row). |
| B3 | Visuals & diagrams | 12 | 1 | 4 | 4 | **v2 fix:** 4 inline SVGs + the course-rubric table. *Held at 4:* static diagrams; the course earns its top marks with draggable widgets — a concrete fix is nameable, so the ceiling is 4. |
| B4 | Accuracy / verifiable data | 12 | 5 | 5 | 5 | Every figure traces to the v3–v5 reports / scorecard. |
| B6 | Clarity (exec reader) | 10 | 4 | 4 | **5** | **v3 fix:** broke the two densest clause-stacked sentences (the agentic-batch sentence; the adapted-rubric swap). |
| B5 | Honest framing | 8 | 5 | 5 | 5 | The 92.2 dip is kept as the centrepiece. |
| B7 | Flow / sequencing | 8 | 4 | 4 | 4 | *Held at 4:* the self-scorecard coda is a deliberate meta-detour a strict cold-read flags. |
| B9 | Skimmability / structure | 8 | 4 | 5 | 5 | Visuals + boxes + table + declarative headings make it scan. |
| B8 | Pacing / runway | 6 | 4 | 4 | **5** | **v3:** the 11 criteria now get a per-row table (runway each) and the new prose is plain-idea-first — no longer a brisk single list. |
| B10 | CxAI voice-match | 6 | 5 | 5 | 5 | Subtitle (no terminal period), framework-by-naming, sign-off. |
| B11 | Takeaways / actionability | 6 | 4 | 5 | 5 | The four-move list + coined frameworks. |

**Total: 78.0 → 92.8 → 96.0.** Band: **Strong → Exemplary → Exemplary.** Draft→v2: the two
weak criteria (**B3 Visuals 1**, **B2 worked examples 3**) were exactly what the enrichment
targeted. v2→v3 (this run): two criteria earned a genuine **4 → 5** — **B6 Clarity** (dense
sentences broken) and **B8 Pacing** (the rubric became a per-row table) — for +2.0 and +1.2
weighted points. Self-scored by the author. **B3 Visuals and B7 Flow are still held at 4**
with a named, concrete residual rather than rounded up — because the rubric reserves 5 for
"couldn't improve it," and both still have a nameable fix.

**Meta pass (post-92.8).** A later edit made the post *practise* the course's pedagogy,
not just describe it: a genuine **predict-before-reveal** on the reader (guess whether a
stricter 11th criterion pushed the average up or down — it fell to 92.2), a one-line
callback setting it up, and an explicit "this article is built like a lesson — concrete,
predict-before-reveal, then graded" frame in the self-scorecard section. The device lands
in **B2** and **B10**, both already at 5, so the total **holds at 92.8** — the meta move
is qualitative, and the four held-at-4 residuals are still real. Honesty over polish: no
score was nudged for feeling cleverer.

**Agentic-iteration pass (post-meta).** Retitled "How to build agentically…" and expanded
the loop section to *show* the iterations: the v3→v4 jump as a single batch of ~40
worst-first fixes across all 13 pages (tasks #59–#71) with two coordinate-level examples
(L5 param count ~5,000→~20,000; L5b "illustrative"→real notebook recall), plus the v5
pacing pass run by **three read-only audit agents in parallel**. All figures trace to the
v4/v5 audit reports. The detail strengthens **B1** (controlling idea now explicitly
"agentic"), **B4** (more artifact-cited specifics) and **B2** — all already at 5 — so the
total again **holds at 92.8**; the new paragraphs were kept plain-idea-first to protect
B8 Pacing, and the four held-at-4 residuals stand.

**Rubric-table pass.** "The real product was the rubric" now *shows* the real instrument —
the actual 11-criterion / 12-row course rubric from `course/agent/SKILL.md` §8 (Interactivity
12 … Pacing 8, summing to 100) as a styled table — instead of describing it in a prose list.
The Pacing row carries a one-line foreshadow ("not in the first draft… hold that thought")
so the later C11 reveal still lands. Lands in **B2/B4/B9**, all already at 5; total **holds at
92.8**. (Added table CSS to the HTML template so it renders in the serif column and the PDF.)

## Scoring `build-story_who-grades-the-grader_2026-06-01` ("Who grades the grader?")

Same score → fix → re-score loop. Controlling idea: a writer that grades its own
work inflates — a blind re-audit (92.7 self → 86.7 independent) is the cure.

| # | Criterion | W | Draft | v2 | Note |
|---|-----------|---|-------|----|------|
| B1 | Controlling idea & narrative | 12 | 5 | 5 | "who grades the grader" holds throughout. |
| B2 | Concrete / worked examples | 12 | 4 | 5 | **fix:** added the L4g "bug graded a 5" before/after box + the 16-weights rename slip. |
| B3 | Visuals & diagrams | 12 | 3 | **4** | new self-vs-blind bar chart + reused spine/loop. *Held at 4:* static; course earns 5 with draggable widgets. |
| B4 | Accuracy / verifiable data | 12 | 5 | 5 | every figure traces to `spine_2026-06-01.md`, `recommend_2026-05-30.md`, scorecard JSON. |
| B6 | Clarity (exec reader) | 10 | 4 | 5 | **fix:** tightened the saturated-criteria paragraph. |
| B5 | Honest framing | 8 | 5 | 5 | the −6.0 self-vs-blind drop is the centrepiece; the self-score caveat is owned in-text. |
| B7 | Flow / sequencing | 8 | 4 | **4** | *Held at 4:* the recap + self-scoring coda are deliberate meta-detours a cold read flags. |
| B9 | Skimmability / structure | 8 | 4 | 5 | declarative headings, purposeful bold, the predict box and worked-example boxes. |
| B8 | Pacing / runway | 6 | 4 | 5 | the blind-audit reveal gets full predict→reveal→explain runway. |
| B10 | CxAI voice-match | 6 | 5 | 5 | "From X to Y" title, no-period subtitle, framework-by-naming, sign-off. |
| B11 | Takeaways / actionability | 6 | 3 | 5 | **fix:** the four moves became specific (separate grader/writer, kill saturated criteria…). |

**Total: 84.0 → 90.0.** Band: **Strong → Exemplary.** Two rows **held at 4** with a
named residual (Visuals: static vs draggable; Flow: the self-scoring coda) — not
rounded up. And the honest meta-note the post makes about *itself*: this 90.0 is
**self-scored**, the exact bias the piece spends a thousand words warning about.
