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

| # | Criterion | W | Before | After | Note |
|---|-----------|---|--------|-------|------|
| B1 | Controlling idea & narrative | 12 | 5 | 5 | "method, not model" holds throughout. |
| B2 | Concrete / worked examples | 12 | 3 | 5 | **Fix:** added before/after worked-example boxes (the softmax pacing fix, a scored rubric row). |
| B3 | Visuals & diagrams | 12 | 1 | 4 | **Fix:** added 4 inline SVGs (score arc, spine, loop, rubric bars). *Held at 4:* static diagrams, not interactive like the course's widgets. |
| B4 | Accuracy / verifiable data | 12 | 5 | 5 | Every figure traces to v3–v5 reports / scorecard. |
| B6 | Clarity (exec reader) | 10 | 4 | 4 | *Held at 4:* a couple of dense clause-stacked sentences remain. |
| B5 | Honest framing | 8 | 5 | 5 | The 92.2 dip is kept as the centrepiece. |
| B7 | Flow / sequencing | 8 | 4 | 4 | *Held at 4:* the self-scorecard coda is a deliberate meta-detour. |
| B9 | Skimmability / structure | 8 | 4 | 5 | Visuals + boxes + declarative headings make it scan. |
| B8 | Pacing / runway | 6 | 4 | 4 | *Held at 4:* the rubric's 11 criteria are introduced briskly in one list. |
| B10 | CxAI voice-match | 6 | 5 | 5 | Subtitle, "From X to Y" title, framework-by-naming, sign-off. |
| B11 | Takeaways / actionability | 6 | 4 | 5 | The four-move list + coined frameworks. |

**Total: 78.0 → 92.8.** Band: **Strong → Exemplary.** The two weak criteria before the
pass were exactly the two this enrichment targeted — **B3 Visuals (1)** and **B2 worked
examples (3)** — which is the whole point: the rubric, not taste, picked the fixes.
Self-scored by the author; B3, B6, B7, B8 held at 4 with a named residual rather than
rounded up.

**Meta pass (post-92.8).** A later edit made the post *practise* the course's pedagogy,
not just describe it: a genuine **predict-before-reveal** on the reader (guess whether a
stricter 11th criterion pushed the average up or down — it fell to 92.2), a one-line
callback setting it up, and an explicit "this article is built like a lesson — concrete,
predict-before-reveal, then graded" frame in the self-scorecard section. The device lands
in **B2** and **B10**, both already at 5, so the total **holds at 92.8** — the meta move
is qualitative, and the four held-at-4 residuals are still real. Honesty over polish: no
score was nudged for feeling cleverer.
