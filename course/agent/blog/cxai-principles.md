# CxAI house style — principles & guidelines

The canonical voice/style guide for blog posts written in the manner of the **CxAI
Substack** (https://cxai100.substack.com/). Any CxAI-style post — built by the
`course-build-blog` skill or by hand — should match this document. It is the "how to
write" half; the "how to grade" half lives in [`blog_rubric.md`](blog_rubric.md).
Keep the two in sync: every guideline here should map to a rubric criterion there.

## 1. What CxAI is

A consumer-experience × AI newsletter written **for executives**. Posts are
boardroom-credible but plain-spoken, argue **one thesis**, and earn the reader's
time in the first sentence. The reader is smart, busy, and allergic to hype.

## 2. Title & subtitle

- **Title** favours the **"From X to Y"** transformation arc — "From Blank Repo to
  93/100", "From Prompt to Curriculum". A colon + reframe is fine.
- **Subtitle** (the teaser under the title): one line, **no terminal period**. Two
  shapes: (a) a "How X is Y" reframe ("How a scoring rubric, not a prompt, became
  the real product"); or (b) a tension stated flatly ("The agent wrote 13 lessons
  in a weekend — keeping them honest took the next three").

## 3. Opening hook

Never open cold. Open with one of: a concrete scene/anecdote, a **callback to prior
work** ("In our last piece we argued…"), or a provocation that sets up tension.
The first sentence earns the read; the first paragraph names the stakes.

## 4. Body organisation

Either (1) **numbered principles/findings** (3–6, each a bolded claim then 1–2
paragraphs), or (2) **thematic narrative sections** with argumentative label
headings. Always end with an action-oriented section ("What this means if you're
building with AI" / a short imperative list) before the close.

## 5. Headings

Short, declarative, and they **make an argument** — not bare topic labels. "The
part nobody warns you about", "Trust is the elephant in the room", "When a stricter
ruler makes the score go down". An occasional question heading is fine
("Opportunity or threat?").

## 6. Voice

- **First-person plural** ("we built", "our rubric"), pivoting to **second person**
  to address the reader-executive directly ("If your team is shipping AI-generated
  content…").
- Polished but plain — never academic. **Dryly contrarian** — enjoys puncturing
  hype ("the agent didn't get tired; that was exactly the problem").
- Rhetorical questions as pivots.

## 7. Formatting devices

- **Bold** is heavy and purposeful: lead bullets with the bolded claim, mark coined
  terms, bold the two halves of a contrast pair.
- **Bullet lists** for principles/drivers/enumerations; **numbered lists** for
  sequences and final recommendations.
- **Em-dashes** for mid-sentence elaboration. *Italics* for coined/technical terms.
  Block quotes for an authority line — or, in this project, for a **worked-example
  box** (see §10).
- A parenthetical aside to steer the reader is on-brand.

## 8. Sentence rhythm

Build-up, then snap. Long clause-stacked explanatory sentences broken by short
declaratives for emphasis. ("You can't review 13 lessons by feel. Not at this bar.")

## 9. Signature moves (use several)

- **Framework-by-naming:** coin or borrow a labelled concept and bold it — the
  **score→fix→re-score loop**, **predict-before-reveal**, the **rubric as source of
  truth**, the **honest-failure arc** ("it got worse, here's why").
- **Data as embedded, attributed evidence:** numbers live inside sentences, not in
  callout boxes — "the average climbed from **86.6 to 92.8**", "**13** spine pages
  scored on **11** criteria", "we added a criterion and the average *dropped* to
  92.2 before the fixes pulled it to 93.2". Always tie a number to the artifact it
  came from.
- **Before/after & X-vs-Y contrasts** as the argumentative engine (human review vs
  rubric review; "wrote it fast" vs "made it honest"; prose page vs interactive page).
- **Callbacks** to the project's own history and to sibling work.
- **Historical/analogical anchoring** where it earns its place (test-driven
  development, code review, a teacher's lesson plan).

## 10. Visuals & worked examples (this project's house additions)

A CxAI post here is not a wall of prose. It carries part of its argument visually,
and it shows its work.

- **Diagrams.** Include 2–4 **inline, self-contained SVG** figures that carry the
  argument (a score arc, a process loop, a structure spine, weighted bars) — never
  decoration. Constraints so they survive every format:
  - Inline `<svg>` (no external image files the HTML/PDF can't bundle); palette
    keyed off the accent (`#b9410a`), ink `#242424`, muted `#6b6b6b`.
  - Every figure carries a **caption** *and* a `role="img"` + `aria-label` text
    equivalent. **Escape `&`, `<`, `>`** in SVG text (raw `&` breaks the file as XML).
  - Must render identically in the Markdown, the Substack-styled HTML, and the PDF.
- **Worked-example boxes.** At least one **before/after** box, written as a
  blockquote so it renders everywhere: *before (the rushed version)* → *after (the
  fixed version, with real numbers)* → one line naming the difference. Concrete
  beats conceptual; show the actual `[2,1,0] → [0.66, 0.24, 0.10]`, not "softmax
  normalises".

## 11. Hold the post to a scorecard

The post argues for the **score→fix→re-score loop** — so run the post through it.
Score the draft cold against [`blog_rubric.md`](blog_rubric.md), fix the
lowest-scoring criteria first, re-score, and include a short **"We graded this
post, too"** section reporting the before→after total and the criteria held below 5
with a **named residual** (don't round up because the prose feels nice). Eating the
dog food is the most on-brand move available.

## 12. Close

A forward-looking, slightly aphoristic line, then the light brand sign-off:
one zinger + **"Stay tuned! CxAI Team"**.

## 13. Length & density

~1,200–2,000 words of prose (a self-scorecard section may push it higher). 7–10
sections. Paragraphs medium (3–6 sentences) with the occasional **one-line
paragraph for punch**.

## Hard rules (a post fails if it breaks these)

- **Every number is real.** Pull averages, counts, weights, and specific fixes from
  project artifacts (rubric, scorecard, audit reports). If you can't source it,
  don't print it.
- **Honesty over polish.** Include at least one genuine wrinkle (e.g. a number
  *dropping* when a stricter criterion was added), not a flawless arc.
- **One controlling idea.** Choose the angle; let the rest serve it. Don't list
  everything you did.
- **No hype, no corporate mush, no invented quotes.** Coin frameworks, attribute
  data, address the reader directly.
- **Author the Markdown in Markdown.** Title `#`, subtitle as an italic line
  directly under it, sections `##`, figures as `![caption](assets/...svg)`,
  worked examples as `>` blockquotes. The styled HTML and PDF are generated *from*
  the Markdown, not authored separately.

## Production formats

Each post ships in three forms in `course/agent/blog/`:
1. **`*.md`** — the source (renders on GitHub, links resolve to the live course).
2. **`*.html`** — a self-contained, Substack-styled page (publication masthead,
   kicker, serif reading column, accent links, inline SVGs, styled worked-example
   boxes). Built by inlining the SVGs into a template around the pandoc-rendered body.
3. **`*.pdf`** — printed from the HTML via headless Chrome (`--no-pdf-header-footer`).
