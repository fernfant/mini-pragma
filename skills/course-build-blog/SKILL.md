---
name: course-build-blog
description: Write a blog post — in the voice of the CxAI Substack (https://cxai100.substack.com/) — that tells the story of HOW the Mini PRAGMA interactive course was built: the process, the structure, the AI-agent workflow, the scoring rubric, and the score→fix→re-score loop. Use this whenever the user asks for a "build story", "how we built it" post, a launch/retrospective writeup, or a CxAI-style article about the course. The post must be grounded in the real artifacts (rubric, scorecard, reports, task ledger) and the project memory — no invented numbers.
---

# course-build-blog — write the "how we built it" post, CxAI-style

The job: produce a finished, publishable blog post that describes **the exact
process and structure we used to build the Mini PRAGMA course**, written so it
would sit naturally in the CxAI Substack archive (https://cxai100.substack.com/).
CxAI is a consumer-experience × AI newsletter for executives; this post is the
"we built an AI-taught course with an AI agent, here's the method" story told to
that audience.

Output is a single Markdown file written to `course/agent/blog/` (filename
`build-story_<topic-slug>_<YYYY-MM-DD>.md`). It is a **deliverable**, not a draft
to discuss — write the whole thing, then tell the user where it is and paste a
short excerpt.

## Workflow when invoked

1. **Read the real sources first** (never invent figures — every number in the
   post must trace to one of these):
   - `course/agent/blog/cxai-principles.md` — **the canonical CxAI voice/style
     guide** (title/subtitle, hook, headings, voice, signature moves, the
     visuals + worked-example + self-scorecard conventions, hard rules). Read this
     before drafting; it is the source of truth for "how to write".
   - `course/agent/blog/blog_rubric.md` — the adapted 0–5 weighted rubric the post
     is scored against (and the self-scorecard template).
   - `course/agent/SKILL.md` — the canonical principles + the scored rubric
     (criteria, weights, bands). This is the "structure" half of the story.
   - `course/html/scorecard.html` — the live per-page scorecard (the `sc-data`
     JSON block has every page's scores, totals, bands, and the criteria/weights).
   - `course/agent/reports/spine_*.md` — the audit reports. The version suffix
     (v3 → v4 → v5) IS the score→fix→re-score arc; read the latest two to get the
     before/after averages and what each batch changed.
   - The course itself: `course/html/index.html` + the 13 spine pages for the
     spine structure (the 13-step table in SKILL.md §4 is the quick reference).
   - Project memory: `~/.claude/projects/-Users-fernando-Pragma-LLM-model/memory/MEMORY.md`
     and any `project-*.md` for prior context, the author's framing, and related
     build-it courses (learn-physics, learn-trading) to cross-reference if useful.
2. **Confirm the angle** with the user only if ambiguous; otherwise default to the
   full build-story angle below. Good alternate angles: "the rubric as the real
   product", "what changed when our reviewer was an AI", "score→fix→re-score as a
   content methodology". Pick ONE controlling idea — CxAI posts argue one thing.
3. **Draft against `cxai-principles.md`** and the structure template below. Build
   the visuals (2–4 inline SVGs) and at least one before/after worked-example box
   as the principles require.
4. **Fact-check every number** against the sources (averages, page count,
   criterion count, weights, band counts, specific fixes). If a figure isn't in an
   artifact, cut it or label it.
5. **Score → fix → re-score the post itself** against `blog_rubric.md`: score the
   draft cold, fix the weakest criteria first, re-score, and add the "We graded
   this post, too" section.
6. **Generate the three formats** (`.md` → Substack-styled `.html` → `.pdf`; see
   the production-formats note in `cxai-principles.md`), report the paths, and
   paste the title + subtitle + first ~150 words so the user can sanity-check the
   voice.

## The CxAI style guide

**The full voice/style guide is `course/agent/blog/cxai-principles.md` — read it and
match it.** It is the single source of truth for title/subtitle shapes, the opening
hook, argumentative headings, the first-person-plural→second-person voice, the
formatting devices, sentence rhythm, the signature moves (framework-by-naming,
data-as-embedded-evidence, before/after contrasts), the visuals + worked-example +
self-scorecard conventions, length, and the hard rules. Do **not** restate the guide
here — edit `cxai-principles.md` so the two never drift.

## Structure template for THIS post (the build story)

Use as a starting skeleton; adapt headings to the CxAI declarative style.

1. **Hook** — a concrete scene: an AI agent that wrote a 13-lesson course, and the
   uncomfortable question that followed (is it any *good*?). State the one
   controlling idea: the method, not the model, made it work.
2. **What we set out to build** — one paragraph: a self-contained, interactive,
   no-build static-HTML course teaching a 12–14-year-old how Transformers (BERT,
   GPT, PRAGMA) actually work. Audience + the "learn by doing beats reading" bet.
3. **The structure: one spine, thirteen steps** — the 13-step linear spine
   (text → ids → vectors → attention → guess → fine-tune), reciprocal pagination,
   side-quests kept off the main path. Structure as a product decision.
4. **The real product was the rubric** — the principles doc + the 0–5, weighted,
   /100 rubric (interactivity, concrete-first, honesty, accessibility, flow,
   pacing…). Why we wrote the grader before trusting the writer.
5. **The loop: score → fix → re-score** — the v3→v4→v5 arc. Real numbers: the
   batch that moved the average from 86.6 to 92.8, the per-criterion targeting,
   the honesty note that the implementer scored conservatively.
6. **The signature move: predict-before-reveal** — and the honest-failure arc
   ("it got worse, here's the fix"). Why these stick.
7. **Keeping the agent honest** — verification discipline: every cited number runs
   from the companion notebook (the L5b recall figures), every inline script
   passes `node --check`, accessibility and keyboard paths checked. The agent
   doesn't get bored — that's the asset and the risk.
8. **The pacing pass** (optional, if the latest report covers it) — adding an 11th
   criterion (C11 Pacing) after a human said "concepts are introduced too fast",
   finding the 9 spots, fixing them, re-scoring to 93.2. A worked example of
   human taste steering an AI loop.
9. **What this means if you're building with AI** — short imperative list for the
   reader-executive: write the rubric first; make the artifact verifiable; keep a
   human holding the taste; let the loop, not the vibes, decide done.
10. **Close** — forward-looking aphorism + "Stay tuned! CxAI Team".

## Hard rules

The hard rules live in `course/agent/blog/cxai-principles.md` (§ Hard rules) — every
number real, honesty over polish, one controlling idea, no hype/invented quotes,
author in Markdown and generate the HTML/PDF from it. Read them there. The two
project-specific reminders for THIS post type:

- **Ground every figure in an artifact** (the rubric, `scorecard.html`, the
  `spine_*.md` reports). The v3→v4→v5 arc — 86.6 → 92.8 → 92.2 → 93.2 — is the
  spine of the story; don't fuzz those numbers.
- **Link to the live product.** Point readers at the published course
  (`https://fernfant.github.io/mini-pragma/…`) and the live scorecard so they can
  check the marking.
