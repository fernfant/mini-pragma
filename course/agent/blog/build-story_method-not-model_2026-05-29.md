# From Blank Repo to 93/100: How We Let an AI Build a Course — and a Rubric Keep It Honest

*Why the method, not the model, is what made an AI-written curriculum worth shipping*

A coding agent wrote a thirteen-lesson course on how Transformers work — the real machinery behind BERT, GPT, and Revolut's PRAGMA — pitched at a curious thirteen-year-old. It did it fast. Static HTML, no build step, interactive widgets, a running example carried from "a computer can't read, it only does math on numbers" all the way to a fine-tuned model. Impressive. Then came the uncomfortable question, the one every team shipping AI-generated content eventually has to answer out loud:

**Is it any good?**

Not "does it run." Not "does it look finished." Is it *honest*, is it *clear*, would a kid actually learn from it — and how would we know, across thirteen pages, without simply reading each one and nodding along? That question is the whole story. Because the thing that made this course defensible wasn't the model that wrote it. It was the method we wrapped around the model. This is that method.

## The bet: learn by doing, taught by a machine

What we set out to build was deliberately narrow. A self-contained, interactive, no-build static-HTML course that teaches a bright 12–14-year-old how Transformers actually work — starting from arithmetic, ending at a working classifier — with no calculus, no linear algebra, and no jargon assumed. The house bet is one line: **you learn more in 30 seconds of playing than 5 minutes of reading.** Every page has something to click, drag, slide, or run. A page that is all prose is, by our own rule, a *failing* page.

That bet matters here because it's exactly the kind of standard an AI writer will quietly violate. Models love prose. They will happily generate four tidy paragraphs explaining attention and call it a lesson. Holding the line against that — page after page — is not a writing problem. It's a *measurement* problem.

## Structure is a product decision, not a table of contents

Before a single lesson, we fixed the shape. The course is **one linear spine of thirteen steps**: text → ids → vectors → attention → a guess → fine-tuning. The same example sentence transforms across lessons, so the reader sees continuity instead of a new toy each time. Pagination is reciprocal — every page's *Next* matches the following page's *Prev* — so nobody falls off the path. Genuinely optional material (side-quests, code walkthroughs) is reachable but kept *off* the spine, clearly labelled, never blocking the main route.

This sounds like housekeeping. It isn't. The spine is what lets you say "step 7 builds on step 6" and mean it — and it's what lets a reviewer, human or machine, check that claim mechanically. A wandering course can't be graded. A spine can.

## The real product was the rubric

Here is the part nobody warns you about: when your writer is an agent, **the writer is not the bottleneck. The grader is.**

So we wrote the grader first. Two artifacts sit upstream of every lesson. The **principles doc** says what a good page *is* — concrete before abstract, every term earned before it's named, analogies that don't have to be un-learned later, every cited number traceable to the companion code. And a **scored rubric** turns those principles into something you can argue with: each page scored 0–5 on a set of weighted criteria, summed to a single number out of 100, with explicit bands (90+ exemplary, 75–89 strong, and down).

Today that rubric has **eleven criteria** across **thirteen pages** — interactivity (weighted heaviest, at 12 of 100), age-appropriate clarity, concrete-first pedagogy, numeric accuracy, honest framing, narrative continuity, reinforcement, structural correctness, technical soundness, accessibility, within-page flow. Weights sum to exactly 100, on purpose. The single most important design choice in the whole project was this: **we built the scoring rubric before we trusted the writer.** The rubric is the source of truth. The lessons are just the current best attempt to satisfy it.

If your team is shipping AI-generated anything and you don't have this artifact — the explicit, weighted definition of "good" that exists *independently* of the output — you don't have a quality process. You have vibes and a fast typist.

## The loop: score → fix → re-score

With a grader in hand, the work becomes a loop, and the loop has a name: **score → fix → re-score.** Read every page cold, score each criterion against the rubric, rank worst-first, fix the lowest-leverage failures, then score again. Repeat until the number stops moving.

The arc is in the audit reports, and the numbers are real. The first full pass (v3) averaged **86.6** — five pages exemplary, eight merely strong. One targeted batch later (v4), the average climbed to **92.8**, with nine pages exemplary. Same model. Same writer. The gain came entirely from grading what it produced and feeding the worst scores back in.

That's the mechanism most "AI writes your content" demos skip. The first draft is never the product. **The loop is the product**, and the rubric is what closes it.

## The moves that make learning stick

Two house techniques carry most of the pedagogical weight, and both are things the rubric explicitly rewards.

The first is **predict-before-reveal.** Before showing any result, the page asks the reader to guess. The "🤔 good guess — here's what actually happens" beat is where understanding lodges, and when two pages tie in score, the one with a genuine prediction at the key moment ranks higher. It's the course's signature move, and it's baked into the grader so the agent can't forget it.

The second is the **honest-failure arc** — "it got worse, here's why." When a change makes accuracy *drop*, we teach the drop instead of hiding it. A real attention regression in one lesson is presented as a regression, then explained. This is a pedagogical choice, but it's also the project's ethic, and — as you're about to see — we held ourselves to it too.

## The agent doesn't get bored. That's the asset and the risk.

There's a reason this method works better with a machine than it ever did with a tired human team: **the agent doesn't get bored.** It will re-read all thirteen pages on the eleventh pass with exactly as much care as the first. It will touch fifteen files to fix one cross-reference. The maintenance burden that kills human-run quality processes — the bookkeeping, the consistency checks, the "did we update that everywhere" — costs the agent almost nothing.

That's the asset. The risk is the same sentence read the other way: an agent that never gets bored also never gets *suspicious*. It won't wrinkle its nose at a number that smells wrong. So we made suspicion mechanical. Every cited figure must trace to the companion notebook — the recall and precision numbers in the streaming-churn lesson are computed, not asserted. Every inline script must pass `node --check`. Every widget must be keyboard-operable with visible focus; every figure must carry a text equivalent. Verification isn't a final step. It's a gate the loop runs through every time.

## When a stricter ruler makes the score go down

And now the wrinkle — because a flawless arc would be exactly the kind of dishonesty this course was built to avoid.

A human read the course and said something the rubric, in all its ten-criteria precision, had missed: *concepts are introduced too fast.* Not wrong, not out of order, not unclear sentence-by-sentence — just named and used in the same breath, with no micro-example in between. None of the existing criteria caught it. So we added an eleventh, **C11 Pacing**, defined tightly: does each genuinely new, load-bearing concept get real runway — *plain idea → name → micro-example → use* — or does it land all at once?

We carved C11's eight points out of the most over-weighted criteria (interactivity dropped from 16 to 12) so the total still summed to 100. Then we re-scored. And the average **went down** — to **92.2** — because the new, stricter ruler immediately found nine pages with a real pacing gap.

That drop is the most honest number in the whole project. It's what a sharper standard is *supposed* to do: surface problems the old standard was blind to. We found the nine spots, fixed each one — a softmax example worked through `[2,1,0] → [0.66, 0.24, 0.10]`, a two-example batch shown before the word "batch size", a plain definition of recall placed *before* the widget that uses it — and re-scored a final time to **93.2**, nine pages exemplary again.

Two pages we deliberately *held* at four out of five, with the residual named in writing rather than rounded up: one still hides its full explanation in a collapsed box, the other still introduces two terms inside table cells. The whole thing is self-scored by the implementer, and the report says so. **93.2 is a conservative, defensible average — not a ceiling we awarded ourselves.** Honesty over polish, even in the scorecard.

## What this means if you're building with AI

Strip away the Transformers and the thirteen-year-old, and the method generalises to any team putting AI-generated work in front of real people. Four moves:

1. **Write the rubric before you trust the writer.** The explicit, weighted definition of "good" is the actual product. The output is a candidate.
2. **Make the artifact verifiable.** If a number can be computed, compute it. If a script can be checked, check it — every pass, not once.
3. **Keep a human holding the taste.** The agent runs the loop tirelessly; the human decides *what to measure.* C11 existed because a person noticed something no criterion did.
4. **Let the loop, not the vibes, decide done.** When the score stops climbing and you can't name the next fix, you're done. Not before.

The model wrote the course. The method made it honest. Don't confuse the two — and when a stricter ruler makes your numbers drop, that's not a setback. That's the ruler working.

Stay tuned! CxAI Team
