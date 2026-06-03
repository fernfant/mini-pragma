---
name: karpathy-transformation
description: Transform a Mini PRAGMA lesson page to teach like Andrej Karpathy — one running example, felt failure→fix arcs, glass-box forward passes with printed shapes, runnable in-page Python, a problem-driven spine, and clutter demoted to optional expanders. Use whenever the user asks to "Karpathyify", "make it more like Karpathy's courses", add runnable code to a lesson, build a failure→fix arc, tighten a page's narrative/spine, or fix a page that "loses shape" / "feels like a tour". Encodes the exact `.runpy` helper recipe and the verification discipline (capture in python3 → verify live in Pyodide → commit).
---

# karpathy-transformation — teach a lesson page the way Karpathy would

This skill captures the method we used to overhaul L1.5 and L2. It turns a
correct-but-flat lesson into one that *teaches by doing*: the reader feels a
problem, watches a thing fail, fixes it, and runs the real code. Apply it to any
spine page in `course/html/`. It sits on top of `course-principles` (the rubric +
conventions) — read that first; this is the **transformation playbook**.

## The five moves (in priority order)

These are Karpathy's teaching instincts, ranked by leverage for this course.

1. **One running example.** Pick a single concrete example and ride it through
   the whole page. Don't introduce a fresh toy per section. L1.5 rides the
   parabola; L2 rides `"the dog runs"` and the dog/puppy/table words. Re-use the
   same numbers, shapes, and words the rest of the spine uses (text → ids →
   vectors → attention → guess). Continuity *is* the teaching.

2. **Runnable code on the page** (`.runpy` helper — recipe below). A printout the
   reader can re-run beats a paragraph describing the printout. Every major
   concept should have a cell they can edit and Run.

3. **Failure → fix arc.** Never present the working thing first. Show the *naive*
   thing failing, let the reader feel the pain, then fix it and re-run the **exact
   same probe**. This is the course's signature move and the highest-leverage edit
   on a flat page.
   - L1.5: a straight line *can't* fit the curve (loss stuck at 2.8) → add `x²` →
     loss crashes to ~0.
   - L2: random embeddings are *useless* (`dog·puppy ≈ dog·table ≈ 0`, "can't tell
     puppy from table") → train on shared company → `dog·puppy → 1.00`, `dog·table → 0`.
   - Test: if a section goes straight to the working result, it's a candidate.

4. **Glass-box forward pass.** Walk one input through the machine with the
   intermediate values and **`.shape` printed at every step**. Make the invisible
   visible: `X shape = (3, 2)` → `W shape = (2, 3)` → `Z shape = (3, 3)`. The
   reader should never have to imagine what a tensor looks like.

5. **Demote clutter to optional.** Anything that orients/reconciles/digresses but
   doesn't advance the main problem goes into `<details class="callout">` with a
   `(optional)` summary — it stops blocking the road but stays for the curious.
   (We moved L2's "where did L1.5's net go?" reconciliation into an expander.)

## The problem-driven spine (why a page "loses shape")

A page reads well when **every section solves a problem the reader can feel**, and
each problem escalates from the last. A page "loses shape" — feels like a *tour of
mechanics* — the moment sections become "here's another thing" instead of "here's
why we *had* to."

Diagnostic, when a page goes slack partway down:

1. **Map each section to the question it answers.** Write the implicit problem next
   to every `<h2 class="step">`. (See the L2 map: dot → "how measure alike?",
   embed → "where does the vector come from?", etc.)
2. **Find where the original problem gets solved.** That's usually the climax
   (often the failure→fix cell). Everything after it goes slack *because the
   tension is gone*.
3. **Re-arm with the next problem.** There is almost always a real, bigger problem
   latent in the back half. Name it as a **felt turning point**, not a "now let's
   also". L2's back half had a genuine wall hiding in a callout — *"we hand-wrote
   every pair; real text has no answer key"* — so we promoted it to a headline
   (`🚧 Nobody writes the pairs for you — so how does it scale?`) and framed the
   next section as its **escape** (masked language modelling = the text labels
   itself). A second failure→fix at a bigger scale.
4. **Move backward-looking reconciliation out of the forward push** (→ expander).

Rule of thumb: a strong page is a chain of *"problem → felt failure → fix →
new, bigger problem"*. When the chain breaks, the page reads as a list.

## The `.runpy` runnable-code recipe

Runnable Python runs in-browser via Pyodide. The helper is already built — you
only write **markup**. No JS per cell.

**Markup (drop anywhere in a lesson):**
```html
<div class="widget runpy">
  <h3>🐍 Run it — short, problem-framed title</h3>
  <p>One line: what the reader will see and why it matters.</p>
  <textarea class="runpy-code">PYTHON SOURCE HERE</textarea>
  <pre class="runpy-expected" hidden>EXACT OFFLINE-FALLBACK OUTPUT</pre>
</div>
```

**How it works (do not re-implement):**
- `course.js` → `initRunpy()` auto-wires every `.widget.runpy`: reads the
  `.runpy-code` textarea, hides+reads `.runpy-expected`, appends Run/Reset buttons
  and a dark `.runpy-out` terminal. Exposed as `window.initRunpy`. Runs on
  DOMContentLoaded.
- `ensurePyodide()` lazy-loads Pyodide from the CDN
  (`https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js`) on first Run only —
  the page stays offline-capable until the reader clicks Run.
- stdout is captured batched; the helper re-adds the trailing `\n` per line.
- Styles (`.runpy-code`, `.runpy-ctrl`, `.runpy-out`) live in `styles.css`.

**Authoring rules for cells:**
- Keep it **short and glass-box** — print intermediate values and shapes, not just
  the final answer.
- Make it **deterministic** (`random.seed(0)`) so the printed output is stable and
  the offline fallback matches.
- The `.runpy-expected` block must be the **byte-exact** output (it's the offline
  fallback shown if Pyodide can't load). Capture it by running the *identical*
  source in `python3` locally first — never hand-write it.
- Pyodide ≈ CPython for stdlib + `random` (same Mersenne Twister) — seeded output
  matches across both. Verify anyway.
- Prefer pure-Python / stdlib so the fallback is trustworthy and load is fast.

## Verification discipline (do every time)

1. **Capture expected output in `python3` first.** Run the exact cell source
   locally; paste its stdout verbatim into `.runpy-expected`.
2. **`node --check` any new inline `<script>`** (extract, exclude `src=` and
   `application/json`). `.runpy` cells add none, but other edits might.
3. **Verify live in the preview** — and beware the **cache gotcha**: appending
   `?v=Date.now()` busts the *HTML* but **not** `course.js`, so a freshly-edited
   helper won't load in the iframe. Work around it: load the page in an iframe,
   then inject a cache-busted course.js and re-init:
   ```js
   const f=document.createElement('iframe'); document.body.appendChild(f);
   await new Promise(r=>{f.onload=r; f.src='/course/html/lesson_XX.html?cb='+Date.now();});
   const w=f.contentWindow,d=f.contentDocument;
   await new Promise((res,rej)=>{const s=d.createElement('script');
     s.src='/course/html/course.js?cb='+Date.now(); s.onload=res; s.onerror=rej; d.body.appendChild(s);});
   w.initRunpy();                                   // re-wire with fresh helper
   const c=d.querySelectorAll('.widget.runpy')[N];  // pick the cell
   [...c.querySelectorAll('button')].find(b=>/run/i.test(b.textContent)).click();
   // poll .runpy-out until non-empty + contains an expected marker (Pyodide cold-start ~4s)
   ```
   Assert the live `.runpy-out` text **equals** `.runpy-expected`.
   - `mcp__Claude_Preview__preview_start` name `course-html` (port 8014). The
     `serverId` changes on restart — `preview_list` to get the current one.
4. **Structural sanity** after restructures: `<details>` open/close balance,
   anchors still resolve (keep the `id` when wrapping a section in an expander),
   pill count / `current` / "Step N of 16" / reciprocal pagination unchanged.

## Workflow when invoked

1. Read `course-principles` (rubric + conventions) and the target page.
2. **Map the spine**: `grep` the `<h2 class="step">` headings; write the problem
   each one solves; mark where the original problem is solved and where shape goes
   slack.
3. Apply the five moves where they bite — prioritise (3) failure→fix and (5)
   demote-clutter; add (2) runnable cells at the key concepts; ensure (1) one
   running example throughout; add (4) glass-box shapes where tensors appear.
4. If the back half is a tour, re-arm it with the next felt problem (spine section
   above).
5. Verify (discipline above). Capture-in-python3 → live Pyodide check → structural
   sanity.
6. Commit with a message naming the move applied (e.g. "reframe X as failure→fix",
   "restore problem-driven spine"). Don't commit/push unless the user asked.

## Anti-patterns (this skill exists to kill these)

- Working result shown before the naive failure — no felt pain, nothing sticks.
- A paragraph *describing* a printout instead of a runnable cell that produces it.
- A new toy example per section instead of one running example.
- Tensors discussed without printed shapes.
- A back half that lists mechanics with no escalating problem.
- Reconciliation/orientation blocking the main path instead of living in an expander.
- A `.runpy-expected` fallback hand-written (drifts from real output) instead of
  captured from `python3`.
- Declaring done without the live Pyodide check (cache gotcha hides stale helpers).
