---
name: fix-github-issues
description: Read open GitHub issues on this repo, fix them in the working tree, and mark them as fixed (commit referencing the issue + close with a summary comment). Use whenever the user says "fix the GitHub issues", "address the issues", "triage issues", "work through the issue tracker", or points at a specific issue number/URL. Works one issue at a time by default; can batch when asked. Requires the `gh` CLI authenticated for the repo.
---

# fix-github-issues — read → fix → mark fixed

The job: turn open GitHub issues into landed fixes. For each issue you read it,
reproduce/locate the problem in the working tree, make the **minimal correct
fix** following the repo's conventions, verify it, commit referencing the issue,
push, and close the issue with a short comment that says what changed and links
the commit. "Marked as fixed" = the issue is **closed** with a comment, backed by
a real commit — never close an issue you didn't actually fix.

## Prerequisites

- `gh auth status` must show logged in. If not, stop and tell the user to run
  `gh auth login`.
- Confirm the target repo: `gh repo view --json nameWithOwner,defaultBranchRef`.
- Read `CLAUDE.md` (repo + user) for git rules and coding style, and any
  domain skill (e.g. `course-principles`) before editing files it governs.

## Inputs

- **No arg** → list all open issues, then work the oldest/most-actionable first
  (confirm with the user which ones if there are several).
- **An issue number or URL** → go straight to that one.
- **"all" / "batch"** → loop over every open issue; still fix + verify each
  individually, one commit per issue.

## Workflow (per issue)

1. **Read it fully.** `gh issue view <N> --json title,body,labels,comments`.
   Read the body AND every comment — the real ask is often in a comment.
2. **Resolve the references.** Issues frequently link to a blob URL like
   `…/blob/<sha>/<path>#L<line>`. Parse out `<path>` and `<line>`; open that file
   at that line in the working tree. The `<sha>` is a *snapshot* — the line may
   have moved, so locate by content, not by line number alone. Screenshots in the
   body usually show the broken output; read the body text for the exact symptom.
3. **Scope the blast radius.** The reported spot is usually one instance of a
   pattern. `grep` for the same bug everywhere before fixing (e.g. a bad URL on
   one page is almost always on every page). Fix the whole class, not just the
   cited line — but say so in the issue comment.
4. **Fix minimally and in-convention.** Smallest diff that fully resolves it.
   Match surrounding style. Don't refactor unrelated code. If the fix needs a
   decision the issue doesn't settle (e.g. *which* replacement URL), use the
   issue's stated preference; if none, pick the obviously-correct option and note
   it in the comment.
5. **Verify before claiming done.** Run whatever proves the fix:
   - HTML/JS course pages: extract every inline `<script>` (not `src=`, not
     `application/json`) and `node --check`; validate `application/json` blocks;
     re-check nav/pagination invariants if you touched structure (see
     `course-principles`).
   - Links: confirm the new target actually exists/resolves (e.g. the file is in
     the repo; the referenced notebooks live in `course/notebooks/`). Don't trade
     a broken link for a 404.
   - Python: `python3 -m py_compile`. Notebooks: confirm still valid JSON.
6. **Commit, referencing the issue.** Stage **by name** (never `git add -A`).
   Message: a concise subject, a one-line body of what/why, and a closing trailer
   `Fixes #<N>` so the push auto-closes the issue. End with the repo's
   `Co-Authored-By:` trailer. Branch first if the repo rules forbid committing on
   the default branch.
7. **Push, then confirm closed.** After push, verify with
   `gh issue view <N> --json state`. If "Fixes #N" didn't auto-close (e.g. you
   committed to a non-default branch / PR flow), close explicitly:
   `gh issue close <N> --comment "<summary>"`. Either way, leave a comment that
   states the fix, the scope (how many files/spots), and the commit SHA:
   `gh issue comment <N> --body "Fixed in <sha>: <one-line what changed>. <scope note>."`
8. **Report** to the user: issue title, what was wrong, what you changed (file
   count), how you verified, and the issue's new state.

## Hard rules

- **Never close an issue without a real fix** committed and pushed. No
  rubber-stamping.
- **Honour the repo git constraints** (from CLAUDE.md / session): stage files by
  name; never `git add -A`/`.`; never `--no-verify`; never force-push; never
  touch git config. Use the project's commit `Co-Authored-By:` trailer.
- **Don't commit unrelated changes.** If the working tree has unstaged edits from
  other work, stage only the files your fix touched.
- **Fix the class, cite the instance.** When the bug is a repeated pattern, fix
  all occurrences and say so in the closing comment.
- **One issue → one commit → one close**, unless the user asks to batch into a
  single commit.
- If an "issue" is actually a feature request or needs product judgement, don't
  silently implement — summarise it for the user and ask before building.

## Worked example — issue #2 "Notebook URL bug"

Symptom (from body): course pages link notebooks via
`http://localhost:8011/notebooks/<nb>.ipynb` — a dev-only Jupyter URL that 404s
for anyone on the published GitHub Pages site. Reporter's fix: point at the
notebooks under `…/tree/main/course/notebooks` so GitHub renders them.

Resolution:
- `grep -rn 'localhost:8011' course/html/*.html` → ~30 hits across 14 pages (not
  just the cited `index.html#L370`).
- Confirm every referenced `<nb>.ipynb` exists in `course/notebooks/` (no 404s).
- Replace each `http://localhost:8011/notebooks/<nb>.ipynb` with
  `https://github.com/<owner>/<repo>/blob/main/course/notebooks/<nb>.ipynb`
  (GitHub renders `.ipynb` in blob view).
- Smoke-test inline JS still parses; spot-check a couple of rewritten links.
- Commit `Fixes #2`, push, confirm issue auto-closed, add a comment noting the
  full scope (all pages, N links).
