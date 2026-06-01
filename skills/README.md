# Skills

Reusable **Claude Code skills** for this repo — each `SKILL.md` is a packaged set
of instructions the agent loads when you ask for that task.

| Skill | What it does |
|---|---|
| [`fix-github-issues`](fix-github-issues/SKILL.md) | Read open GitHub issues → fix them → mark fixed (commit + close + comment). |
| [`course-build-blog`](course-build-blog/SKILL.md) | Write a CxAI-style "how we built the Mini PRAGMA course" blog post, grounded in the real artifacts. |
| `course-principles` | Principles + scoring rubric for the course lessons. The canonical file lives at [`course/agent/SKILL.md`](../course/agent/SKILL.md). |

## Why these are *here* and not only in `.claude/`

Claude Code only **discovers** skills under `.claude/skills/`, but that folder is
hidden (dot-prefixed), so the files are easy to miss on GitHub. To get both —
discovery *and* visibility — the real, readable files live here in `skills/`, and
`.claude/skills/<name>/SKILL.md` is a **symlink** pointing back to them. Edit the
file here; the symlink follows automatically.
