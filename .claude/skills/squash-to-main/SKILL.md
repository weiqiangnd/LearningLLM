---
name: squash-to-main
description: >-
  Generate a single bash block the user can copy-paste into their local
  terminal to squash all commits on a branch (default: the branch currently
  being worked on in this Claude Code session) into one commit on `main`,
  push, then delete the branch from `origin` (and locally if it happens to
  exist). The commit follows the CLAUDE.md convention — author
  `weiqiang <weiqiangnd@gmail.com>` and a `Co-Authored-By: Claude <model>`
  trailer with the model running this session. The user has NOT cloned
  the branch locally; the script fetches from origin first. **Claude
  generates the script, the user runs it.** Trigger on: "squash to main"
  "压成一个提交到 main" "把这个分支合并到 main" "squash 分支" "合到主分支"
  "压平到 main" "squash branch".
---

# Squash branch → main: script generator

The user wants a single copy-pasteable bash block that:

1. Fetches a (possibly remote-only) branch from `origin`
2. Squashes its entire diff against `main` into one commit
3. Pushes `main`
4. Deletes the branch from `origin` (and locally if a local copy exists)

This is **a generation procedure for Claude**, not a runtime tool. Claude
inspects the branch in this session's container (where the repo state lives)
and emits a self-contained script for the user's local machine.

## What Claude does when this skill is invoked

1. **Identify the branch.**
   - If the user named one, use that.
   - Otherwise default to whatever this session's working branch is —
     `git rev-parse --abbrev-ref HEAD` in this container.
2. **Inspect the diff so the commit message can summarize it.**
   - `git fetch origin main` if `origin/main` isn't already up to date
   - `git log origin/main..HEAD --reverse --pretty=format:'%h %s'` (or
     `origin/main..origin/<branch>` if not on the branch) — gives the
     per-commit theme list
   - `git diff origin/main...HEAD --stat | tail -30` — confirms file scope
3. **Synthesize the squash commit message.**
   - 1 headline + 2–6 short bullets/lines, **Chinese**, declarative tone,
     matching this repo's existing commit-log style (see `git log origin/main`
     for tone).
   - Group by intent, not by file. Don't list every file changed.
   - Mention the *why*; readers can `git diff` for the *what*.
4. **Pick the Co-Authored-By model string.**
   - Use the **actual model running this session** — read it from the system
     prompt's "model identity" section. Examples: `Opus 4.7`, `Sonnet 4.6`.
   - **Never** copy a model string from a prior commit; it goes stale.
5. **Emit one fenced bash block** with the template below filled in.
   - Don't interleave prose inside the block.
   - Don't `eval` / `source`; the user pastes literally.
   - **Don't run any part of it yourself via Bash** — the user explicitly
     said *"我来执行"*.

## Script template

The whole body is wrapped in `bash <<'SCRIPT' ... SCRIPT` so the user pastes
into an **interactive shell** without `set -euo pipefail` taking down their
session — see [Why the outer `bash <<'SCRIPT'` wrapper](#why-the-outer-bash-script-wrapper).

```bash
bash <<'SCRIPT'
set -euo pipefail

BRANCH="<branch>"  # filled by Claude

# Refuse to run on a dirty working tree — a half-staged hunk would silently
# end up squashed into the commit and pushed to main.
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is dirty; commit or stash first." >&2
  exit 1
fi

# Fetch both refs we need. The branch may be remote-only (user never
# checked it out); `main` may also be behind.
git fetch origin "$BRANCH" main

# Land on main and bring it current. --ff-only refuses if local main has
# diverged, which would otherwise create a surprising merge commit.
git checkout main
git pull --ff-only origin main

# Stage the entire branch diff without making a commit yet.
git merge --squash "origin/$BRANCH"

# Single commit with the CLAUDE.md-required author + Co-Authored-By trailer.
git commit --author="weiqiang <weiqiangnd@gmail.com>" -m "$(cat <<'EOF'
<COMMIT MESSAGE — Claude fills this in based on the branch's commits>

Co-Authored-By: Claude <MODEL> <noreply@anthropic.com>
EOF
)"

git push origin main

# Delete the branch from origin, plus any local copy that happens to exist,
# plus the stale `origin/$BRANCH` remote-tracking ref.
git push origin --delete "$BRANCH"
git branch -D "$BRANCH" 2>/dev/null || true
git remote prune origin
SCRIPT
```

**Marker choice**: outer `SCRIPT` and inner `EOF` must be different — otherwise
the first `EOF` inside the commit-message heredoc would terminate the outer
heredoc prematurely. Both are quoted (`'SCRIPT'` and `'EOF'`) so the shell
won't expand `$BRANCH` / `$(...)` until the bash subprocess actually evaluates
them — variables are interpolated in the subprocess's context, not the outer
interactive shell's.

## Why these specific choices

### Why the outer `bash <<'SCRIPT'` wrapper

If we naively paste raw `set -euo pipefail` + the rest of the lines into the
user's interactive shell, two failure modes hit:

1. **`set -e` kills the terminal on any error.** Once `set -e` is active in
   an interactive shell, the *very next* command that returns non-zero exits
   the shell. The `exit 1` in the dirty-tree guard literally closes the
   user's session — they see nothing but a logout.
2. **Continuation prompts (`>`) can desync paste buffers.** Multi-line
   constructs (`if/fi`, `$(cat <<'EOF' ... EOF)`) make bash drop into a
   `>` continuation. If the user's terminal pastes too fast or the paste
   contains stray characters, the parsing can drift and execute fragments
   out of order.

Wrapping the body in `bash <<'SCRIPT' ... SCRIPT` solves both:

- The outer shell only sees one heredoc; it reads everything until `SCRIPT`
  on a line by itself, then pipes it as stdin to a fresh `bash` subprocess.
- `set -e` and `exit 1` only affect that subprocess — failure exits the
  subprocess, the outer terminal stays alive and shows the error.
- Pasting is atomic from the shell's perspective: no continuation prompts,
  no possibility of half-executed multi-line constructs.

### Other choices

- **`merge --squash` + manual commit** (not `git rebase -i` or `git reset
  --soft`) — squash leaves a clean staged-only state on `main` so the user
  can `git diff --cached` to inspect right before committing if they want;
  rebase would require already having the branch locally.
- **`--ff-only` on the pull** — if local `main` has diverged from
  `origin/main`, fail loudly instead of creating a merge commit. The
  squash workflow assumes `main` is a clean fast-forward target.
- **`branch -D 2>/dev/null || true`** — local branch may or may not
  exist; either case should succeed silently.
- **`git remote prune origin`** at the end — removes `refs/remotes/origin/$BRANCH`
  so the user's local view matches origin.

## Output rules

- One fenced bash block. The user copies the whole thing and pastes.
- No `#!/usr/bin/env bash` shebang — they're pasting into an interactive
  shell, not saving a file.
- No `claude.ai/code/...` link in the commit (this repo's CLAUDE.md
  explicitly excludes that).
- Don't suggest force-push, amending, or `--no-verify`.
- Don't run `git push` or the script yourself. The user said they'll execute.
