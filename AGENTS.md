# AGENTS.md

Working guidelines for agents on this repo. `CLAUDE.md` imports this file. Do not modify `AGENTS.md` unless explicitly instructed.

## General

- Prefer quality, simplicity, robustness, scalability, and long-term maintainability over minimizing development cost or churn.
- Surface meaningful design alternatives before implementing, with a recommendation. Don't ask about choices with an obvious default.
- Report results honestly. If something is skipped, unverified, or failing, say so.
- Treat inline `TODO`s as intentional unless told otherwise. Ask before deleting them.
- If a bug blocks the ideal deliverable, ask before working around it.

## Prose

Style preferences for text checked into the repo: docstrings, comments, README, ARCHITECTURE.md, BACKLOG.md, demo text, PR titles and descriptions, commit messages. They do not apply to conversation with the user.

- Keep it short. A docstring says what a thing is and returns; a comment explains a mechanism the code cannot show. Design rationale goes in the PR.
- Write for someone reading the current code with no memory of earlier versions. Don't narrate what changed or contrast with how it used to be.
- Plain, direct sentences. Avoid rhetorical devices that draw attention to the writing (em-dash asides, "X, not Y" framing, trailing qualifiers).
- A test's name or docstring states the claim it checks.
- A PR title or commit subject is a short imperative summary of the change, like "Unify problem construction and solve API".

## Git

- Unless explicitly told otherwise, do all work on a dedicated branch. Never commit directly to `main`.
- Use one branch per logical effort.
- Keep `main` linear: squash-merge the branch into a single commit, then delete the branch. Never use `git merge --no-ff`.
- Don't delete a merged branch while another PR is stacked on it: retarget the child to `main` first (`gh pr edit <child> --base main`), then rebase it onto `main` (`git rebase --onto origin/main <base> <child>`) and force-push. A deleted base branch auto-closes its child PRs, unreopenably.
- Squash-merge through GitHub (`gh pr merge --squash`), not locally, so the PR shows as merged.
- Parallel efforts run in linked worktrees under `.claude/worktrees/<branch>`; keep the primary checkout on `main` (never do feature work there) and give every worktree its own branch off `main`. A branch is then only ever checked out in one place, so `git checkout` and `gh pr merge --delete-branch` never collide across worktrees.
- Stage explicit paths only. Never use `git add -A` or `git add .`.

## Verification

- Every PR runs `uv run pytest`, `uv run ruff check`, and `uv run pyright` in CI, and all must pass before merging. Rely on CI for the full suite rather than running it locally; run only targeted checks locally (a single test, or `ruff`/`pyright` on changed files) when useful. If CI has not run by merge time, ask.
- Use `uv`; don't invoke `pip` or bare `python`. The dev env is pinned to 3.11 (`.python-version`); the package itself supports 3.10+.
- Add test coverage before refactoring untested code.
- New physics or numerics ships with a test against an analytic or manufactured solution.
- Demos tell a physical story, one per story. Show a new feature by extending an existing demo; propose a new one only when no existing demo can carry it.
- When fixing a defect, search for the same pattern elsewhere and fix every instance in the same PR.
- Keep the MMS convergence test (`tests/test_convergence.py`) passing through numerical refactors; don't weaken it to accommodate changes.

## BACKLOG.md and ARCHITECTURE.md

- `BACKLOG.md` tracks only open work. Remove completed items as part of the effort that completes them.
- `ARCHITECTURE.md` describes the object model. Update it in the PR that changes it.
