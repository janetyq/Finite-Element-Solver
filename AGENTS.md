# AGENTS.md

Working guidelines for agents on this repo. `CLAUDE.md` imports this file. Do not modify `AGENTS.md` unless explicitly instructed.

## General

- Prefer quality, simplicity, robustness, scalability, and long-term maintainability over minimizing development cost or churn.
- Surface meaningful design alternatives before implementing, with a recommendation. Don't ask about choices with an obvious default.
- Report results honestly. If something is skipped, unverified, or failing, say so.
- Treat inline `TODO`s as intentional unless told otherwise. Ask before deleting them.
- When a bug blocks the ideal deliverable, say so before working around it, so the choice between the workaround and fixing the root cause is made deliberately.

## Prose

Applies to docstrings, comments, README/ARCHITECTURE, demo text, and PR titles and descriptions.

- A docstring says what a thing is and returns; a comment says the mechanism. Lead with the one-line what.
- Write for the next reader of the code as it stands. No history: never describe what the code used to do or why it changed ("previously", "no longer", "now", "instead of", "rather than").
- No contrast phrasing: state what the code does and drop the negated alternative ("X, not Y"). This includes titles.
- No design arguments or emphatic asides in code. The case for a choice belongs in the PR; if the next reader needs a constraint the code cannot show, state the constraint in one sentence.
- Simple language. No em-dashes; rewrite an aside as its own sentence.
- Tests carry a claim: the name or docstring states what would break if the test failed.

## Git

- Unless explicitly told otherwise, do all work on a dedicated branch. Never commit directly to `main`.
- Use one branch per logical effort.
- Keep `main` linear: squash-merge the branch into a single commit, then delete the branch. Never use `git merge --no-ff`.
- Don't delete a merged branch while another PR is stacked on it: retarget the child to `main` first (`gh pr edit <child> --base main`), then rebase the child onto `main` (`git rebase --onto origin/main <base> <child>`) and force-push so its diff shows only its own changes. A deleted base branch auto-closes its child PRs, unreopenably.
- Squash-merge through GitHub (`gh pr merge --squash`), not locally, so the PR shows as merged.
- Parallel efforts run in linked worktrees under `.claude/worktrees/<branch>`; keep the primary checkout on `main` (never do feature work there) and give every worktree its own branch off `main`. A branch is then only ever checked out in one place, so `git checkout` and `gh pr merge --delete-branch` never collide across worktrees.
- Stage explicit paths only. Never use `git add -A` or `git add .`.

## Verification

- Every PR runs `uv run pytest`, `uv run ruff check`, and `uv run pyright` in CI, and all must pass before merging. Rely on CI for the full suite rather than running it locally; run only targeted checks locally (a single test, or `ruff`/`pyright` on changed files) when useful. If CI has not run on a PR at merge time, run the full suite locally then and say so in the PR.
- Use `uv`; don't invoke `pip` or bare `python`. The dev env is pinned to 3.11 (`.python-version`); the package itself supports 3.10+.
- Add test coverage before refactoring untested code.
- Keep the MMS convergence test (`tests/test_convergence.py`) passing through numerical refactors; don't weaken it to accommodate changes.

## BACKLOG.md and ARCHITECTURE.md

- `BACKLOG.md` tracks only open work. Remove completed items as part of the effort that completes them.
- `ARCHITECTURE.md` is the overview of the object model. Update it in the same PR that changes what it describes.
