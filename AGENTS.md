# AGENTS.md

Working guidelines for agents on this repo. `CLAUDE.md` imports this file. Do not modify `AGENTS.md` unless explicitly instructed.

## General

- Prefer quality, simplicity, robustness, scalability, and long-term maintainability over minimizing development cost or churn.
- Surface meaningful design alternatives before implementing, with a recommendation. Don't ask about choices with an obvious default.
- Comments should be concise and carry only what the next reader needs to understand and use the code correctly, not why you wrote it that way.
- Report results honestly. If something is skipped, unverified, or failing, say so.
- Treat inline `TODO`s as intentional unless told otherwise. Ask before deleting them.

## Git

- Unless explicitly told otherwise, do all work on a dedicated branch. Never commit directly to `main`.
- Use one branch per logical effort.
- Keep `main` linear: squash-merge the branch into a single commit, then delete the branch. Never use `git merge --no-ff`.
- Squash-merge through GitHub (`gh pr merge --squash`), not locally, so the PR shows as merged.
- Stage explicit paths only. Never use `git add -A` or `git add .`.
- Don't push, tag, open PRs, or perform other outward-facing actions unless explicitly asked.

## Verification

- Before merging a PR, `uv run pytest`, `uv run ruff check`, and `uv run pyright` must all pass. It is not necessary to run these tests after small changes.
- Use `uv`; don't invoke `pip` or bare `python`. The dev env is pinned to 3.11 (`.python-version`); the package itself supports 3.10+.
- Add test coverage before refactoring untested code.
- Keep the MMS convergence test (`tests/test_convergence.py`) passing through numerical refactors; don't weaken it to accommodate changes.

## BACKLOG.md

Tracks only open work. Remove completed items as part of the effort that completes them.
