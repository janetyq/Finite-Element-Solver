# AGENTS.md

Working guidelines for agents on this repo. `CLAUDE.md` imports this file, so this
is the only place the content lives — edit here.

## Decision-making

- When making technical decisions, do not give much weight to development cost.
  Instead, prefer quality, simplicity, robustness, scalability and long term
  maintainability.
- Churn is acceptable in service of a better long-term design. Say so when a change
  is large, but don't pick the worse design to avoid the work.
- Surface genuine design forks before implementing them, with a recommendation.
  Don't ask about choices that have an obvious default.

## Git workflow

- **Branch per logical effort.** Small atomic commits *on the branch*, one logical
  change each, conventional prefixes (`fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
- **`main` stays linear: squash-land the branch as a single commit**, then delete the
  branch. Not `git merge --no-ff` — merge bubbles were reviewed and rejected.
- **Stage explicit paths. Never `git add -A` / `git add .`** — a blanket add once swept
  the user's unrelated uncommitted work into a commit.
- **Never push or tag without an explicit ask.** Same for anything else outward-facing
  (deleting remote branches, opening PRs). Landing locally is fine; publishing is not.

## Verification

- `uv run pytest` and `uv run ruff check` must both pass before landing. The project
  uses uv and Python 3.11; don't invoke `pip` or a bare `python`.
- Add coverage for an untested path *before* refactoring it. The MMS convergence test
  in `tests/test_convergence.py` is the safety net for solver changes — keep it green
  through numerical refactors rather than adjusting it to fit.
- Report results honestly: if something is skipped, unverified, or failing, say so.

## Code conventions

- **Fail loudly.** Gate half-built features with `NotImplementedError` rather than
  returning a quietly wrong answer. Prefer `raise` over `assert` for real capability
  boundaries, so they survive `python -O`.
- **Prefer typed over stringly-typed.** Enums (`BCType`, `PlotMode`) and per-PDE classes
  (`Poisson`, `Heat`, `LinearElastic`) instead of magic strings and dict-of-params
  dispatch. Typing the data is what makes dead parameters visible.
- **`logging`, not `print`.** Per-module `logging.getLogger(__name__)`; the library stays
  quiet via the `NullHandler` in `fem/__init__.py`. Applications configure handlers.
- **Comments explain *why*.** Match the surrounding density and idiom; a comment that
  restates the code is noise.

## Respect existing intent

- **Inline `TODO`s are deliberate — leave them.** They mark known open work.
- **Commented-out blocks are parked design ideas, not dead code.** The residual
  estimator in `fem/solver.py` and the gradient/hessian checks in `fem/energy_solver.py`
  are intentional; do not "tidy" them away.
- If something looks like cruft, ask rather than delete.

## BACKLOG.md

The living list of still-open work. Only open items — no notes about what was fixed.
When an effort closes an item, remove it in that same effort.
