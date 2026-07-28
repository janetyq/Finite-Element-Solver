"""Shared `Demo` descriptor used by each example file's `DEMOS` registry and by cli.py."""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Demo:
    name: str
    func: Callable
    needs_mesh: bool = True      # cli.py loads --mesh and passes it as the first arg
    returns_plotter: bool = True  # False: demo manages its own display/output; --save is rejected
    # Why this demo cannot run unattended, or None if it can. `tests/test_demos.py` runs
    # every demo headlessly so a moved API surfaces there rather than on a human's screen;
    # a demo that needs a person at a widget, or that is blocked on unimplemented work,
    # names the reason here and is skipped. Kept beside the demo rather than in a list
    # inside the test, so whoever changes the demo sees the claim.
    smoke_skip: str | None = None
    # An optional dependency the demo needs, skipped when absent. Distinct from
    # `smoke_skip`: this demo can run unattended, just not on every install.
    smoke_requires: str | None = None
    # Overrides for that headless run: cheaper sizes, and output paths left relative so
    # they land in the test's temporary directory instead of the repo.
    smoke_kwargs: dict[str, Any] = field(default_factory=dict)
