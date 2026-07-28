"""Shared descriptors used by each example file's `DEMOS` registry and by cli.py.

`Demo` is what the registry lists; `DemoResult` is what running one gives back. The
split matters: a demo never shows, saves, or prints anything itself, it returns what it
produced and the caller decides. That is what lets `run`, the gallery, and
`tests/test_demos.py` treat every demo the same way instead of special-casing the ones
that used to write a GIF into the working directory or print a table to stdout.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fem.plot.plotter import Plotter


@dataclass
class Figure:
    """One figure a demo produced, with the caption that belongs beside it.

    `slug` names the file when a demo produces more than one figure, so they save as
    `wave-animation.png` / `wave-snapshots.png` rather than `wave_0` / `wave_1`.
    """
    plotter: 'Plotter'
    caption: str
    slug: str = ''

    @property
    def animated(self) -> bool:
        """Whether this figure only means anything in motion.

        An animation renders on `show()` and nowhere else, so anything saving to a
        still image skips it and uses the demo's static figures instead.
        """
        return bool(self.plotter.anims)


@dataclass
class DemoResult:
    """Everything one demo produced.

    Empty is legal for none of these individually -- a demo with no figures may still
    have `text` -- but a demo yielding nothing at all shows up nowhere, which
    `tests/test_demos.py` rejects.
    """
    figures: list[Figure] = field(default_factory=list)
    text: str | None = None              # tables and printed summaries
    artifacts: list[Path] = field(default_factory=list)  # files the demo wrote

    @property
    def still_figures(self) -> list[Figure]:
        """The figures a still image can capture."""
        return [figure for figure in self.figures if not figure.animated]


@dataclass
class Demo:
    name: str
    func: Callable[..., DemoResult]
    needs_mesh: bool = True      # cli.py loads --mesh and passes it as the first arg
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
