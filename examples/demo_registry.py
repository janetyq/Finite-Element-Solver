"""Shared descriptors used by each example file's `DEMOS` registry and by cli.py.

`Demo` is what the registry lists; `DemoResult` is what running one gives back. The
split matters: a demo never shows, saves, or prints anything itself, it returns what it
produced and the caller decides. That is what lets `run`, the gallery, and
`tests/test_demos.py` treat every demo the same way instead of special-casing the ones
that used to write a GIF into the working directory or print a table to stdout.
"""
import functools
import inspect
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
    # An optional dependency the demo needs; it is skipped where that is absent.
    smoke_requires: str | None = None
    # Cheaper arguments for `tests/test_demos.py`, which runs every demo on every
    # commit. Only a few demos set it, and each says why: the rest are cheap enough
    # either way. Everyone else -- the CLI and the gallery -- runs demos exactly as
    # written, so a demo's defaults are what a reader actually sees.
    smoke_kwargs: dict[str, Any] = field(default_factory=dict)

    def source(self) -> str:
        """The demo function's own source, for readers who came for the code.

        Unwraps `functools.partial` the way `description` does, so a preconfigured demo
        shows the function that was bound rather than failing to have a source at all.
        The bound arguments are not shown: they are the gallery's cheaper settings, not
        part of what the demo is saying.
        """
        func = self.func
        while isinstance(func, functools.partial):
            func = func.func
        try:
            return inspect.getsource(func)
        except OSError:      # no source available -- a REPL-defined or C function
            return ''

    def description(self) -> str:
        """The demo's docstring on one line, for `list` and for its gallery page.

        Unwraps `functools.partial`, so binding arguments to preconfigure a demo does
        not replace its description with partial's own docstring.
        """
        func = self.func
        while isinstance(func, functools.partial):
            func = func.func
        doc = inspect.getdoc(func)
        return ' '.join(doc.split()) if doc else '(no description)'
