"""Shared descriptors used by each example file's `DEMOS` registry and by cli.py.

`Demo` is what the registry lists; `DemoResult` is what running one gives back. The
split matters: a demo never shows, saves, or prints anything itself, it returns what it
produced and the caller decides, so `run`, the gallery, and `tests/test_demos.py`
treat every demo the same way.
"""
import functools
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh
    from fem.plot.plotter import Plotter


@dataclass
class Figure:
    """One figure a demo produced, with the caption that belongs beside it.

    `slug` names the file when a demo produces more than one figure, so they save as
    `wave-animation.png` / `wave-snapshots.png` rather than `wave_0` / `wave_1`.

    `thumbnail` nominates this figure as the demo's gallery card instead of the first.
    Most demos return their result first and their setup after, the order a reader
    arriving from a card wants, so the default suffices; it is for a demo like
    `stress_concentration`, whose card should be the final stress, not its outline.

    `setup` marks a figure as how the problem was posed (the conditions imposed, the
    state it started from) rather than what came out. The gallery collects those at the
    foot of the page, beside the source.
    """
    plotter: 'Plotter'
    caption: str
    slug: str = ''
    thumbnail: bool = False
    setup: bool = False

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

    Empty is legal for none of these individually (a demo with no figures may still
    have `text`), but a demo yielding nothing at all shows up nowhere, which
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
    # How to build the domain this demo runs on, passed as its first argument. `None`
    # means the demo takes no mesh: it builds its own, or has nothing to solve on.
    # A factory rather than a Mesh so nothing is meshed until the demo is actually run,
    # and per demo rather than one default for all, because a cantilever wants a beam
    # and a projection wants a fine square. `cli.py --mesh` overrides it.
    domain: 'Callable[[], Mesh] | None' = None
    # The gallery index heading this demo belongs under. Declared rather than inferred
    # from the module it is written in: which file a demo lives in is a fact about the
    # code's layout, and it disagreed with where a reader would look for it.
    section: str = ''
    # An optional dependency the demo needs; it is skipped where that is absent.
    smoke_requires: str | None = None
    # Cheaper arguments for `tests/test_demos.py`, which runs every demo on every
    # commit. Only a few demos set it, and each says why: the rest are cheap enough
    # either way. Everyone else (the CLI and the gallery) runs demos exactly as
    # written, so a demo's defaults are what a reader actually sees.
    smoke_kwargs: dict[str, Any] = field(default_factory=dict)

    def _unwrapped(self) -> Callable[..., DemoResult]:
        """The demo function itself, from behind any `functools.partial` around it.

        Binding arguments to preconfigure a demo must not change what the demo is:
        a partial has its own docstring, its own module, and no source at all.
        """
        func = self.func
        while isinstance(func, functools.partial):
            func = func.func
        return func

    def source(self) -> str:
        """The demo function's own source, for readers who came for the code.

        The bound arguments of a preconfigured demo are not shown: they are the
        gallery's cheaper settings, not part of what the demo is saying.
        """
        try:
            return inspect.getsource(self._unwrapped())
        except OSError:      # no source available: a REPL-defined or C function
            return ''

    def description(self) -> str:
        """The demo's docstring on one line, for `list` and for its gallery page."""
        doc = inspect.getdoc(self._unwrapped())
        return ' '.join(doc.split()) if doc else '(no description)'
