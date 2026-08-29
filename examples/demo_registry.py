"""Shared descriptors used by each demo package's `DEMO` and by cli.py.

`Demo` is what the registry lists; `DemoResult` is what running one gives back. The
split matters: a demo never shows, saves, or prints anything itself, it returns what it
produced and the caller decides, so `run`, the gallery, and `tests/test_demos.py`
treat every demo the same way.
"""
import ast
import functools
import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from collections.abc import Sequence
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

    `frames` is how many frames the gallery plays of an animation; None is the gallery's
    default budget. A short run whose every step matters asks for all of them.

    `body` is a longer explanation, one string per paragraph, that renders as prose
    under the figure for a demo whose reading needs more than a caption. The page
    decides the paragraph breaks, so each string is just its own text; the caption
    stays the one-line summary beside the image.
    """
    plotter: 'Plotter'
    caption: str
    slug: str = ''
    thumbnail: bool = False
    setup: bool = False
    frames: int | None = None
    body: list[str] = field(default_factory=list)

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


@dataclass(frozen=True)
class _ModuleByName:
    """A `show_source` module in transit through pickle; see `Demo.__getstate__`."""
    name: str


def _source_of(obj: Callable[..., Any] | ModuleType) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):      # no source: a REPL-defined, C, or built-in object
        return ''


def _split_docstring(module_source: str) -> tuple[list[str], str]:
    """A module's docstring as paragraphs, and its source from the line after it.

    The docstring is what the module says about itself, and the gallery renders it as
    prose above the code rather than as the first thing inside the code block.
    """
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return [], module_source
    doc = ast.get_docstring(tree)
    if doc is None or not tree.body:
        return [], module_source
    paragraphs = [' '.join(p.split()) for p in doc.split('\n\n') if p.strip()]
    end = tree.body[0].end_lineno or 0
    lines = module_source.splitlines(keepends=True)[end:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return paragraphs, ''.join(lines)


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
    # What the gallery shows as the demo's source: the part that poses and solves the
    # problem, not the part that draws it. A module (a demo split into `physics.py` and
    # `figures.py` passes its physics module), or a list of functions to show in that
    # order. `None` shows the demo function itself, which is right for a demo short
    # enough to read whole. A demo setting this also gets the module the demo function
    # lives in offered behind a fold, for a reader reproducing a figure.
    show_source: ModuleType | Sequence[Callable[..., Any]] | None = None

    # A gallery build sends each demo to a worker process by pickling it, and a module
    # does not pickle; it goes across by name and is imported again on arrival.
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if isinstance(self.show_source, ModuleType):
            state['show_source'] = _ModuleByName(self.show_source.__name__)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if isinstance(state['show_source'], _ModuleByName):
            state['show_source'] = importlib.import_module(state['show_source'].name)
        self.__dict__.update(state)

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
        """The source a reader who came for the code is shown: `show_source`, or the
        demo function itself.

        The bound arguments of a preconfigured demo are not shown: they are the
        gallery's cheaper settings, not part of what the demo is saying.
        """
        if self.show_source is None:
            return _source_of(self._unwrapped())
        if isinstance(self.show_source, ModuleType):
            return _split_docstring(_source_of(self.show_source))[1]
        return '\n\n'.join(filter(None, (_source_of(f) for f in self.show_source)))

    def source_notes(self) -> list[str]:
        """What the shown module says about itself, as paragraphs: its docstring, which
        `source` leaves out so it reads once, as prose. Empty unless `show_source` is a
        module."""
        if isinstance(self.show_source, ModuleType):
            return _split_docstring(_source_of(self.show_source))[0]
        return []

    def full_source(self) -> str:
        """The module the demo function lives in, plotting included, for a demo whose
        `source` is only the part that poses and solves; empty otherwise."""
        if self.show_source is None:
            return ''
        module = inspect.getmodule(self._unwrapped())
        return _source_of(module) if module is not None else ''

    def description(self) -> str:
        """The first paragraph of the demo's docstring on one line, for `list` and for
        its gallery card. Later paragraphs are notes for a reader of the source."""
        doc = inspect.getdoc(self._unwrapped())
        if not doc:
            return '(no description)'
        return ' '.join(doc.split('\n\n', 1)[0].split())
