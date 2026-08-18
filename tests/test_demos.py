"""Every registered demo still runs.

The demos in `examples/` are the only thing exercising the plot layer, and they are
the first thing to break when an API moves -- two of them had rotted against
`BoundaryConditions.plot` and `Solution.get_values` with nothing to catch it. Each
demo runs here on a small mesh, asserting "still callable and still returns what the
registry claims", not "still correct": the numerics have their own tests.

A demo needing an optional dependency names it in `Demo.smoke_requires` and is skipped
where that is absent; `Demo.smoke_kwargs` supplies the cheapest arguments that still
exercise the code.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from fem.plot.plotter import Plotter

# `examples/` is a directory of scripts rather than a package -- cli.py imports its
# siblings by bare name -- so it has to be on the path to be importable the same way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

import cli  # noqa: E402
from demo_registry import DemoResult  # noqa: E402

DEMOS = list(cli.build_registry().values())


def test_interactive_is_rejected_where_there_is_no_such_mode(monkeypatch, capsys):
    """`--interactive` dispatches on the demo's signature. Asking for it where the demo
    has no widget mode should say so and name the ones that do, not fail inside the demo
    with a TypeError."""
    monkeypatch.setattr(sys, 'argv', ['cli.py', 'run', 'poisson', '--interactive'])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code != 0
    message = capsys.readouterr().err
    assert 'has no interactive mode' in message
    assert 'outline_zoo' in message


@pytest.fixture(autouse=True)
def close_figures():
    """Demos build Plotters and never close them; without this the run leaks every
    figure it opens and matplotlib starts warning partway through."""
    yield
    plt.close('all')


# An animation demo builds a FuncAnimation that only renders on show()/save(), so
# under Agg it is always collected unrendered. Expected here, not a symptom.
@pytest.mark.filterwarnings('ignore:Animation was deleted without rendering anything')
@pytest.mark.parametrize('demo', DEMOS, ids=lambda demo: demo.name)
def test_demo_runs(demo, make_unit_square, tmp_path, monkeypatch):
    if demo.smoke_requires is not None:
        pytest.importorskip(demo.smoke_requires)

    # Demos write their output relative to the working directory, so run them
    # somewhere disposable rather than leaving files in the repo.
    monkeypatch.chdir(tmp_path)

    # The demo's own domain is replaced by a tiny one: this asserts "still callable",
    # and meshing a beam at demo resolution is most of what a demo costs.
    args = [make_unit_square(10)] if demo.domain is not None else []
    result = demo.func(*args, **demo.smoke_kwargs)

    assert isinstance(result, DemoResult), (
        f'{demo.name} returned {type(result).__name__}; demos return a DemoResult so the '
        'caller decides what to show, save, or print'
    )
    assert all(isinstance(f.plotter, Plotter) for f in result.figures), (
        f'{demo.name} put a non-Plotter in a Figure'
    )
    assert all(f.caption.strip() for f in result.figures), (
        f'{demo.name} has a figure with no caption; the gallery has nothing to say about it'
    )
    # More than one figure means the filenames come from slugs, so they have to exist
    # and be distinct or the figures overwrite each other on save.
    if len(result.figures) > 1:
        slugs = [f.slug for f in result.figures]
        assert all(slugs) and len(set(slugs)) == len(slugs), (
            f'{demo.name} has {len(slugs)} figures needing distinct slugs, got {slugs}'
        )

    # The point of the whole contract: a demo that yields nothing appears nowhere, which
    # is how a demo rendering a blank panel stayed invisible before.
    assert result.figures or result.text or result.artifacts, (
        f'{demo.name} produced no figures, no text, and no files'
    )


@pytest.mark.parametrize('demo', DEMOS, ids=lambda demo: demo.name)
def test_demo_declares_a_known_section(demo):
    """Sections are declared per demo, so a new one can silently land in "Other". The
    gallery still renders it -- this is what stops that going unnoticed."""
    from gallery import SECTIONS

    assert demo.section in SECTIONS, (
        f'{demo.name} declares section {demo.section!r}; expected one of {SECTIONS}'
    )
