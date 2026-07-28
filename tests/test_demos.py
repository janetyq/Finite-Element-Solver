"""Every registered demo still runs.

The demos in `examples/` are the only thing exercising the plot layer, and they are
the first thing to break when an API moves -- two of them had rotted against
`BoundaryConditions.plot` and `Solution.get_values` with nothing to catch it. Each
demo runs here on a small mesh, asserting "still callable and still returns what the
registry claims", not "still correct": the numerics have their own tests.

Demos that need a person at a widget, or that are blocked on unimplemented work, name
the reason in `Demo.smoke_skip` and are skipped.
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

DEMOS = list(cli.build_registry().values())


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
    if demo.smoke_skip is not None:
        pytest.skip(demo.smoke_skip)
    if demo.smoke_requires is not None:
        pytest.importorskip(demo.smoke_requires)

    # Demos write their output relative to the working directory, so run them
    # somewhere disposable rather than leaving files in the repo.
    monkeypatch.chdir(tmp_path)

    args = [make_unit_square(10)] if demo.needs_mesh else []
    result = demo.func(*args, **demo.smoke_kwargs)

    if demo.returns_plotter:
        # cli.py shows or saves whatever comes back, so the registry's claim about the
        # return type has to hold or `run <demo>` fails on a human's screen.
        plotters = result if isinstance(result, list) else [result]
        assert plotters, f'{demo.name} is registered as returning a Plotter but returned nothing'
        assert all(isinstance(p, Plotter) for p in plotters), (
            f'{demo.name} is registered as returning a Plotter, got {type(result).__name__}'
        )
