"""Shared pytest setup and fixtures.

The project's modules import matplotlib (via Plotter). Force the non-interactive
Agg backend so importing/exercising them never tries to open a window — required
for headless/CI runs.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

# `examples/` is a directory of scripts rather than a package (cli.py imports its
# siblings by bare name), so it goes on the path once here for every test that
# imports a demo or the MMS studies in `examples/mms.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

from fem.mesh.structured import box_mesh


@pytest.fixture
def make_unit_square():
    """Factory fixture: build a fresh Mesh on the unit square [0,1]^2.

    Returns a callable so each test can pick its own resolution, e.g.
    ``mesh = make_unit_square(20)``. Geometry only -- a solver builds its own
    FunctionSpace, so there is no assembled state to bleed between tests.
    """
    def _make(n=20):
        return box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    return _make
