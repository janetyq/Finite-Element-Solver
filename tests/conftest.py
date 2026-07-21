"""Shared pytest setup and fixtures.

The project's modules import matplotlib (via Plotter). Force the non-interactive
Agg backend so importing/exercising them never tries to open a window — required
for headless/CI runs.
"""
import matplotlib

matplotlib.use("Agg")

import pytest

from fem.mesh.generation import create_rect_mesh


@pytest.fixture
def make_unit_square():
    """Factory fixture: build a fresh Mesh on the unit square [0,1]^2.

    Returns a callable so each test can pick its own resolution, e.g.
    ``mesh = make_unit_square(20)``. Geometry only -- a solver builds its own
    FunctionSpace, so there is no assembled state to bleed between tests.
    """
    def _make(n=20):
        return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    return _make
