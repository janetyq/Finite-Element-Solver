"""Shared pytest setup and fixtures.

The project's modules import matplotlib (via Plotter). Force the non-interactive
Agg backend so importing/exercising them never tries to open a window — required
for headless/CI runs.
"""
import matplotlib

matplotlib.use("Agg")

import pytest

from fem.mesh.generation import create_rect_mesh
from fem.mesh.femesh import FEMesh


@pytest.fixture
def make_unit_square():
    """Factory fixture: build a fresh FEMesh on the unit square [0,1]^2.

    Returns a callable so each test can pick its own resolution, e.g.
    ``femesh = make_unit_square(20)``. A fresh mesh per call avoids solver
    state (assembled matrices, n_components) bleeding between tests.
    """
    def _make(n=20):
        base = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
        return FEMesh(base.vertices, base.elements, base.boundary)

    return _make
