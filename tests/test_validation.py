"""Tests for the guardrails around half-implemented / easily-misused surfaces.

These lock in that the gated features fail loudly (with a clear error) rather
than silently returning wrong or empty results.
"""
import numpy as np
import pytest

from fem.energies import NeohookeanEnergyDensity
from fem.boundary import BoundaryConditions
from fem.solver import Solver, Equation


def test_neohookean_is_gated():
    """The unfinished Neohookean material must raise, not silently do nothing."""
    density = NeohookeanEnergyDensity(E=200, nu=0.3)
    with pytest.raises(NotImplementedError):
        density.set_grad_u(np.zeros((2, 2)))


def test_wave_rejects_dirichlet_bcs(make_unit_square):
    """solve_wave ignores Dirichlet BCs, so it must refuse them rather than
    return a solution that violates them."""
    femesh = make_unit_square(8)
    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", [int(femesh.boundary_idxs[0])], [0])

    n = len(femesh.vertices)
    eq = Equation(
        "wave",
        {"u_initial": np.zeros(n), "dudt_initial": np.zeros(n),
         "c": 1, "dt": 0.02, "iters": 3},
        dim=1,
    )
    with pytest.raises(NotImplementedError):
        Solver(femesh, eq, bc).solve()


def test_check_rejects_interior_bc(make_unit_square):
    """A BC placed on a non-boundary vertex is a modeling error and must be caught."""
    femesh = make_unit_square(8)
    interior = (set(range(len(femesh.vertices))) - set(int(i) for i in femesh.boundary_idxs)).pop()

    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", [interior], [0])
    with pytest.raises(ValueError):
        bc.check()


def test_check_rejects_dirichlet_neumann_overlap(make_unit_square):
    """A vertex fixed by Dirichlet silently ignores its Neumann load; flag it."""
    femesh = make_unit_square(8)
    node = int(femesh.boundary_idxs[0])

    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", [node], [0])
    bc.add("neumann", [node], [0])
    with pytest.raises(ValueError):
        bc.check()
