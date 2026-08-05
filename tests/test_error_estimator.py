"""Tests for the residual-based error estimator."""
from math import e

import numpy as np
import pytest

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.equations import Poisson
from fem.regions import everywhere
from fem.solver import Solver


def test_error_estimator_returns_correct_shape(make_unit_square):
    """The estimator must return one non-negative finite value per element."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=1.0), bc)
    solver.solve()

    eta = solver.equation.error_estimate(solver)

    assert len(eta) == len(mesh.elements)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)


def test_error_estimator_linear_solution_small_jumps(make_unit_square):
    """A problem with linear exact solution has near-zero gradient jumps."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), lambda p: p[0])
    solver = Solver(mesh, Poisson(source=None), bc)
    solver.solve()

    eta = solver.equation.error_estimate(solver)

    # Linear u = x has constant gradient, so jumps are numerical only
    assert np.all(eta < 1e-10)


def test_error_estimator_concentrates_near_peak(make_unit_square):
    """Error should be largest near a peaked source."""
    mesh = make_unit_square(10)

    def peaked_source(point):
        a = 50
        x, y = point - 0.5
        r2 = x**2 + y**2
        return 4*a*a*(1-a*r2)*e**(-a*r2)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=peaked_source), bc)
    solver.solve()

    eta = solver.equation.error_estimate(solver)
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)

    near_center = center_dist < 0.15
    far_from_center = center_dist > 0.35
    assert eta[near_center].mean() > eta[far_from_center].mean()


def test_error_estimator_requires_solved_system(make_unit_square):
    """Calling error_estimate before solve raises."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=1.0), bc)

    with pytest.raises(ValueError, match='requires a solved system'):
        solver.equation.error_estimate(solver)


def test_adaptive_refinement_with_error_estimator(make_unit_square):
    """The full loop: estimator drives refinement, mesh grows near the source."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=lambda p: 10.0 if np.linalg.norm(p - 0.5) < 0.1 else 0.0)
    solver = Solver(mesh, equation, bc)

    n_before = len(mesh.elements)
    AdaptiveRefinement(
        solver,
        equation.error_estimate,
        max_triangles=300,
        max_iters=5,
    ).run()

    assert len(solver.mesh.elements) > n_before

    # Verify refinement concentrates near the source
    centroids = solver.mesh.vertices[solver.mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    near_center = (center_dist < 0.2).sum()
    far_away = (center_dist > 0.35).sum()
    # The source is localised, so more refinement happens near the center
    assert near_center > far_away * 0.3
