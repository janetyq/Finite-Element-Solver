"""Tests for the Zienkiewicz-Zhu recovery error estimator.

The residual estimator (tests/test_error_estimator.py) is checked against
hand-derived values; a recovery estimator is checked a stronger way -- its
*effectivity index*, the ratio of the estimate to the true error, which for a good
recovery approaches 1 under refinement. The true error is available here because
the manufactured-solution machinery (fem.convergence) supplies an exact field, so
its exact flux is known in closed form and integrated against the discrete one.

The manufactured problems are interior-dominant (u = sin(pi x) sin(pi y), zero on
the boundary), so simple nodal-averaging recovery -- biased at boundaries -- is at
its best and the index is a clean check rather than a loose one.
"""
import numpy as np
import pytest

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import (
    ELASTIC_E,
    ELASTIC_NU,
    elastic_source,
    exact_gradient,
    h1_seminorm_error,
    quadrature_l2,
)
from fem.equations import LinearElastic, Poisson
from fem.estimators import recovery_estimator
from fem.materials import Enu_to_Lame
from fem.mesh.ruppert import create_rect_mesh
from fem.regions import everywhere
from fem.solver import Solver


def _poisson_source(point):
    return [2 * np.pi**2 * np.sin(np.pi * point[0]) * np.sin(np.pi * point[1])]


def _global(eta):
    """The global estimate sqrt(sum eta_K^2) from the per-element indicators."""
    return float(np.sqrt((np.asarray(eta) ** 2).sum()))


def _elastic_true_stress_error(solver):
    """||sigma_exact - sigma_h||_L2 (in-plane, Frobenius) for the elasticity MMS,
    the norm the recovered-stress estimate targets. Shares `quadrature_l2` with the
    gradient error; the Frobenius norm of the full symmetric 2x2 difference is what
    the primitive computes from the trailing tensor axes."""
    mu, lamb = Enu_to_Lame(ELASTIC_E, ELASTIC_NU)
    space = solver.space
    geometry = space.geometry_at(2)
    x, y = geometry.points[..., 0], geometry.points[..., 1]
    # sigma_exact from the manufactured u = (sin(pi x) sin(pi y), 0), plane strain.
    eps_xx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    eps_xy = 0.5 * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    sxx = (2 * mu + lamb) * eps_xx
    syy = lamb * eps_xx
    sxy = 2 * mu * eps_xy
    row0 = np.stack([sxx, sxy], axis=-1)
    row1 = np.stack([sxy, syy], axis=-1)
    sigma_exact = np.stack([row0, row1], axis=-2)          # (n_el, n_qp, 2, 2)
    sigma_h = solver.solution.stress[:, None, :2, :2]      # (n_el, 1, 2, 2), constant per element
    return quadrature_l2(geometry, sigma_exact - sigma_h)


def _solve_poisson(n):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=_poisson_source), bc)
    solver.solve()
    return solver


def _solve_elastic(n):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0])
    solver = Solver(mesh, LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU, source=elastic_source), bc)
    solver.solve()
    return solver


# -- effectivity: the estimate tracks the true error -------------------------


def test_poisson_recovery_is_asymptotically_exact():
    """The effectivity index stays bounded and tends to 1 as the mesh refines --
    the defining property of a good recovery estimator."""
    indices = []
    for n in (11, 21, 41):
        solver = _solve_poisson(n)
        eta = _global(recovery_estimator(solver.equation).estimate(solver))
        true_error = h1_seminorm_error(solver.space, solver.solution.u, exact_gradient)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)          # bounded everywhere
    assert abs(indices[-1] - 1.0) < 0.1                 # asymptotically exact
    # Refinement moves the index toward 1, it does not drift away from it.
    assert abs(indices[-1] - 1.0) <= abs(indices[0] - 1.0) + 1e-9


def test_elastic_recovery_is_asymptotically_exact():
    """The same effectivity check for the vector, coupled elastic path."""
    indices = []
    for n in (11, 21, 41):
        solver = _solve_elastic(n)
        eta = _global(recovery_estimator(solver.equation).estimate(solver))
        true_error = _elastic_true_stress_error(solver)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)
    assert abs(indices[-1] - 1.0) < 0.1


# -- shape / sanity ----------------------------------------------------------


@pytest.mark.parametrize('solve', [_solve_poisson, _solve_elastic])
def test_recovery_returns_one_nonnegative_value_per_element(solve):
    solver = solve(8)
    eta = recovery_estimator(solver.equation).estimate(solver)
    assert eta.shape == (len(solver.mesh.elements),)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)


def test_recovery_of_a_linear_field_is_near_zero(make_unit_square):
    """A globally linear solution has constant gradient, so its recovery equals it
    and the estimate vanishes -- the patch test for a recovery estimator."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), lambda p: p[0])
    solver = Solver(mesh, Poisson(source=None), bc)
    solver.solve()

    eta = recovery_estimator(solver.equation).estimate(solver)
    assert np.all(eta < 1e-10)


def test_recovery_requires_a_solved_system(make_unit_square):
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=1.0), bc)

    with pytest.raises(ValueError, match='requires a solved system'):
        recovery_estimator(solver.equation).estimate(solver)


# -- drives adaptive refinement ----------------------------------------------


def test_recovery_drives_adaptive_refinement(make_unit_square):
    """The full loop, mirroring the residual estimator's: recovery drives the
    refiner, and the mesh grows and concentrates near a localised source."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=lambda p: 10.0 if np.linalg.norm(p - 0.5) < 0.1 else 0.0)
    solver = Solver(mesh, equation, bc)

    n_before = len(mesh.elements)
    AdaptiveRefinement(
        solver, recovery_estimator(equation), max_triangles=300, max_iters=5,
    ).run()

    assert len(solver.mesh.elements) > n_before
    centroids = solver.mesh.vertices[solver.mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    near_center = (center_dist < 0.2).sum()
    far_away = (center_dist > 0.35).sum()
    assert near_center > far_away * 0.3
