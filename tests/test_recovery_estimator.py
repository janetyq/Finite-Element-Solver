"""The Zienkiewicz-Zhu recovery error estimator, checked by its effectivity index, the
ratio of the estimate to the true error, which approaches 1 under refinement. The true
error is available because the manufactured-solution machinery supplies an exact flux.
"""
import numpy as np

from fem.analysis.adaptivity import AdaptiveRefinement
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from mms import (
    ELASTIC_E,
    ELASTIC_NU,
    elastic_source,
    exact_gradient,
    h1_seminorm_error,
    quadrature_l2,
)
from fem.physics.equations import LinearElastic, Poisson
from fem.analysis.estimators import RecoveryEstimator
from fem.physics.materials import Enu_to_Lame
from fem.mesh.structured import box_mesh
from fem.regions import everywhere
from fem.loads import Source

POISSON = Poisson()
POISSON_SOURCE = Source(lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])])
ELASTIC = LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU)
ELASTIC_SOURCE = Source(elastic_source)


def _global(eta):
    """The global estimate sqrt(sum eta_K^2) from the per-element indicators."""
    return float(np.sqrt((np.asarray(eta) ** 2).sum()))


def _elastic_true_stress_error(problem, solution):
    """||sigma_exact - sigma_h||_L2 (in-plane, Frobenius) for the elasticity MMS,
    the norm the recovered-stress estimate targets. Shares `quadrature_l2` with the
    gradient error; the Frobenius norm of the full symmetric 2x2 difference is what
    the primitive computes from the trailing tensor axes."""
    mu, lamb = Enu_to_Lame(ELASTIC_E, ELASTIC_NU)
    geometry = problem.space.geometry_at(2)
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
    sigma_h = solution.stress[:, None, :2, :2]             # (n_el, 1, 2, 2), constant per element
    return quadrature_l2(geometry, sigma_exact - sigma_h)


def _solved(equation, mesh, bc_value, source=None):
    bc = Conditions(Dirichlet(everywhere(), bc_value))
    problem = equation.problem(mesh, bc if source is None else bc + source)
    return problem, problem.solve()


def _square(n):
    return box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))


# -- effectivity: the estimate tracks the true error -------------------------


def test_poisson_recovery_is_asymptotically_exact():
    """The effectivity index stays bounded and tends to 1 as the mesh refines --
    the defining property of a good recovery estimator."""
    indices = []
    for n in (11, 21, 41):
        problem, solution = _solved(POISSON, _square(n), 0.0, POISSON_SOURCE)
        eta = _global(RecoveryEstimator().estimate(problem, solution))
        true_error = h1_seminorm_error(problem.space, solution.dofs, exact_gradient)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)          # bounded everywhere
    assert abs(indices[-1] - 1.0) < 0.1                 # asymptotically exact
    # Refinement moves the index toward 1, it does not drift away from it.
    assert abs(indices[-1] - 1.0) <= abs(indices[0] - 1.0) + 1e-9


def test_elastic_recovery_is_asymptotically_exact():
    """The same effectivity check for the vector, coupled elastic path."""
    indices = []
    for n in (11, 21, 41):
        problem, solution = _solved(ELASTIC, _square(n), [0.0, 0.0], ELASTIC_SOURCE)
        eta = _global(RecoveryEstimator().estimate(problem, solution))
        true_error = _elastic_true_stress_error(problem, solution)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)
    assert abs(indices[-1] - 1.0) < 0.1


# -- shape / sanity ----------------------------------------------------------


def test_recovery_of_a_linear_field_is_near_zero(make_unit_square):
    """A globally linear solution has constant gradient, so the estimate vanishes: the patch
    test for a recovery estimator."""
    equation = Poisson()
    problem, solution = _solved(equation, make_unit_square(6), lambda p: p[0])

    eta = RecoveryEstimator().estimate(problem, solution)
    assert np.all(eta < 1e-10)


# -- drives adaptive refinement ----------------------------------------------


def test_recovery_drives_adaptive_refinement(make_unit_square):
    """The full loop, mirroring the residual estimator's: recovery drives the
    refiner, and the mesh grows and concentrates near a localised source."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(everywhere(), 0.0))
    equation = Poisson()

    n_before = len(mesh.elements)
    driver = AdaptiveRefinement(
        mesh, lambda m: equation.problem(m, bc + Source(lambda p: 10.0 if np.linalg.norm(p - 0.5) < 0.1 else 0.0)),
        RecoveryEstimator(), max_triangles=300, max_iters=5,
    )
    driver.run()

    assert len(driver.mesh.elements) > n_before
    centroids = driver.mesh.vertices[driver.mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    near_center = (center_dist < 0.2).sum()
    far_away = (center_dist > 0.35).sum()
    assert near_center > far_away * 0.3
