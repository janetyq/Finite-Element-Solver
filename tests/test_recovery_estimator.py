"""The Zienkiewicz-Zhu recovery error estimator: what it measures, on P1 and P2.

The shared contract (shape, the patch test, the peak, the refinement loop) lives in
`test_estimator_contract.py`. Here: the effectivity index, the ratio of the estimate to
the true error, which the manufactured solutions make available. On P1 nodal averaging
is asymptotically exact; on P2 the flux is sampled per quadrature point and the index
stays bounded.
"""
import numpy as np
import pytest
from helpers import global_estimate, pinned, solved
from mms import (
    ELASTIC_E,
    ELASTIC_NU,
    elastic_source,
    exact_gradient,
    h1_seminorm_error,
    quadrature_l2,
    source_term,
)

from fem.analysis.estimators import RecoveryEstimator
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.derived import GradientFlux, StressFlux
from fem.physics.equations import LinearElastic, Poisson
from fem.physics.materials import Enu_to_Lame
from fem.regions import everywhere

ELASTIC = LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU)


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


# -- effectivity: the estimate tracks the true error -------------------------


def test_poisson_recovery_is_asymptotically_exact(make_unit_square):
    """The effectivity index stays bounded and tends to 1 as the mesh refines: the
    defining property of a good recovery estimator."""
    indices = []
    for n in (11, 21, 41):
        problem, solution = solved(Poisson(), make_unit_square(n), pinned() + Source(source_term))
        eta = global_estimate(RecoveryEstimator().estimate(problem, solution))
        true_error = h1_seminorm_error(problem.space, solution.dofs, exact_gradient)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)          # bounded everywhere
    assert abs(indices[-1] - 1.0) < 0.1                 # asymptotically exact
    # Refinement moves the index toward 1, it does not drift away from it.
    assert abs(indices[-1] - 1.0) <= abs(indices[0] - 1.0) + 1e-9


def test_elastic_recovery_is_asymptotically_exact(make_unit_square):
    """The same effectivity check for the vector, coupled elastic path."""
    indices = []
    for n in (11, 21, 41):
        problem, solution = solved(ELASTIC, make_unit_square(n), pinned(2) + Source(elastic_source))
        eta = global_estimate(RecoveryEstimator().estimate(problem, solution))
        true_error = _elastic_true_stress_error(problem, solution)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)
    assert abs(indices[-1] - 1.0) < 0.1


def test_p2_recovery_effectivity_stays_bounded(make_unit_square):
    """The estimate tracks the true H1 error across a P2 refinement sequence: the
    effectivity index stays comfortably bounded around 1. L2-projection recovery on P2
    is not asymptotically exact the way P1 nodal averaging is, so this checks that it
    stays a faithful indicator, not that the index tends to 1."""
    indices = []
    for n in (6, 11, 21):
        problem, solution = solved(Poisson(), make_unit_square(n), pinned() + Source(source_term),
                                   element_type=QuadraticTriangleElement)
        eta = global_estimate(RecoveryEstimator().estimate(problem, solution))
        true_error = h1_seminorm_error(problem.space, solution.dofs, exact_gradient)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)


# -- the mechanism ------------------------------------------------------------


def test_sample_sees_within_element_variation_on_p2_but_not_p1(make_unit_square):
    """The core of P2 recovery: `sample` reads the flux at each quadrature point, so on
    P2 (a linear flux) it varies within an element, while on P1 (a constant flux) it does
    not. Estimating a P2 solution's error hangs on seeing that variation."""
    spreads = {}
    for element_type, name in [(LinearTriangleElement, 'P1'), (QuadraticTriangleElement, 'P2')]:
        problem, solution = solved(Poisson(), make_unit_square(4), pinned() + Source(source_term),
                                   element_type=element_type)
        geometry = problem.space.geometry_at(4)
        sampled = GradientFlux().sample(solution, geometry)   # (n_el, n_qp, 1, d)
        spreads[name] = np.ptp(sampled, axis=1).max()          # spread across qp

    assert spreads['P1'] == pytest.approx(0.0, abs=1e-12)
    assert spreads['P2'] > 0.1


def test_elastic_recovery_reads_the_full_stress_in_3d():
    """The stress flux slices the tensor to the mesh's own dimension, so a 3D solve
    recovers all three rows and the estimate is one finite indicator per tet."""
    mesh = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(4, 4, 4))
    bc = Conditions(Dirichlet(everywhere(), [0.0, 0.0, 0.0]), Source([0.0, 0.0, -1.0]))
    problem, solution = solved(ELASTIC, mesh, bc)

    assert StressFlux().evaluate(solution).shape == (len(mesh.elements), 3, 3)
    eta = RecoveryEstimator().estimate(problem, solution)
    assert eta.shape == (len(mesh.elements),)
    assert np.all(np.isfinite(eta)) and np.all(eta >= 0.0)
    assert eta.max() > 0.0
