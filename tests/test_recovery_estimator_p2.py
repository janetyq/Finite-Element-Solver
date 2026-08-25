"""The recovery estimator on quadratic (P2) elements: `sample` sees the within-element
variation, the estimate vanishes on a field P2 represents exactly, its effectivity
stays bounded, and it drives adaptive refinement on a P2 space.
"""
import numpy as np
import pytest

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import (
    exact_gradient,
    h1_seminorm_error,
)
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.equations import Poisson
from fem.estimators import recovery_estimator
from fem.mesh.structured import create_rect_mesh
from fem.postprocess import GradientField
from fem.regions import everywhere
from fem.solver import Solver


def _poisson_source(point):
    return [2 * np.pi**2 * np.sin(np.pi * point[0]) * np.sin(np.pi * point[1])]


def _global(eta):
    return float(np.sqrt((np.asarray(eta) ** 2).sum()))


def _solve(equation, n, element_type, bc_value=0.0):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), bc_value)
    solver = Solver(mesh, equation, bc, element_type=element_type)
    solver.solve()
    return solver


# -- the mechanism: the flux is sampled per point, not per element -----------


def test_sample_sees_within_element_variation_on_p2_but_not_p1():
    """The core of P2 recovery: `sample` reads the flux at each quadrature point, so on
    P2 (a linear flux) it varies within an element, while on P1 (a constant flux) it does
    not. Estimating a P2 solution's error hangs on seeing that variation."""
    spreads = {}
    for element_type, name in [(LinearTriangleElement, 'P1'), (QuadraticTriangleElement, 'P2')]:
        solver = _solve(Poisson(source=_poisson_source), 4, element_type)
        geometry = solver.space.geometry_at(4)
        sampled = GradientField().sample(solver.solution, geometry)   # (n_el, n_qp, 1, d)
        spreads[name] = np.ptp(sampled, axis=1).max()                 # spread across qp

    assert spreads['P1'] == pytest.approx(0.0, abs=1e-12)
    assert spreads['P2'] > 0.1


# -- correctness: vanishes on a representable field, effectivity bounded ------


def test_p2_recovery_vanishes_on_a_quadratic_field():
    """The patch test for a P2 recovery estimator: a globally quadratic solution is
    represented exactly, so its recovered flux equals the discrete one and every
    indicator is zero. u = x^2 - y^2 is harmonic, so no source is needed."""
    solver = _solve(Poisson(source=None), 5, QuadraticTriangleElement,
                    bc_value=lambda p: p[0]**2 - p[1]**2)
    eta = recovery_estimator(solver.equation).estimate(solver)
    assert np.all(eta < 1e-10)


def test_p2_recovery_effectivity_stays_bounded():
    """The estimate tracks the true H1 error across a P2 refinement sequence: the
    effectivity index stays comfortably bounded around 1. L2-projection recovery on P2
    is not asymptotically exact the way P1 nodal averaging is, so this checks that it
    stays a faithful indicator, not that the index tends to 1."""
    indices = []
    for n in (6, 11, 21):
        solver = _solve(Poisson(source=_poisson_source), n, QuadraticTriangleElement)
        eta = _global(recovery_estimator(solver.equation).estimate(solver))
        true_error = h1_seminorm_error(solver.space, solver.solution.u, exact_gradient)
        indices.append(eta / true_error)

    assert all(0.5 < i < 2.0 for i in indices)


# -- drives adaptive refinement on a P2 space --------------------------------


def test_p2_recovery_drives_adaptive_refinement():
    """The full loop on a P2 space: recovery drives the refiner, the mesh grows and
    concentrates near a localised source, and the solve stays P2 across remeshes."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=lambda p: 10.0 if np.linalg.norm(p - 0.5) < 0.1 else 0.0)
    solver = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement)

    n_before = len(mesh.elements)
    AdaptiveRefinement(
        solver, recovery_estimator(equation), max_triangles=300, max_iters=5,
    ).run()

    assert len(solver.mesh.elements) > n_before
    assert solver.solution.element_type is QuadraticTriangleElement
    centroids = solver.mesh.vertices[solver.mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    near_center = (center_dist < 0.2).sum()
    far_away = (center_dist > 0.35).sum()
    assert near_center > far_away * 0.3
