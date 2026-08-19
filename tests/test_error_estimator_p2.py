"""The residual error estimator on quadratic (P2) elements.

The P1 residual estimator drops two terms that vanish only for a constant flux: the
interior term keeps just the source (`div(flux) = 0` inside a constant-strain element),
and the edge jump compares one representative flux per element. On P2 the flux varies
within each element, so both need the flux's real behaviour: the interior residual gains
`div(flux)` (a Laplacian, or the Navier stress divergence), computed from the P2 shape
functions' second derivatives, and the jump is read at the shared edge from each side.

These tests pin the new machinery (shape Hessians, the field Hessian, the two
divergences), the correctness (the estimator vanishes on a field P2 represents exactly),
and that it drives adaptive refinement on a P2 space.
"""
import numpy as np
import pytest

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import exact_gradient, h1_seminorm_error
from fem.elements import QuadraticTriangleElement
from fem.equations import LinearElastic, Poisson
from fem.estimators import residual_estimator
from fem.materials import Enu_to_Lame
from fem.mesh.structured import create_rect_mesh
from fem.postprocess import GradientField
from fem.regions import everywhere, on_plane
from fem.solution import ElasticSolution, ScalarFieldSolution
from fem.solver import Solver
from fem.space import FunctionSpace


def _poisson_source(point):
    return [2 * np.pi**2 * np.sin(np.pi * point[0]) * np.sin(np.pi * point[1])]


def _global(eta):
    return float(np.sqrt((np.asarray(eta) ** 2).sum()))


def _solve(equation, n, bc_value=0.0):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), bc_value)
    solver = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement)
    solver.solve()
    return solver


# -- the new primitives: shape Hessians, field Hessian, the divergences -------


def test_shape_hessians_match_finite_differences():
    """The analytic P2 Hessians are the derivative of the shape gradients: a central
    difference of `shape_gradients` reproduces them, so the hand-written constants are
    the real second derivatives, not a transcription slip."""
    points = np.array([[0.3, 0.4], [0.1, 0.25], [0.5, 0.2]])
    hessians = QuadraticTriangleElement.shape_hessians(points)   # (n_pts, 6, 2, 2)

    h = 1e-6
    fd = np.zeros_like(hessians)
    for j, step in enumerate(np.eye(2) * h):
        plus = QuadraticTriangleElement.shape_gradients(points + step)    # (n_pts, 6, 2)
        minus = QuadraticTriangleElement.shape_gradients(points - step)
        fd[..., j] = (plus - minus) / (2 * h)

    assert np.allclose(hessians, fd, atol=1e-6)
    # Symmetric, as a Hessian must be.
    assert np.allclose(hessians, np.swapaxes(hessians, -1, -2))


def test_element_field_hessian_recovers_a_quadratic_fields_curvature():
    """u = a x^2 + b xy + c y^2 has constant Hessian [[2a, b], [b, 2c]]; the space
    recovers exactly that on every element, the physical mapping of the reference
    Hessian through the inverse Jacobian working out."""
    mesh = create_rect_mesh(corners=[[0, 0], [2, 1]], resolution=(4, 3))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)
    a, b, c = 1.5, -0.7, 2.0
    x, y = space.node_coords[:, 0], space.node_coords[:, 1]
    u = a * x**2 + b * x * y + c * y**2

    hessian = space.element_field_hessian(u[space.element_nodes])   # (n_el, 2, 2)

    expected = np.array([[2 * a, b], [b, 2 * c]])
    assert np.allclose(hessian, expected)


def test_p2_poisson_divergence_is_the_laplacian():
    """div(grad u) = laplacian(u). For u = x^2 + 2 y^2 that is a constant 6, recovered
    per element from the P2 field."""
    mesh = create_rect_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 4))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)
    x, y = space.node_coords[:, 0], space.node_coords[:, 1]
    u = x**2 + 2 * y**2
    solution = ScalarFieldSolution(mesh, 1, u, flux=np.zeros((len(mesh.elements), 2)),
                                   element_type=QuadraticTriangleElement)

    div = Poisson().derived_field().divergence(solution)
    assert np.allclose(div, 6.0)


def test_p2_elastic_divergence_is_the_navier_operator():
    """div(sigma) in Navier form (lambda + mu) grad(div u) + mu laplacian(u). For
    u = (x^2, 0) that is (2 lambda + 4 mu, 0), the exact strong-form residual the
    interior term needs."""
    E, nu = 200.0, 0.3
    mu, lamb = Enu_to_Lame(E, nu)
    mesh = create_rect_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 4))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)
    u = np.zeros((space.n_nodes, 2))
    u[:, 0] = space.node_coords[:, 0]**2
    n_el = len(mesh.elements)
    solution = ElasticSolution(mesh, 2, u.ravel(), strain=np.zeros((n_el, 3, 3)),
                               stress=np.zeros((n_el, 3, 3)), compliance=np.zeros(n_el),
                               element_type=QuadraticTriangleElement)

    div = LinearElastic(E, nu).derived_field().divergence(solution)
    assert np.allclose(div, [2 * lamb + 4 * mu, 0.0])


# -- correctness: vanishes on a representable field ---------------------------


def test_p2_residual_vanishes_on_a_quadratic_field():
    """The patch test: u = x^2 - y^2 is harmonic, so the P2 solve is exact. Its interior
    residual (f + laplacian u = 0), its edge jumps (a globally continuous linear
    gradient), and its boundary term (Dirichlet everywhere, so every edge is skipped) all
    vanish, and every indicator is zero."""
    solver = _solve(Poisson(source=None), 5, bc_value=lambda p: p[0]**2 - p[1]**2)
    eta = residual_estimator(solver.equation).estimate(solver)
    assert np.all(eta < 1e-10)


def test_p2_residual_interior_term_is_active():
    """A guard that the interior div term is really carried: the P2 Laplacian of a
    non-harmonic solve is not identically zero, so dropping it would change the estimate."""
    solver = _solve(Poisson(source=_poisson_source), 8)
    laplacian = GradientField().divergence(solver.solution)
    assert np.abs(laplacian).max() > 1.0


# -- reliability and behaviour ------------------------------------------------


def test_p2_residual_reliability_is_bounded_and_stable():
    """The estimate stays a faithful upper bound on the true error across a P2 refinement
    sequence: the effectivity index is bounded and does not drift up. A residual estimator
    over-estimates (its constant is looser than a recovery estimator's), so this checks
    boundedness and stability, not that the index tends to 1."""
    indices = []
    for n in (6, 11, 21):
        solver = _solve(Poisson(source=_poisson_source), n)
        eta = _global(residual_estimator(solver.equation).estimate(solver))
        true_error = h1_seminorm_error(solver.space, solver.solution.u, exact_gradient)
        indices.append(eta / true_error)

    assert all(1.0 < i < 15.0 for i in indices)
    assert indices[-1] <= indices[0] + 1e-9        # does not drift upward under refinement


def test_p2_residual_concentrates_near_a_peaked_source():
    """The indicator is largest where the solution is hardest to resolve."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(10, 10))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    def peaked_source(point):
        a = 50
        x, y = point - 0.5
        r2 = x**2 + y**2
        return 4 * a * a * (1 - a * r2) * np.exp(-a * r2)

    solver = Solver(mesh, Poisson(source=peaked_source), bc,
                    element_type=QuadraticTriangleElement)
    solver.solve()

    eta = residual_estimator(solver.equation).estimate(solver)
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    assert eta[center_dist < 0.15].mean() > eta[center_dist > 0.35].mean()


@pytest.mark.parametrize('equation, bc_value', [
    (Poisson(source=_poisson_source), 0.0),
    (LinearElastic(E=200, nu=0.3), [0.0, 0.0]),
])
def test_p2_residual_returns_one_nonnegative_value_per_element(equation, bc_value):
    solver = _solve(equation, 8, bc_value=bc_value)
    eta = residual_estimator(solver.equation).estimate(solver)
    assert eta.shape == (len(solver.mesh.elements),)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)


def test_p2_residual_drives_adaptive_refinement():
    """The full loop on a P2 space: the mesh grows, concentrates near a localised source,
    and the solve stays P2 across remeshes."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=lambda p: 10.0 if np.linalg.norm(p - 0.5) < 0.1 else 0.0)
    solver = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement)

    n_before = len(mesh.elements)
    AdaptiveRefinement(
        solver, residual_estimator(equation), max_triangles=300, max_iters=5,
    ).run()

    assert len(solver.mesh.elements) > n_before
    assert solver.solution.element_type is QuadraticTriangleElement
    centroids = solver.mesh.vertices[solver.mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    near_center = (center_dist < 0.2).sum()
    far_away = (center_dist > 0.35).sum()
    assert near_center > far_away * 0.3


def test_p2_elastic_residual_runs_end_to_end():
    """A P2 elastic solve with a Neumann edge produces a finite, non-negative estimate,
    exercising the stress divergence and the masked boundary term together."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement)
    solver.solve()

    eta = residual_estimator(equation).estimate(solver)
    assert eta.shape == (len(mesh.elements),)
    assert np.all(np.isfinite(eta)) and np.all(eta >= 0)
