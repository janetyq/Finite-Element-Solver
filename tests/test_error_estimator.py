"""Tests for the residual-based error estimator."""
from math import e

import numpy as np

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.equations import LinearElastic, Poisson
from fem.estimators import ResidualEstimator
from fem.mesh.mesh import Mesh
from fem.regions import everywhere, on_plane
from fem.solution import ElasticSolution
from fem.solve import LinearSolve


def _solved(mesh, equation, bc):
    """The problem on `mesh` and its solution."""
    problem = equation.problem(equation.space(mesh), bc)
    return problem, problem.solution(LinearSolve().solve(problem))


def _for(equation, bc):
    return lambda mesh: equation.problem(equation.space(mesh), bc)


def test_error_estimator_linear_solution_small_jumps(make_unit_square):
    """A problem with linear exact solution has near-zero gradient jumps."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), lambda p: p[0])
    equation = Poisson(source=None)
    problem, solution = _solved(mesh, equation, bc)

    eta = ResidualEstimator().estimate(problem, solution)

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
    equation = Poisson(source=peaked_source)
    problem, solution = _solved(mesh, equation, bc)

    eta = ResidualEstimator().estimate(problem, solution)
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)

    near_center = center_dist < 0.15
    far_from_center = center_dist > 0.35
    assert eta[near_center].mean() > eta[far_from_center].mean()


def test_adaptive_refinement_with_error_estimator(make_unit_square):
    """The full loop: estimator drives refinement, mesh grows near the source."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=lambda p: 10.0 if np.linalg.norm(p - 0.5) < 0.1 else 0.0)

    n_before = len(mesh.elements)
    driver = AdaptiveRefinement(mesh, _for(equation, bc), ResidualEstimator(),
                                max_triangles=300, max_iters=5)
    driver.run()

    assert len(driver.mesh.elements) > n_before

    # Verify refinement concentrates near the source
    centroids = driver.mesh.vertices[driver.mesh.elements].mean(axis=1)
    center_dist = np.linalg.norm(centroids - 0.5, axis=1)
    near_center = (center_dist < 0.2).sum()
    far_away = (center_dist > 0.35).sum()
    # The source is localised, so more refinement happens near the center
    assert near_center > far_away * 0.3


# -- the elastic boundary term ------------------------------------------------
#
# An error estimator can be tested two ways, and these tests use the second:
#
#   1. Solve a real problem and check refinement lands where the error should
#      be. This is what the Poisson tests above do -- a peaked source pulls
#      refinement toward the centre.
#   2. Feed in a stress by hand and check eta equals the number the formula
#      gives for it.
#
# Way 1 is unreliable for elasticity: the error has no single home. A clamped
# edge and the corners are real sources of error competing with the hole rim,
# and which one dominates depends on mesh resolution and refinement budget. An
# assertion about *where* refinement lands would test this problem's physics,
# not whether the formula is implemented correctly.
#
# Way 2 works because the estimator never solves: it reads the problem's space and
# resolved conditions and the solution's stress. So these tests state the problem on a
# hand-sized mesh, pair it with a synthetic solution carrying a known constant stress
# (`_constant_stress`), and check eta against a value worked out by hand.

def _constant_stress(problem, sigma_xx, sigma_yy, sigma_xy):
    """A synthetic ElasticSolution carrying the same constant in-plane stress on every
    element of `problem`'s space."""
    space = problem.space
    n = len(space.mesh.elements)
    stress = np.zeros((n, 3, 3))
    stress[:, 0, 0] = sigma_xx
    stress[:, 1, 1] = sigma_yy
    stress[:, 0, 1] = stress[:, 1, 0] = sigma_xy
    return ElasticSolution(
        space, u=np.zeros(space.n_dofs),
        strain=np.zeros((n, 3, 3)), stress=stress, compliance=np.zeros(n),
    )


def _unit_square_two_triangles():
    """A 2-triangle unit square, split along the (0,2) diagonal, small enough to work every
    edge's outward normal out by hand."""
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    elements = [[0, 1, 2], [0, 2, 3]]
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return Mesh(vertices, elements, boundary)


def test_elastic_error_estimator_linear_solution_small_jumps(make_unit_square):
    """A globally linear displacement is constant-strain, an exact equilibrium solution with
    zero body force, so eta vanishes. Dirichlet everywhere, so only the interior and
    jump terms are exercised."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), lambda p: [0.01 * p[0], -0.003 * p[1]])
    equation = LinearElastic(E=200, nu=0.3)
    problem, solution = _solved(mesh, equation, bc)

    eta = ResidualEstimator().estimate(problem, solution)

    assert np.all(eta < 1e-8)


def test_elastic_error_estimator_boundary_term_matches_hand_derivation():
    """The Neumann/natural boundary residual against a hand derivation: the bottom edge is
    Dirichlet and skipped, the other three are natural with g=0, so each free edge's
    residual is ||sigma.n||^2 for the injected stress."""
    mesh = _unit_square_two_triangles()
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])  # bottom edge only
    equation = LinearElastic(E=200, nu=0.3)
    problem = equation.problem(equation.space(mesh), bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    eta = ResidualEstimator().estimate(problem, _constant_stress(problem, Sxx, Syy, Sxy))

    h_K = np.sqrt(2.0)
    right_residual = Sxx**2 + Sxy**2          # element 0: bottom (skip) + right
    left_residual = Sxx**2 + Sxy**2           # element 1: top + left
    top_residual = Sxy**2 + Syy**2
    expected_e0 = np.sqrt(h_K * right_residual)
    expected_e1 = np.sqrt(h_K * (top_residual + left_residual))

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_elastic_error_estimator_neumann_matching_traction_is_quiet():
    """A Neumann condition that exactly matches the injected stress's traction contributes
    nothing on its edge. `g` is nodal, so it also reaches the neighbouring free edges
    through the shared corner; that is worked through by hand."""
    mesh = _unit_square_two_triangles()
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])   # bottom: skip
    bc.add(BCType.NEUMANN, on_plane(1, 1.0), [0.5, 1.0])  # top: matches sigma.n exactly
    equation = LinearElastic(E=200, nu=0.3)
    problem = equation.problem(equation.space(mesh), bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    eta = ResidualEstimator().estimate(problem, _constant_stress(problem, Sxx, Syy, Sxy))

    h_K = np.sqrt(2.0)
    # top's g=[0.5, 1.0] is shared by nodes 2 and 3; right (1,2) and left (0,3)
    # each average one matched endpoint (g=[0.5,1.0]) with one untouched
    # endpoint (g=[0,0]), giving g=[0.25, 0.5] on both. top itself matches
    # sigma.n exactly and drops out.
    g = np.array([0.25, 0.5])
    right_residual = np.sum((g - np.array([Sxx, Sxy]))**2)   # sigma.n on (1,0)
    left_residual = np.sum((g - np.array([-Sxx, -Sxy]))**2)  # sigma.n on (-1,0)
    expected_e0 = np.sqrt(h_K * right_residual)
    expected_e1 = np.sqrt(h_K * left_residual)

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_elastic_error_estimator_roller_edge_only_tests_its_free_component():
    """On a roller (bottom pinned in x, free in y) only the free component is tested; the
    pinned direction's reaction stress must not count as error."""
    mesh = _unit_square_two_triangles()
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, None])  # bottom: roller
    equation = LinearElastic(E=200, nu=0.3)
    problem = equation.problem(equation.space(mesh), bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    eta = ResidualEstimator().estimate(problem, _constant_stress(problem, Sxx, Syy, Sxy))

    h_K = np.sqrt(2.0)
    # bottom's outward normal is (0, -1), so sigma.n = (-Sxy, -Syy); only the
    # free y-component (residual Syy^2, g=0) counts, not x (-Sxy is pinned
    # reaction stress, not error).
    bottom_residual = Syy**2
    right_residual = Sxx**2 + Sxy**2   # untouched: same free edge as before
    top_residual = Sxy**2 + Syy**2     # untouched: bottom's roller doesn't reach it
    left_residual = Sxx**2 + Sxy**2    # untouched
    expected_e0 = np.sqrt(h_K * (bottom_residual + right_residual))
    expected_e1 = np.sqrt(h_K * (top_residual + left_residual))

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_adaptive_refinement_elasticity_runs_end_to_end(make_unit_square):
    """The full loop: the mesh grows, the solution stays finite, and the geometric Dirichlet
    conditions still hold exactly on the refined mesh."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])
    equation = LinearElastic(E=200, nu=0.3)

    n_before = len(mesh.elements)
    driver = AdaptiveRefinement(mesh, _for(equation, bc), ResidualEstimator(),
                                max_triangles=n_before + 200, max_iters=5)
    solution = driver.run()

    final = driver.mesh
    assert len(final.elements) > n_before
    assert solution.mesh is final
    assert np.all(np.isfinite(solution.u))

    left = np.flatnonzero(np.abs(final.vertices[:, 0]) < 1e-12)
    assert np.allclose(solution.u.reshape(-1, 2)[left], 0.0, atol=1e-12)
