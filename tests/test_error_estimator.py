"""Tests for the residual-based error estimator."""
from math import e

import numpy as np
import pytest

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.equations import LinearElastic, Poisson
from fem.mesh.mesh import Mesh
from fem.regions import everywhere, on_plane
from fem.solution import ElasticSolution
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


# -- LinearElastic.error_estimate --------------------------------------------
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
# Way 2 works because error_estimate never solves -- it only reads `.mesh`,
# `.space`, `.solution.stress`, and `.boundary_conditions`. So these tests set a
# known constant stress on a hand-sized mesh (`_inject_constant_stress`) and
# check eta against a value worked out by hand.

def _inject_constant_stress(solver, sigma_xx, sigma_yy, sigma_xy):
    """Replace `solver.solution` with a synthetic ElasticSolution carrying the
    same constant in-plane stress on every element -- enough for
    `error_estimate` (it reads `.stress` only) without an actual solve."""
    mesh = solver.mesh
    n = len(mesh.elements)
    stress = np.zeros((n, 3, 3))
    stress[:, 0, 0] = sigma_xx
    stress[:, 1, 1] = sigma_yy
    stress[:, 0, 1] = stress[:, 1, 0] = sigma_xy
    solver.solution = ElasticSolution(
        mesh=mesh, n_components=2, u=np.zeros(2 * len(mesh.vertices)),
        strain=np.zeros((n, 3, 3)), stress=stress, compliance=np.zeros(n),
    )


def _unit_square_two_triangles():
    """A 2-triangle unit square, split along the (0,2) diagonal -- small enough
    that every edge's outward normal and the elements it touches can be worked
    out by hand."""
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    elements = [[0, 1, 2], [0, 2, 3]]
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return Mesh(vertices, elements, boundary)


def test_elastic_error_estimator_returns_correct_shape(make_unit_square):
    """The estimator must return one non-negative finite value per element."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)
    solver.solve()

    eta = equation.error_estimate(solver)

    assert len(eta) == len(mesh.elements)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)


def test_elastic_error_estimator_requires_solved_system(make_unit_square):
    """Calling error_estimate before solve raises."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0, 0])
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)

    with pytest.raises(ValueError, match='requires a solved system'):
        equation.error_estimate(solver)


def test_elastic_error_estimator_linear_solution_small_jumps(make_unit_square):
    """A globally linear displacement field is constant-strain, hence an exact
    equilibrium solution with zero body force, so eta should vanish. Dirichlet
    everywhere means every boundary edge is skipped (both endpoints pinned), so
    this exercises the interior/jump terms only; the boundary term is covered
    separately below."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), lambda p: [0.01 * p[0], -0.003 * p[1]])
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)
    solver.solve()

    eta = equation.error_estimate(solver)

    assert np.all(eta < 1e-8)


def test_elastic_error_estimator_boundary_term_matches_hand_derivation():
    """Direct check of the new Neumann/natural-boundary residual term.

    Bottom edge is Dirichlet (pinned both ends) and must be skipped; nothing
    else carries a condition, so the other three edges are natural with g=0,
    and each free edge's residual is exactly ||sigma.n||^2 for the injected
    stress. Expected eta is worked out by hand from the mesh's own geometry
    (h_K = sqrt(2) for both elements, every boundary edge length 1) and the
    known outward normals: right -> (1,0), left -> (-1,0), top -> (0,1).
    """
    mesh = _unit_square_two_triangles()
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])  # bottom edge only
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)  # no .solve() -- only .mesh/.space are needed

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    _inject_constant_stress(solver, Sxx, Syy, Sxy)

    eta = equation.error_estimate(solver)

    h_K = np.sqrt(2.0)
    right_residual = Sxx**2 + Sxy**2          # element 0: bottom (skip) + right
    left_residual = Sxx**2 + Sxy**2           # element 1: top + left
    top_residual = Sxy**2 + Syy**2
    expected_e0 = np.sqrt(h_K * right_residual)
    expected_e1 = np.sqrt(h_K * (top_residual + left_residual))

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_elastic_error_estimator_neumann_matching_traction_is_quiet():
    """A Neumann condition that exactly matches the injected stress's traction
    contributes nothing on the edge it's declared on -- the estimator
    shouldn't flag a boundary as erroneous when the prescribed data and the
    solution agree.

    `g` is nodal, so a Neumann value also reaches the *neighbouring* free
    edges through their shared corner vertex: `error_estimate` averages the
    two endpoints' nodal `g` from `resolved.neumann_load`, which carries the
    traction as a nodal field. Declaring top's traction therefore also lifts
    `g` on right and left away from zero at the shared corner, changing their
    residuals too; this is that corner-sharing worked through by hand rather
    than assumed away. (The assembled *load* no longer spreads this way -- it
    integrates each traction over its own region's facets -- but the estimator
    reads the nodal `g` directly, where the corner is still shared.)
    """
    mesh = _unit_square_two_triangles()
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])   # bottom: skip
    bc.add(BCType.NEUMANN, on_plane(1, 1.0), [0.5, 1.0])  # top: matches sigma.n exactly
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    _inject_constant_stress(solver, Sxx, Syy, Sxy)

    eta = equation.error_estimate(solver)

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
    """A roller (bottom pinned in x, free in y) rather than a full clamp: the
    x-component has a live essential condition (no residual, matching the
    fully-clamped case), but y is natural there and must still be tested --
    using the full vector residual would wrongly count the pinned x-direction's
    reaction stress as error, which is exactly the false signal a roller fix
    needs the estimator not to produce."""
    mesh = _unit_square_two_triangles()
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, None])  # bottom: roller
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    _inject_constant_stress(solver, Sxx, Syy, Sxy)

    eta = equation.error_estimate(solver)

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
    """The full loop, mirroring test_adaptive_refinement_with_error_estimator:
    the mesh grows, the solution stays finite, and Dirichlet BCs -- carried as
    geometric regions rather than vertex indices -- still hold exactly on the
    refined mesh. Where refinement concentrates is not asserted here: for a
    fully clamped edge that is a real, budget-dependent question (see the
    module docstring above), not a property this loop-mechanics test should
    pin down.
    """
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)

    n_before = len(mesh.elements)
    solution = AdaptiveRefinement(
        solver, equation.error_estimate, max_triangles=n_before + 200, max_iters=5,
    ).run()

    final = solver.mesh
    assert len(final.elements) > n_before
    assert solution.mesh is final
    assert np.all(np.isfinite(solution.u))

    left = np.flatnonzero(np.abs(final.vertices[:, 0]) < 1e-12)
    assert np.allclose(solution.u.reshape(-1, 2)[left], 0.0, atol=1e-12)
