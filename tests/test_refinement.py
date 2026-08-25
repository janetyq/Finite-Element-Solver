"""The adaptive refinement loop's machinery, driven with a hand-written estimator: it
re-solves on each new mesh rather than reusing a stale estimate, and the conditions it
cannot carry across a remesh fail loudly.
"""
import numpy as np
import pytest

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.regions import everywhere, at_indices, on_plane
from fem.equations import LinearElastic, Projection, Poisson
from fem.energy_solver import EnergySolver
from fem.solver import Solver


def refine_near_centre(solver):
    """Estimator stand-in: 'error' is largest at the centre of the domain."""
    centroids = np.array([
        solver.mesh.vertices[element].mean(axis=0)
        for element in solver.mesh.elements
    ])
    return 1.0 / (0.05 + np.linalg.norm(centroids - 0.5, axis=1))


def test_adaptive_refinement_grows_mesh_and_resolves(make_unit_square):
    """The loop must refine repeatedly and leave a solution on the final mesh."""
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))

    n_before = len(mesh.elements)
    solution = AdaptiveRefinement(solver, refine_near_centre, max_triangles=400, max_iters=3).run()

    assert len(solver.mesh.elements) > n_before, "mesh never grew"
    # The solution must belong to the *final* mesh, not the one we started on.
    u = solution.u
    assert solution.mesh is solver.mesh
    assert len(u) == len(solver.mesh.vertices)
    assert np.all(np.isfinite(u))


def test_adaptive_refinement_respects_max_triangles(make_unit_square):
    """The old guard was `< max_triangles or max_iters == 0`, so max_iters never
    bound the loop and the element cap was the only thing stopping it."""
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))

    cap = len(mesh.elements) + 1
    AdaptiveRefinement(solver, refine_near_centre, max_triangles=cap, max_iters=50).run()
    # One round may overshoot the cap; the point is that it stops, not that it
    # lands exactly on it.
    assert len(solver.mesh.elements) < 400


def test_adaptive_refinement_respects_max_iters(make_unit_square):
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))

    AdaptiveRefinement(solver, refine_near_centre, max_triangles=10**6, max_iters=1).run()
    after_one = len(solver.mesh.elements)

    AdaptiveRefinement(solver, refine_near_centre, max_triangles=10**6, max_iters=1).run()
    assert len(solver.mesh.elements) > after_one, "max_iters=1 did no work"


def test_adaptive_refinement_carries_geometric_dirichlet_bcs(make_unit_square):
    """The payoff of position-based specs: a Dirichlet condition described as a
    region is re-resolved on each refined mesh, so it keeps holding on nodes that
    did not exist when it was written."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=1.0), bc)

    n_before = len(mesh.vertices)
    solution = AdaptiveRefinement(solver, refine_near_centre, max_triangles=400, max_iters=3).run()

    final = solver.mesh
    assert len(final.vertices) > n_before, "mesh never grew"
    u = solution.u
    # Every boundary node of the *refined* mesh is pinned, including the new ones.
    assert np.allclose(u[final.boundary_idxs], 0.0, atol=1e-12)
    assert np.abs(u).max() > 0, "solution is trivially zero"


def test_adaptive_refinement_rejects_index_based_bcs(make_unit_square):
    """at_indices names nodes of one specific mesh, so the loop must refuse it
    rather than quietly relocate the condition after renumbering."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, at_indices(mesh.boundary_idxs), 0.0)
    solver = Solver(mesh, Projection(source=1.0), bc)

    with pytest.raises(NotImplementedError):
        AdaptiveRefinement(solver, refine_near_centre).run()


def test_adaptive_refinement_rejects_mismatched_estimator(make_unit_square):
    """An estimator sized to the wrong mesh is refused instead of indexing unrelated elements."""
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))

    with pytest.raises(ValueError):
        AdaptiveRefinement(solver, lambda s: np.ones(len(s.mesh.elements) + 1)).run()


def test_bc_spec_is_reusable_across_meshes(make_unit_square):
    """The spec holds no mesh, so the same object resolves on any of them."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    coarse, fine = make_unit_square(4), make_unit_square(9)
    assert len(bc.resolve(coarse, n_components=1).fixed_idxs) == len(coarse.boundary_idxs)
    assert len(bc.resolve(fine, n_components=1).fixed_idxs) == len(fine.boundary_idxs)


def test_adaptive_refinement_drives_the_energy_solver(make_unit_square):
    """The driver takes a RefinableSolver, not a concrete Solver, so the nonlinear
    facade can be refined too. Before EnergySolver had `remesh` this was
    unreachable: the driver was typed to Solver and the energy path had no seam."""
    mesh = make_unit_square(5)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [0.02, 0.0])
    solver = EnergySolver(mesh, LinearElastic(E=200, nu=0.3), bc)

    n_before = len(mesh.elements)
    solution = AdaptiveRefinement(solver, refine_near_centre, max_triangles=200, max_iters=2).run()

    assert len(solver.mesh.elements) > n_before, "mesh never grew"
    assert solution.mesh is solver.mesh
    assert len(solution.u) == len(solver.mesh.vertices) * 2
    assert np.all(np.isfinite(solution.u))


def test_energy_solver_remesh_rebuilds_derived_state(make_unit_square):
    """`remesh` rebuilds the space, not just the mesh: a stale space is sized to the old
    vertex count."""
    coarse, fine = make_unit_square(4), make_unit_square(7)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    solver = EnergySolver(coarse, LinearElastic(E=200, nu=0.3), bc)

    solver.remesh(fine)

    assert solver.mesh is fine
    assert solver.space.mesh is fine
    assert solver.space.n_dofs == len(fine.vertices) * 2
    # The constraints follow the new mesh because the problem resolves them per
    # solve, rather than the solver holding a partition from the old one.
    _, fixed, _ = solver.problem().constraints
    assert len(fixed) == 2 * sum(1 for v in fine.vertices[fine.boundary_idxs] if v[0] == 0.0)
