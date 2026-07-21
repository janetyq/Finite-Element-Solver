"""Tests for the adaptive refinement loop's machinery.

The error estimator itself is still open (BACKLOG section 3), so these drive the
loop with a hand-written estimator. What they pin down is everything around it:
that the loop re-solves on each new mesh rather than reusing a stale estimate,
and that the conditions it cannot carry across a remesh fail loudly.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.regions import everywhere, at_indices
from fem.solver import Solver, Projection, Poisson


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
    solver.solve()

    n_before = len(mesh.elements)
    solution = solver.adaptive_refinement(refine_near_centre, max_triangles=400, max_iters=3)

    assert len(solver.mesh.elements) > n_before, "mesh never grew"
    # The solution must belong to the *final* mesh, not the one we started on.
    u = solution.get_values("u")
    assert solution.mesh is solver.mesh
    assert len(u) == len(solver.mesh.vertices)
    assert np.all(np.isfinite(u))


def test_adaptive_refinement_respects_max_triangles(make_unit_square):
    """The old guard was `< max_triangles or max_iters == 0`, so max_iters never
    bound the loop and the element cap was the only thing stopping it."""
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))
    solver.solve()

    cap = len(mesh.elements) + 1
    solver.adaptive_refinement(refine_near_centre, max_triangles=cap, max_iters=50)
    # One round may overshoot the cap; the point is that it stops, not that it
    # lands exactly on it.
    assert len(solver.mesh.elements) < 400


def test_adaptive_refinement_respects_max_iters(make_unit_square):
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))
    solver.solve()

    solver.adaptive_refinement(refine_near_centre, max_triangles=10**6, max_iters=1)
    after_one = len(solver.mesh.elements)

    solver.adaptive_refinement(refine_near_centre, max_triangles=10**6, max_iters=1)
    assert len(solver.mesh.elements) > after_one, "max_iters=1 did no work"


def test_adaptive_refinement_carries_geometric_dirichlet_bcs(make_unit_square):
    """The payoff of position-based specs: a Dirichlet condition described as a
    region is re-resolved on each refined mesh, so it keeps holding on nodes that
    did not exist when it was written."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=1.0), bc)
    solver.solve()

    n_before = len(mesh.vertices)
    solution = solver.adaptive_refinement(refine_near_centre, max_triangles=400, max_iters=3)

    final = solver.mesh
    assert len(final.vertices) > n_before, "mesh never grew"
    u = solution.get_values("u")
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
    solver.solve()

    with pytest.raises(NotImplementedError):
        solver.adaptive_refinement(refine_near_centre)


def test_adaptive_refinement_rejects_mismatched_estimator(make_unit_square):
    """An estimator sized to the wrong mesh is exactly the staleness bug the old
    implementation had; catch it instead of indexing unrelated elements."""
    mesh = make_unit_square(6)
    solver = Solver(mesh, Projection(source=1.0))
    solver.solve()

    with pytest.raises(ValueError):
        solver.adaptive_refinement(lambda s: np.ones(len(s.mesh.elements) + 1))


def test_bc_spec_is_reusable_across_meshes(make_unit_square):
    """The spec holds no mesh, so the same object resolves on any of them."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    coarse, fine = make_unit_square(4), make_unit_square(9)
    assert len(bc.resolve(coarse, n_components=1).fixed_idxs) == len(coarse.boundary_idxs)
    assert len(bc.resolve(fine, n_components=1).fixed_idxs) == len(fine.boundary_idxs)
