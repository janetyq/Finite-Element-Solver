"""Tests for the adaptive refinement loop's machinery.

The error estimator itself is still open (BACKLOG section 3), so these drive the
loop with a hand-written estimator. What they pin down is everything around it:
that a refined FEMesh stays a usable FEMesh, that the loop re-solves on each new
mesh rather than reusing a stale estimate, and that the conditions it cannot
carry across a remesh fail loudly.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.mesh.femesh import FEMesh
from fem.mesh.refinement import RedGreenRefiner
from fem.regions import everywhere, at_indices
from fem.solver import Solver, Projection, Poisson


def refine_near_centre(solver):
    """Estimator stand-in: 'error' is largest at the centre of the domain."""
    centroids = np.array([
        solver.femesh.vertices[element].mean(axis=0)
        for element in solver.femesh.elements
    ])
    return 1.0 / (0.05 + np.linalg.norm(centroids - 0.5, axis=1))


def test_refined_mesh_is_still_an_femesh(make_unit_square):
    """The refiner once built a bare Mesh, so the refined result silently lost
    element_objs/M/K and blew up on the next solve."""
    femesh = make_unit_square(6)
    refiner = RedGreenRefiner(femesh)
    refined = refiner.refine([0, 1, 2])

    assert isinstance(refined, FEMesh)
    assert refined.element_type is femesh.element_type
    assert len(refined.element_objs) == len(refined.elements)
    assert refined.K.shape == (len(refined.vertices),) * 2


def test_copy_preserves_concrete_mesh_type(make_unit_square):
    femesh = make_unit_square(6)
    assert isinstance(femesh.copy(), FEMesh)


def test_adaptive_refinement_grows_mesh_and_resolves(make_unit_square):
    """The loop must refine repeatedly and leave a solution on the final mesh."""
    femesh = make_unit_square(6)
    solver = Solver(femesh, Projection(source=1.0))
    solver.solve()

    n_before = len(femesh.elements)
    solution = solver.adaptive_refinement(refine_near_centre, max_triangles=400, max_iters=3)

    assert len(solver.femesh.elements) > n_before, "mesh never grew"
    # The solution must belong to the *final* mesh, not the one we started on.
    u = solution.get_values("u")
    assert solution.femesh is solver.femesh
    assert len(u) == len(solver.femesh.vertices)
    assert np.all(np.isfinite(u))


def test_adaptive_refinement_respects_max_triangles(make_unit_square):
    """The old guard was `< max_triangles or max_iters == 0`, so max_iters never
    bound the loop and the element cap was the only thing stopping it."""
    femesh = make_unit_square(6)
    solver = Solver(femesh, Projection(source=1.0))
    solver.solve()

    cap = len(femesh.elements) + 1
    solver.adaptive_refinement(refine_near_centre, max_triangles=cap, max_iters=50)
    # One round may overshoot the cap; the point is that it stops, not that it
    # lands exactly on it.
    assert len(solver.femesh.elements) < 400


def test_adaptive_refinement_respects_max_iters(make_unit_square):
    femesh = make_unit_square(6)
    solver = Solver(femesh, Projection(source=1.0))
    solver.solve()

    solver.adaptive_refinement(refine_near_centre, max_triangles=10**6, max_iters=1)
    after_one = len(solver.femesh.elements)

    solver.adaptive_refinement(refine_near_centre, max_triangles=10**6, max_iters=1)
    assert len(solver.femesh.elements) > after_one, "max_iters=1 did no work"


def test_adaptive_refinement_carries_geometric_dirichlet_bcs(make_unit_square):
    """The payoff of position-based specs: a Dirichlet condition described as a
    region is re-resolved on each refined mesh, so it keeps holding on nodes that
    did not exist when it was written."""
    femesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(femesh, Poisson(source=1.0), bc)
    solver.solve()

    n_before = len(femesh.vertices)
    solution = solver.adaptive_refinement(refine_near_centre, max_triangles=400, max_iters=3)

    final = solver.femesh
    assert len(final.vertices) > n_before, "mesh never grew"
    u = solution.get_values("u")
    # Every boundary node of the *refined* mesh is pinned, including the new ones.
    assert np.allclose(u[final.boundary_idxs], 0.0, atol=1e-12)
    assert np.abs(u).max() > 0, "solution is trivially zero"


def test_adaptive_refinement_rejects_index_based_bcs(make_unit_square):
    """at_indices names nodes of one specific mesh, so the loop must refuse it
    rather than quietly relocate the condition after renumbering."""
    femesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, at_indices(femesh.boundary_idxs), 0.0)
    solver = Solver(femesh, Projection(source=1.0), bc)
    solver.solve()

    with pytest.raises(NotImplementedError):
        solver.adaptive_refinement(refine_near_centre)


def test_adaptive_refinement_rejects_mismatched_estimator(make_unit_square):
    """An estimator sized to the wrong mesh is exactly the staleness bug the old
    implementation had; catch it instead of indexing unrelated elements."""
    femesh = make_unit_square(6)
    solver = Solver(femesh, Projection(source=1.0))
    solver.solve()

    with pytest.raises(ValueError):
        solver.adaptive_refinement(lambda s: np.ones(len(s.femesh.elements) + 1))


def test_bc_spec_is_reusable_across_meshes(make_unit_square):
    """The spec holds no mesh, so the same object resolves on any of them."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    coarse, fine = make_unit_square(4), make_unit_square(9)
    assert len(bc.resolve(coarse, n_components=1).fixed_idxs) == len(coarse.boundary_idxs)
    assert len(bc.resolve(fine, n_components=1).fixed_idxs) == len(fine.boundary_idxs)
