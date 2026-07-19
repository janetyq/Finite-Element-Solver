"""Tests for the adaptive refinement loop's machinery.

The error estimator itself is still open (BACKLOG section 3), so these drive the
loop with a hand-written estimator. What they pin down is everything around it:
that a refined FEMesh stays a usable FEMesh, that the loop re-solves on each new
mesh rather than reusing a stale estimate, and that the conditions it cannot
carry across a remesh fail loudly.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions
from fem.mesh.femesh import FEMesh
from fem.mesh.refinement import RefinementMesh
from fem.solver import Solver, Projection


def refine_near_centre(solver):
    """Estimator stand-in: 'error' is largest at the centre of the domain."""
    centroids = np.array([
        solver.femesh.vertices[element].mean(axis=0)
        for element in solver.femesh.elements
    ])
    return 1.0 / (0.05 + np.linalg.norm(centroids - 0.5, axis=1))


def test_refined_mesh_is_still_an_femesh(make_unit_square):
    """RefinementMesh built a bare Mesh, so the refined result silently lost
    element_objs/M/K and blew up on the next solve."""
    femesh = make_unit_square(6)
    refiner = RefinementMesh(femesh)
    refiner.refine_triangles([0, 1, 2])
    refined = refiner.get_mesh()

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
    bc = BoundaryConditions(femesh)
    bc.add_force(lambda p: [1.0])
    solver = Solver(femesh, Projection(), bc)
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
    bc = BoundaryConditions(femesh)
    bc.add_force(lambda p: [1.0])
    solver = Solver(femesh, Projection(), bc)
    solver.solve()

    cap = len(femesh.elements) + 1
    solver.adaptive_refinement(refine_near_centre, max_triangles=cap, max_iters=50)
    # One round may overshoot the cap; the point is that it stops, not that it
    # lands exactly on it.
    assert len(solver.femesh.elements) < 400


def test_adaptive_refinement_respects_max_iters(make_unit_square):
    femesh = make_unit_square(6)
    bc = BoundaryConditions(femesh)
    bc.add_force(lambda p: [1.0])
    solver = Solver(femesh, Projection(), bc)
    solver.solve()

    solver.adaptive_refinement(refine_near_centre, max_triangles=10**6, max_iters=1)
    after_one = len(solver.femesh.elements)

    solver.adaptive_refinement(refine_near_centre, max_triangles=10**6, max_iters=1)
    assert len(solver.femesh.elements) > after_one, "max_iters=1 did no work"


def test_adaptive_refinement_rejects_index_based_bcs(make_unit_square):
    """Dirichlet conditions cannot survive vertex renumbering, so the loop must
    refuse rather than quietly relocate them to unrelated nodes."""
    femesh = make_unit_square(6)
    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", femesh.boundary_idxs, np.zeros(len(femesh.boundary_idxs)))
    solver = Solver(femesh, Projection(), bc)
    solver.solve()

    with pytest.raises(NotImplementedError):
        solver.adaptive_refinement(refine_near_centre)


def test_adaptive_refinement_rejects_mismatched_estimator(make_unit_square):
    """An estimator sized to the wrong mesh is exactly the staleness bug the old
    implementation had; catch it instead of indexing unrelated elements."""
    femesh = make_unit_square(6)
    bc = BoundaryConditions(femesh)
    bc.add_force(lambda p: [1.0])
    solver = Solver(femesh, Projection(), bc)
    solver.solve()

    with pytest.raises(ValueError):
        solver.adaptive_refinement(lambda s: np.ones(len(s.femesh.elements) + 1))
