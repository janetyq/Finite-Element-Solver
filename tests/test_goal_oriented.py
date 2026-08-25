"""The goal-oriented (dual-weighted-residual) error estimator: its refinement
concentrates near the quantity of interest, more strongly than the global estimator.
"""
import numpy as np

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.equations import Poisson
from fem.estimators import goal_oriented_estimator, recovery_estimator
from fem.mesh.structured import create_rect_mesh
from fem.regions import everywhere
from fem.sensitivity import PointValue
from fem.solver import Solver


def _nearest_node(space, point):
    return int(np.argmin(np.linalg.norm(space.node_coords - np.asarray(point), axis=1)))


def _poisson_solver(n=12):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=lambda p: 1.0), bc)
    solver.solve()
    return solver


def _near_far(mesh, point, near=0.15, far=0.35):
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    dist = np.linalg.norm(centroids - np.asarray(point), axis=1)
    return int((dist < near).sum()), int((dist > far).sum())


def test_indicator_peaks_near_the_quantity_of_interest():
    """The dual solution is the influence function of the point value, peaked at the
    point, so the product indicator is largest near it."""
    solver = _poisson_solver(16)
    target = (0.72, 0.72)
    qoi = PointValue(_nearest_node(solver.space, target))
    eta = goal_oriented_estimator(solver.equation, qoi).estimate(solver)

    centroids = solver.mesh.vertices[solver.mesh.elements].mean(axis=1)
    dist = np.linalg.norm(centroids - np.asarray(target), axis=1)
    # The largest-indicator element sits near the quantity of interest, not across the
    # domain: the peak of the indicator tracks the goal.
    assert dist[int(np.argmax(eta))] < 0.2


def test_refines_toward_the_quantity_of_interest_more_than_global():
    """Goal-oriented refinement concentrates elements near the point value; the global
    recovery estimator, blind to the goal, spreads them out. The goal-oriented mesh
    therefore has a higher near-to-far element ratio around the point."""
    target = (0.72, 0.72)

    goal_solver = _poisson_solver(12)
    qoi = PointValue(_nearest_node(goal_solver.space, target))
    AdaptiveRefinement(
        goal_solver, goal_oriented_estimator(goal_solver.equation, qoi),
        max_triangles=400, max_iters=6,
    ).run()

    global_solver = _poisson_solver(12)
    AdaptiveRefinement(
        global_solver, recovery_estimator(global_solver.equation),
        max_triangles=400, max_iters=6,
    ).run()

    goal_near, goal_far = _near_far(goal_solver.mesh, target)
    global_near, global_far = _near_far(global_solver.mesh, target)

    goal_ratio = goal_near / max(goal_far, 1)
    global_ratio = global_near / max(global_far, 1)
    assert goal_ratio > global_ratio
