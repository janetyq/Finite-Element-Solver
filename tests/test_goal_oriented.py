"""The goal-oriented (dual-weighted-residual) error estimator: its refinement
concentrates near the quantity of interest, more strongly than the global estimator.
"""
import numpy as np

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, Dirichlet
from fem.equations import Poisson
from fem.estimators import GoalOrientedEstimator, RecoveryEstimator
from fem.mesh.structured import create_rect_mesh
from fem.regions import everywhere
from fem.sensitivity import PointValue

EQUATION = Poisson(source=lambda p: 1.0)


def _nearest_node(space, point):
    return int(np.argmin(np.linalg.norm(space.node_coords - np.asarray(point), axis=1)))


def _problem_for(mesh):
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    return EQUATION.problem(mesh, bc)


def _square(n):
    return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))


def _near_far(mesh, point, near=0.15, far=0.35):
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    dist = np.linalg.norm(centroids - np.asarray(point), axis=1)
    return int((dist < near).sum()), int((dist > far).sum())


def test_indicator_peaks_near_the_quantity_of_interest():
    """The dual solution is the influence function of the point value, peaked at the
    point, so the product indicator is largest near it."""
    mesh = _square(16)
    problem = _problem_for(mesh)
    solution = problem.solve()
    target = (0.72, 0.72)
    qoi = PointValue(_nearest_node(problem.space, target))
    eta = GoalOrientedEstimator(qoi).estimate(problem, solution)

    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    dist = np.linalg.norm(centroids - np.asarray(target), axis=1)
    # The largest-indicator element sits near the quantity of interest, not across the
    # domain: the peak of the indicator tracks the goal.
    assert dist[int(np.argmax(eta))] < 0.2


def test_refines_toward_the_quantity_of_interest_more_than_global():
    """Goal-oriented refinement concentrates elements near the point value; the global
    recovery estimator, blind to the goal, spreads them out. The goal-oriented mesh
    therefore has a higher near-to-far element ratio around the point."""
    target = (0.72, 0.72)

    mesh = _square(12)
    qoi = PointValue(_nearest_node(EQUATION.space(mesh), target))
    goal = AdaptiveRefinement(mesh, _problem_for, GoalOrientedEstimator(qoi),
                              max_triangles=400, max_iters=6)
    goal.run()

    global_ = AdaptiveRefinement(_square(12), _problem_for, RecoveryEstimator(),
                                 max_triangles=400, max_iters=6)
    global_.run()

    goal_near, goal_far = _near_far(goal.mesh, target)
    global_near, global_far = _near_far(global_.mesh, target)

    goal_ratio = goal_near / max(goal_far, 1)
    global_ratio = global_near / max(global_far, 1)
    assert goal_ratio > global_ratio
