"""The goal-oriented (adjoint-weighted) error estimator: its refinement
concentrates near the quantity of interest, more strongly than the global estimator.
"""
import numpy as np

from fem.analysis.adaptivity import AdaptiveRefinement
from fem.physics.equations import Poisson
from fem.analysis.estimators import GoalOrientedEstimator, RecoveryEstimator
from fem.analysis.sensitivity import PointValue
from fem.loads import Source
from helpers import near_far_counts, pinned, problem_for, solved

PROBLEM_FOR = problem_for(Poisson(), pinned() + Source(lambda p: 1.0))


def test_indicator_peaks_near_the_quantity_of_interest(make_unit_square):
    """The dual solution is the influence function of the point value, peaked at the
    point, so the product indicator is largest near it."""
    mesh = make_unit_square(16)
    problem, solution = solved(Poisson(), mesh, pinned() + Source(lambda p: 1.0))
    target = (0.72, 0.72)
    qoi = PointValue(np.asarray(target))
    eta = GoalOrientedEstimator(qoi).estimate(problem, solution)

    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    dist = np.linalg.norm(centroids - np.asarray(target), axis=1)
    # The largest-indicator element sits near the quantity of interest, not across the
    # domain: the peak of the indicator tracks the goal.
    assert dist[int(np.argmax(eta))] < 0.2


def test_refines_toward_the_quantity_of_interest_more_than_global(make_unit_square):
    """Goal-oriented refinement concentrates elements near the point value; the global
    recovery estimator, blind to the goal, spreads them out. The goal-oriented mesh
    therefore has a higher near-to-far element ratio around the point."""
    target = (0.72, 0.72)
    qoi = PointValue(np.asarray(target))
    goal = AdaptiveRefinement(make_unit_square(12), PROBLEM_FOR, GoalOrientedEstimator(qoi),
                              max_triangles=400, max_iters=6)
    goal.run()

    global_ = AdaptiveRefinement(make_unit_square(12), PROBLEM_FOR, RecoveryEstimator(),
                                 max_triangles=400, max_iters=6)
    global_.run()

    goal_near, goal_far = near_far_counts(goal.mesh, target, near=0.15)
    global_near, global_far = near_far_counts(global_.mesh, target, near=0.15)

    goal_ratio = goal_near / max(goal_far, 1)
    global_ratio = global_near / max(global_far, 1)
    assert goal_ratio > global_ratio
