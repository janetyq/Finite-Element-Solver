"""The contract every error estimator shares, checked once for all of them.

Each estimator returns one finite, non-negative value per element; vanishes on a field
the space represents exactly; is largest where the solution is hardest to resolve; and
drives adaptive refinement toward a localised source. The estimator-specific files test
what each one measures; this file is the only place the shared behaviour is asserted,
over every estimator and element type at once.
"""
import numpy as np
import pytest

from fem.analysis.adaptivity import AdaptiveRefinement
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.physics.equations import LinearElastic, Poisson
from fem.analysis.estimators import GoalOrientedEstimator, RecoveryEstimator, ResidualEstimator
from fem.regions import everywhere
from fem.analysis.sensitivity import PointValue
from fem.loads import Source
from helpers import cantilever_bc, localised_source, near_far_counts, pinned, problem_for, solved


def _poisson(mesh, element_type):
    return solved(Poisson(), mesh, pinned() + Source(1.0), element_type=element_type)


def _elastic(mesh, element_type):
    return solved(LinearElastic(E=200, nu=0.3), mesh, cantilever_bc(traction=(1.0, 0.0)),
                  element_type=element_type)


def _goal_oriented():
    return GoalOrientedEstimator(PointValue(np.array([0.5, 0.5])))


def _peaked_source(point):
    a = 50
    x, y = (point - 0.5).T
    r2 = x**2 + y**2
    return 4 * a * a * (1 - a * r2) * np.exp(-a * r2)


# The field each space represents exactly: linear on P1, a harmonic quadratic on P2, so a
# Dirichlet-only Poisson solve of it is the field itself and any estimate is noise.
REPRESENTABLE = {
    LinearTriangleElement: lambda p: p[:, 0],
    QuadraticTriangleElement: lambda p: p[:, 0]**2 - p[:, 1]**2,
}

# The goal-oriented estimator weights the residual by a dual solution, so it answers
# "where does the error reach the goal" rather than "where is the error": it shares the
# shape contract and the refinement loop, not the global estimators' patch and peak claims.
ESTIMATORS = [ResidualEstimator, RecoveryEstimator, _goal_oriented]
GLOBAL_ESTIMATORS = [ResidualEstimator, RecoveryEstimator]
PROBLEMS = [_poisson, _elastic]
ELEMENTS = [LinearTriangleElement, QuadraticTriangleElement]

every_estimator = pytest.mark.parametrize('estimator', ESTIMATORS, ids=lambda f: f.__name__.strip('_'))
global_estimator = pytest.mark.parametrize('estimator', GLOBAL_ESTIMATORS, ids=lambda f: f.__name__)
every_element = pytest.mark.parametrize('element_type', ELEMENTS, ids=lambda e: e.__name__)


@every_element
@pytest.mark.parametrize('problem', PROBLEMS, ids=lambda p: p.__name__.strip('_'))
@every_estimator
def test_returns_one_finite_nonnegative_value_per_element(estimator, problem, element_type, make_unit_square):
    problem, solution = problem(make_unit_square(8), element_type)
    eta = estimator().estimate(problem, solution)
    assert eta.shape == (len(problem.space.mesh.elements),)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)


@every_element
@global_estimator
def test_vanishes_on_a_field_the_space_represents_exactly(estimator, element_type, make_unit_square):
    """The patch test. The solve reproduces the field, so its flux is globally smooth: no
    interior residual, no jumps, nothing to recover, and every indicator is zero."""
    bc = Conditions(Dirichlet(everywhere(), REPRESENTABLE[element_type]))
    problem, solution = solved(Poisson(), make_unit_square(5), bc, element_type=element_type)
    eta = estimator().estimate(problem, solution)
    assert np.all(eta < 1e-10)


@every_element
@global_estimator
def test_is_largest_near_a_peaked_source(estimator, element_type, make_unit_square):
    """The indicator is largest where the solution is hardest to resolve."""
    mesh = make_unit_square(10)
    problem, solution = solved(Poisson(), mesh, pinned() + Source(_peaked_source), element_type=element_type)
    eta = estimator().estimate(problem, solution)

    distance = np.linalg.norm(mesh.vertices[mesh.elements].mean(axis=1) - 0.5, axis=1)
    assert eta[distance < 0.15].mean() > eta[distance > 0.35].mean()


@every_element
@every_estimator
def test_drives_refinement_toward_a_localised_source(estimator, element_type, make_unit_square):
    """The full loop: the estimator drives the refiner, the mesh grows, it concentrates
    near the source, and the solve keeps its element type across remeshes."""
    mesh = make_unit_square(6)
    driver = AdaptiveRefinement(
        mesh, problem_for(Poisson(), pinned() + localised_source(), element_type=element_type),
        estimator(), max_triangles=300, max_iters=5,
    )
    solution = driver.run()

    assert len(driver.mesh.elements) > len(mesh.elements)
    assert solution.element_type is element_type
    near, far = near_far_counts(driver.mesh)
    assert near > far * 0.3
