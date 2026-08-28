"""The contract every error estimator shares, checked once for all of them.

Each estimator returns one finite, non-negative value per element. The
estimator-specific files test what each one measures; this file is the only place the
shared shape is asserted.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.equations import LinearElastic, Poisson
from fem.estimators import GoalOrientedEstimator, RecoveryEstimator, ResidualEstimator
from fem.mesh.structured import box_mesh
from fem.regions import everywhere, on_plane
from fem.sensitivity import PointValue


def _poisson(element_type):
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(8, 8))
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    equation = Poisson(source=1.0)
    return equation, equation.problem(mesh, bc, element_type=element_type)


def _elastic(element_type):
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(8, 8))
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [1.0, 0]),
    )
    equation = LinearElastic(E=200, nu=0.3)
    return equation, equation.problem(mesh, bc, element_type=element_type)


def _goal_oriented():
    return GoalOrientedEstimator(PointValue(0))


ESTIMATORS = [ResidualEstimator, RecoveryEstimator, _goal_oriented]
PROBLEMS = [_poisson, _elastic]
ELEMENTS = [LinearTriangleElement, QuadraticTriangleElement]


@pytest.mark.parametrize('element_type', ELEMENTS, ids=lambda e: e.__name__)
@pytest.mark.parametrize('problem', PROBLEMS, ids=lambda p: p.__name__.strip('_'))
@pytest.mark.parametrize('estimator', ESTIMATORS, ids=lambda f: f.__name__.strip('_'))
def test_returns_one_finite_nonnegative_value_per_element(estimator, problem, element_type):
    _, problem = problem(element_type)
    solution = problem.solve()
    eta = estimator().estimate(problem, solution)
    assert eta.shape == (len(problem.space.mesh.elements),)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)
