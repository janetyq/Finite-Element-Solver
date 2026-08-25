"""The contract every error estimator shares, checked once for all of them.

Each estimator returns one finite, non-negative value per element and refuses a solver
that has not been solved. The estimator-specific files test what each one measures;
this file is the only place the shared shape and precondition are asserted.
"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.equations import LinearElastic, Poisson
from fem.estimators import goal_oriented_estimator, recovery_estimator, residual_estimator
from fem.mesh.structured import create_rect_mesh
from fem.regions import everywhere, on_plane
from fem.sensitivity import PointValue
from fem.solver import Solver


def _poisson(element_type):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(8, 8))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    return Solver(mesh, Poisson(source=1.0), bc, element_type=element_type)


def _elastic(element_type):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(8, 8))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])
    return Solver(mesh, LinearElastic(E=200, nu=0.3), bc, element_type=element_type)


def _goal_oriented(equation):
    return goal_oriented_estimator(equation, PointValue(0))


ESTIMATORS = [residual_estimator, recovery_estimator, _goal_oriented]
PROBLEMS = [_poisson, _elastic]
ELEMENTS = [LinearTriangleElement, QuadraticTriangleElement]


@pytest.mark.parametrize('element_type', ELEMENTS, ids=lambda e: e.__name__)
@pytest.mark.parametrize('problem', PROBLEMS, ids=lambda p: p.__name__.strip('_'))
@pytest.mark.parametrize('estimator', ESTIMATORS, ids=lambda f: f.__name__.strip('_'))
def test_returns_one_finite_nonnegative_value_per_element(estimator, problem, element_type):
    solver = problem(element_type)
    solver.solve()
    eta = estimator(solver.equation).estimate(solver)
    assert eta.shape == (len(solver.mesh.elements),)
    assert np.all(np.isfinite(eta))
    assert np.all(eta >= 0)


@pytest.mark.parametrize('estimator', ESTIMATORS, ids=lambda f: f.__name__.strip('_'))
def test_requires_a_solved_system(estimator):
    solver = _poisson(LinearTriangleElement)   # not solved
    with pytest.raises(ValueError, match='requires a solved system'):
        estimator(solver.equation).estimate(solver)
