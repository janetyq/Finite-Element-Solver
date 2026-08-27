"""The steady-solve facade: mesh + equation + boundary conditions -> Solution.

`Solver` builds a `Problem` from the three (`Equation.problem`) and hands it to a
strategy; the problem packages the result. The physics is the equation's, the
algebra the backend's, and the constraints the problem's.
"""
import logging

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import Equation
from fem.solution import Solution
from fem.backends import Backend
from fem.problem import Problem
from fem.solve import SolveStrategy, default_strategy

logger = logging.getLogger(__name__)


class Solver:
    '''One solve of `equation` on `mesh` under `boundary_conditions`.

    `strategy` None is `default_strategy`: a direct linear solve for a constant
    tangent (small-strain elasticity, the scalar family), line-searched Newton
    otherwise (Green-Lagrange elasticity). `backend` selects the linear algebra of
    whichever strategy that is. `element_type` None is the linear element for the
    mesh; pass `QuadraticTriangleElement` for a P2 solve.
    '''

    def __init__(
        self,
        mesh: Mesh,
        equation: Equation,
        boundary_conditions: BoundaryConditions | None = None,
        backend: Backend | None = None,
        element_type: type[Element] | None = None,
        strategy: SolveStrategy | None = None,
    ) -> None:
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        self.backend = backend
        self.strategy = strategy
        self.space = equation.space(mesh, element_type)

    def problem(self) -> Problem:
        '''The composition on the solver's space: operator, source, constraints.'''
        return self.equation.problem(self.space, self.boundary_conditions)

    def solve(self) -> Solution:
        logger.info('Solving steady system...')
        problem = self.problem()
        strategy = self.strategy if self.strategy is not None else default_strategy(problem, self.backend)
        return problem.solution(strategy.solve(problem))
