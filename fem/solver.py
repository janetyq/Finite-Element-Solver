"""The steady-solve facade: mesh + equation + boundary conditions -> Solution.

`Solver` is `equation.problem(mesh, bc, element_type).solve(strategy, backend)` held
as an object. The physics is the equation's, the algebra the backend's, and the
constraints the problem's.
"""
import logging

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import Equation
from fem.solution import Solution
from fem.backends import Backend
from fem.problem import Problem
from fem.solve import SolveStrategy

logger = logging.getLogger(__name__)


class Solver:
    '''One solve of `equation` on `mesh` under `boundary_conditions`.

    `strategy` and `backend` are passed to `Problem.solve`; `element_type` None is
    the linear element for the mesh.
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
        return self.problem().solve(strategy=self.strategy, backend=self.backend)
