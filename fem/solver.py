"""The steady-solve facade: mesh + equation + boundary conditions -> Solution.

`Solver` builds a `LinearProblem` from the three (`Equation.problem`) and hands it to
`LinearSolve`; the problem packages the result. The physics is the equation's, the
algebra the backend's, and the constraints the problem's.
"""
import logging

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import Equation
from fem.solution import Solution
from fem.backends import Backend
from fem.problem import LinearProblem
from fem.solve import LinearSolve

logger = logging.getLogger(__name__)


class Solver:
    def __init__(
        self,
        mesh: Mesh,
        equation: Equation,
        boundary_conditions: BoundaryConditions | None = None,
        backend: Backend | None = None,
        element_type: type[Element] | None = None,
    ) -> None:
        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        # Direct by default, or an IterativeBackend for a large SPD system.
        self.backend = backend
        # `None` means the linear element for the mesh's node count; pass
        # `QuadraticTriangleElement` for a P2 solve.
        self.element_type = element_type
        self.space = equation.space(mesh, element_type)
        self.n_components = self.space.n_components

    def problem(self) -> LinearProblem:
        '''The composition on the solver's space: operator, source, constraints.'''
        return self.equation.problem(self.space, self.boundary_conditions)

    def solve(self) -> Solution:
        logger.info('Solving steady system...')
        problem = self.problem()
        return problem.solution(LinearSolve(self.backend).solve(problem))
