"""The steady-solve facade: mesh + equation + boundary conditions -> Solution.

`Solver` builds a `LinearProblem` from the three and hands it to `LinearSolve`; the
problem packages the result. The physics is the equation's (`Equation.operator`), the
algebra the backend's, and the constraints the problem's. `remesh` is what
`AdaptiveRefinement` advances the solver through.
"""
import logging

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import Equation
from fem.solution import Solution
from fem.space import FunctionSpace
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
        # Derived from the equation's field and the mesh, never passed.
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.space = FunctionSpace(mesh, element_type, n_components=self.n_components)
        # The most recent solve, so an adaptive-refinement estimator can read it.
        self.solution: Solution | None = None

    def remesh(self, mesh: Mesh) -> None:
        '''Rebind the solver to a new mesh, rebuilding the space.

        A refined mesh renumbers vertices, so the space and its cached operators are
        rebuilt; the boundary conditions are geometric and resolve again at the next
        solve.
        '''
        self.mesh = mesh
        self.space = FunctionSpace(mesh, self.element_type, n_components=self.n_components)

    def solve(self) -> Solution:
        logger.info('Solving steady system...')
        problem = self._steady_problem()
        u = LinearSolve(self.backend).solve(problem)
        self.solution = problem.solution(u)
        return self.solution

    def _steady_problem(self) -> LinearProblem:
        '''The problem for a steady equation: operator + source + constraints, on the
        solver's current space.'''
        operator = self.equation.operator(self.n_components)
        return LinearProblem(self.space, operator, self.equation.source, self.boundary_conditions)
