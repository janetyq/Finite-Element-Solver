"""The steady-solve facade: mesh + equation + boundary conditions -> Solution.

`Solver` builds a `LinearProblem` from the three and hands it to `LinearSolve`.
The physics is the equation's (`Equation.operator`), the algebra the backend's, and
the constraints the problem's. `remesh` is what `AdaptiveRefinement` advances the
solver through.
"""
import logging

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import Equation, LinearElastic
from fem.solution import ElasticSolution, FieldSolution, ScalarFieldSolution, Solution
from fem.space import FunctionSpace
from fem.forms import RecoversElasticFields
from fem.backends import Backend, IterativeBackend, rigid_body_modes
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
        self.solution = self._solve_steady()
        return self.solution

    def _steady_problem(self) -> LinearProblem:
        '''The problem for a steady equation: operator + source + constraints, on the
        solver's current space.'''
        operator = self.equation.operator(self.n_components)
        return LinearProblem(self.space, operator, self.equation.source, self.boundary_conditions)

    def _backend_for(self, problem: LinearProblem) -> Backend | None:
        '''The solve backend, giving an elastic AMG solve its rigid-body near-kernel.

        An elasticity stiffness has the rigid-body modes as its low-energy near-kernel,
        and AMG needs them to keep CG's iteration count flat under refinement. They are
        restricted to the free DOFs to match the block the backend factors. An explicit
        near-kernel the caller set is left untouched.
        '''
        if isinstance(self.equation, LinearElastic) and isinstance(self.backend, IterativeBackend) \
                and self.backend.near_null_space is None:
            free = problem.constraints[0]
            # Built from the space's node coordinates, not the mesh vertices, so a P2
            # elastic solve gives AMG the rigid-body modes at its edge nodes too.
            modes = rigid_body_modes(self.space.node_coords, self.n_components)[free]
            return self.backend.with_near_null_space(modes)
        return self.backend

    def _solve_steady(self) -> Solution:
        '''Steady linear solve, packaged by what the form can recover: an
        `ElasticSolution` for a form that recovers stress, a `ScalarFieldSolution` for an
        equation naming a derived field (Poisson's flux), else a bare `FieldSolution`.'''
        logger.info('Solving steady system...')
        problem = self._steady_problem()
        u = LinearSolve(self._backend_for(problem)).solve(problem)

        if isinstance(problem.operator, RecoversElasticFields):
            return ElasticSolution.from_solve(self.space, u, problem.operator)
        if self.equation.derived_field() is not None:
            return ScalarFieldSolution.from_solve(self.space, u)
        return FieldSolution(self.mesh, self.n_components, u,
                             element_type=self.space.element_type)
