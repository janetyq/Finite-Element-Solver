"""The steady-solve facade: mesh + equation + boundary conditions -> Solution.

`Solver` composes rather than computes. It holds the three things a steady solve
needs, builds a `LinearProblem` from them, and hands it to `LinearSolve`. The
physics is the equation's (`Equation.operator`), the algebra is the backend's, and
the constraints are the problem's; what is left here is the composition itself plus
`remesh`, the seam an `AdaptiveRefinement` driver advances the solver through.
"""
import logging

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import Equation, LinearElastic
from fem.solution import ElasticSolution, FieldSolution, Solution
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
        # The linear-algebra backend for the steady solve: direct by default, or an
        # IterativeBackend for a large SPD system. A steady LinearElastic / Poisson is
        # SPD; the facade forwards it to LinearSolve untouched.
        self.backend = backend
        # The element order, `None` meaning the linear element for the mesh's node
        # count. Pass `QuadraticTriangleElement` for a P2 solve (O(h^3)); the
        # adaptive-refinement estimator is P1-only, so a refined P2 solve is not yet
        # supported, but a single P2 solve is.
        self.element_type = element_type
        # Derived, never passed: the component count follows from the equation's
        # field and the mesh, so a space that disagrees with the equation it is
        # solving is not constructible here.
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.space = FunctionSpace(mesh, element_type, n_components=self.n_components)
        # The most recent solve, so an adaptive-refinement estimator can read it.
        self.solution: Solution | None = None

    def remesh(self, mesh: Mesh) -> None:
        '''Rebind the solver to a new mesh, rebuilding the space and re-resolving BCs.

        A refined mesh renumbers vertices, so the space -- which owns cached
        operators sized to the old mesh -- is rebuilt from its specification
        rather than carried over. Nothing index-keyed survives here: the boundary
        conditions are resolved by the `LinearProblem` built for each solve. This
        is what lets an outer driver (AdaptiveRefinement) advance the solver
        across meshes without reaching into its state.
        '''
        self.mesh = mesh
        self.space = FunctionSpace(mesh, self.element_type, n_components=self.n_components)

    def solve(self) -> Solution:
        self.solution = self._solve_steady()
        return self.solution

    def _steady_problem(self) -> LinearProblem:
        '''The composition for a steady equation: operator + source + constraints.

        The equation supplies its own operator, so this stays free of any "which
        PDE is this?" branch. Built on the solver's own space, so adaptive
        refinement (which rebuilds the space) is picked up on the next solve.
        '''
        operator = self.equation.operator(self.n_components)
        return LinearProblem(self.space, operator, self.equation.source, self.boundary_conditions)

    def _backend_for(self, problem: LinearProblem) -> Backend | None:
        '''The solve backend, giving an elastic AMG solve its rigid-body near-kernel.

        A vector elasticity stiffness has the rigid-body modes as its low-energy
        near-kernel; AMG needs them to keep CG's iteration count flat under
        refinement. This is the one solve detail that depends on *which* equation is
        being solved, so the equation-aware facade supplies it -- restricted to the
        free DOFs, to match the block the backend factors -- rather than the
        physics-agnostic backend guessing. An explicit near-kernel the caller set is
        left untouched; the scalar Laplacian family needs none.
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
        '''Steady linear solve, through the composition core.

        A LinearProblem hands a matrix, a load, and the constraints to LinearSolve;
        a form that can recover derived fields additionally yields an
        ElasticSolution rather than a bare FieldSolution. The capability is asked
        for rather than the class named, so a form this facade has never heard of
        reports its stresses through the same path.
        '''
        logger.info('Solving steady system...')
        problem = self._steady_problem()
        u = LinearSolve(self._backend_for(problem)).solve(problem)

        if isinstance(problem.operator, RecoversElasticFields):
            return ElasticSolution.from_solve(self.space, u, problem.operator)
        return FieldSolution(self.mesh, self.n_components, u)
