"""The nonlinear-solve facade: minimise a stored energy instead of solving Ku = b.

`EnergySolver` is to the energy path what `Solver` is to the linear one, and the
two are deliberately the same shape: hold a mesh, an equation, and a boundary-
condition spec; build a `Problem` per solve; hand it to a strategy. Here the
problem is an `EnergyProblem` and the strategy is `NewtonSolve`, because the
tangent depends on the state.

Nothing index-keyed lives on the solver. The DOF partition belongs to the
`EnergyProblem` built for the current mesh, which lets `remesh` rebuild
everything from the mesh-independent specification rather than carrying stale
indices across a refinement.
"""
import logging
from typing import TYPE_CHECKING

import numpy as np

from fem.backends import Backend
from fem.boundary import BoundaryConditions
from fem.equations import LinearElastic
from fem.forms import EnergyForm
from fem.mesh.mesh import Mesh
from fem.problem import EnergyProblem
from fem.solve import BacktrackingLineSearch, NewtonSolve, TangentRegularization
from fem.solution import ElasticSolution, Solution
from fem.space import FunctionSpace
from fem.typing import DofVector, SparseMatrix

if TYPE_CHECKING:
    from fem.elements import Element

logger = logging.getLogger(__name__)


class EnergySolver:
    '''Solve for the displacement that minimises the internal elastic energy.'''

    def __init__(
        self,
        mesh: Mesh,
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions | None = None,
        backend: Backend | None = None,
        element_type: 'type[Element] | None' = None,
    ) -> None:
        if not isinstance(equation, LinearElastic):
            raise ValueError(
                f'EnergySolver solves elastic energies; got {type(equation).__name__}.'
            )
        # This solver minimizes the internal elastic energy and never builds a
        # load vector, so a source term would be accepted and then quietly
        # ignored: the answer would just be the unforced one.
        if equation.source is not None:
            raise NotImplementedError(
                'EnergySolver does not support a source term yet: it minimizes the '
                'internal energy only, with no external work term, so the source '
                'would be silently dropped. Use Solver for forced problems.'
            )

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = (
            boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        )
        # The linear-algebra backend for each Newton tangent solve. The St-Venant-Kirchhoff
        # Hessian is indefinite near the seed, so an iterative backend must be indefinite-
        # capable (MinresBackend, not the SPD-only CG). solve() pairs any backend with the
        # regularization that keeps each step a descent direction. Direct by default, which
        # handles the indefinite tangent unaided.
        self.backend = backend
        # Derived, never passed: the component count follows from the equation's
        # field and the mesh, so a space that disagrees with the equation it is
        # solving is not constructible here.
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.element_type = element_type
        self.space = FunctionSpace(mesh, element_type, n_components=self.n_components)
        # The equation names its own material law; this facade only asks for it.
        self.form = EnergyForm(equation.energy_density())
        # The most recent solve, so an adaptive-refinement estimator can read it.
        self.solution: Solution | None = None

    def remesh(self, mesh: Mesh) -> None:
        '''Rebind the solver to a new mesh, rebuilding the space.

        The mirror of `Solver.remesh`: a refined mesh renumbers vertices, so every
        derived object is rebuilt from its specification rather than carried over.
        The boundary conditions need no rebinding here because the `EnergyProblem`
        resolves them per solve.
        '''
        self.mesh = mesh
        self.space = FunctionSpace(mesh, self.element_type, n_components=self.n_components)

    def problem(self) -> EnergyProblem:
        '''The composition for the current mesh: space + energy form + constraints.

        Built per call rather than cached, so a `remesh` is picked up on the next
        solve and the resolved constraints are never older than the space.
        '''
        return EnergyProblem(self.space, self.form, self.boundary_conditions)

    # energy / gradient / hessian are the raw, unconstrained quantities: the total
    # energy Pi(u), its gradient (nonzero at fixed DOFs, the reaction forces),
    # and its Hessian. The Dirichlet constraint is applied by the DiscreteSystem
    # inside NewtonSolve, not baked into these.
    def energy(self, u: DofVector) -> float:
        return self.space.total_energy(self.form, u)

    def energy_gradient(self, u: DofVector) -> DofVector:
        return self.space.assemble_residual(self.form, u)

    def energy_hessian(self, u: DofVector) -> SparseMatrix:
        return self.space.assemble_tangent(self.form, u)

    def solve(self, max_iters: int = 100) -> Solution:
        '''Minimise the energy from a seed carrying the Dirichlet values.

        The seed is built from the problem's own constraints rather than from a
        partition stored on the solver, so it cannot disagree with the system
        `NewtonSolve` goes on to eliminate.
        '''
        problem = self.problem()
        _, fixed, fixed_values = problem.constraints
        u = np.zeros(self.space.n_dofs)
        u[fixed] = fixed_values

        logger.info('Initial energy: %s', self.energy(u))
        # Line-searched: the St-Venant–Kirchhoff energy is non-convex, so a full Newton
        # step from this seed can raise the energy and diverge. Backtracking on Π(u)
        # keeps each step a descent, at no cost near the solution where alpha = 1.
        # An iterative backend is paired with the regularization that keeps each step a
        # descent direction on the indefinite tangent; the direct default needs neither and
        # is left unregularized, so its path (and the recorded results) are unchanged.
        regularization = TangentRegularization() if self.backend is not None else None
        newton = NewtonSolve(
            max_iters=max_iters,
            line_search=BacktrackingLineSearch(),
            backend=self.backend,
            regularization=regularization,
        )
        u = newton.solve(problem, u0=u)
        # The energy form recovers Cauchy stress from the same derivative chain
        # Newton just used, so the nonlinear path reports the stress state the
        # linear one does rather than displacement alone.
        self.solution = ElasticSolution.from_solve(self.space, u, self.form)
        return self.solution
