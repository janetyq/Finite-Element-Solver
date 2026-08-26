"""The nonlinear-solve facade: minimise a stored energy instead of solving Ku = b.

`EnergySolver` is to the energy path what `Solver` is to the linear one, and the
two are the same shape: hold a mesh, an equation, and a boundary-
condition spec; build a `Problem` per solve; hand it to a strategy. Here the
problem is an `EnergyProblem` and the strategy a `NewtonSolve`, because the
tangent depends on the state; the default strategy is line-searched, and a caller
may pass its own.

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
from fem.solution import Solution
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
        strategy: NewtonSolve | None = None,
    ) -> None:
        if not isinstance(equation, LinearElastic):
            raise ValueError(
                f'EnergySolver solves elastic energies; got {type(equation).__name__}.'
            )
        # The minimised energy has no external work term, so a source would be
        # silently dropped.
        if equation.source is not None:
            raise NotImplementedError(
                'EnergySolver has no external work term, so a source term would be '
                'silently dropped; a forced problem needs a LinearProblem.'
            )

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = (
            boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        )
        # The default strategy is line-searched Newton: the St-Venant-Kirchhoff energy is
        # non-convex, so a full step from the seed can raise the energy and diverge.
        # Its Hessian is indefinite near the seed, so an iterative `backend` must handle
        # that (MinresBackend), and is paired with the regularization that keeps each
        # step a descent direction; the direct default needs neither.
        if strategy is None:
            strategy = NewtonSolve(
                line_search=BacktrackingLineSearch(),
                backend=backend,
                regularization=TangentRegularization() if backend is not None else None,
            )
        self.strategy = strategy
        self.element_type = element_type
        self.space = equation.space(mesh, element_type)
        self.n_components = self.space.n_components
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
        self.space = self.equation.space(mesh, self.element_type)

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

    def solve(self) -> Solution:
        '''Minimise the energy from a seed carrying the Dirichlet values.'''
        problem = self.problem()
        _, fixed, fixed_values = problem.constraints
        u = np.zeros(self.space.n_dofs)
        u[fixed] = fixed_values

        logger.info('Initial energy: %s', self.energy(u))
        u = self.strategy.solve(problem, u0=u)
        self.solution = problem.solution(u)
        return self.solution
