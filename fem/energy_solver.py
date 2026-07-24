import logging

import numpy as np

from fem.boundary import BoundaryConditions
from fem.energies import StVenantKirchhoff
from fem.forms import EnergyForm
from fem.mesh.mesh import Mesh
from fem.problem import EnergyProblem
from fem.solve import NewtonSolve
from fem.solution import Solution
from fem.solver import Equation, LinearElastic
from fem.space import FunctionSpace
from fem.typing import DofVector, SparseMatrix

logger = logging.getLogger(__name__)

class EnergySolver:
    def __init__(
        self,
        mesh: Mesh,
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions,
        verbose: bool = True,
    ) -> None:
        assert isinstance(equation, LinearElastic), "EnergySolver only supports linear elastic equation"
        # note: does not exactly match linear elastic solve for larger deformations bc doesn't use small strain approx

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.space = FunctionSpace(mesh, n_components=self.n_components)
        self.solution = Solution(mesh, self.n_components)
        # This solver minimizes the internal elastic energy and never builds a
        # load vector, so a source term would be accepted and then quietly
        # ignored -- the answer would just be the unforced one.
        if equation.source is not None:
            raise NotImplementedError(
                'EnergySolver does not support a source term yet: it minimizes the '
                'internal energy only, with no external work term, so the source '
                'would be silently dropped. Use Solver for forced problems.'
            )

        self.resolved_bc = self.boundary_conditions.resolve(mesh, self.n_components)
        self.free = self.resolved_bc.free_idxs
        self.fixed = self.resolved_bc.fixed_idxs
        self.fixed_values = self.resolved_bc.fixed_values

        self.energy_density = self._select_energy(equation)
        self.form = EnergyForm(self.energy_density)

        # other options, TODO: inheritance to hide these options
        self.verbose = verbose

        # TODO: flat u + bc handling weird

        # check_gradient(self.energy, self.energy_gradient, len(self.mesh.vertices) * self.n_components)
        # check_hessian(self.energy_gradient, self.energy_hessian, len(self.mesh.vertices) * self.n_components)

    def _select_energy(self, equation: Equation) -> StVenantKirchhoff:
        if not isinstance(equation, LinearElastic):
            raise ValueError(f"Unsupported equation type: {type(equation).__name__}")
        # `LinearElastic.E` may be per-element -- TopologyOptimizer sets a
        # density-scaled modulus -- but a density carries one pair of Lame
        # parameters for the whole mesh, and an array lamb broadcasts wrongly
        # against the constant d2W/dS2. `Solver` is the path for varying moduli.
        if not isinstance(equation.E, int | float):
            raise NotImplementedError(
                'EnergySolver needs a scalar Youngs modulus, got a per-element '
                'array. Use Solver for density-scaled moduli.'
            )
        return StVenantKirchhoff(equation.E, equation.nu)

    # energy / gradient / hessian are the raw, unconstrained quantities: the total
    # energy Pi(u), its gradient (nonzero at fixed DOFs -- the reaction forces),
    # and its Hessian. The Dirichlet constraint is applied by the DiscreteSystem
    # in newton_solve, not baked into these, the same way Solver assembles a raw K
    # and eliminates in solve_linear_system.
    def energy(self, u: DofVector) -> float:
        return self.space.total_energy(self.form, u)

    def energy_gradient(self, u: DofVector) -> DofVector:
        return self.space.assemble_residual(self.form, u)

    def energy_hessian(self, u: DofVector) -> SparseMatrix:
        return self.space.assemble_tangent(self.form, u)

    def solve(self, max_iters: int = 100) -> Solution:
        u = np.zeros(len(self.mesh.vertices) * self.n_components)
        u[self.fixed] = self.fixed_values
        logger.info("Initial energy: %s", self.energy(u))
        u = self.newton_solve(u, max_iters)
        self.solution.set_values("u", u)
        return self.solution

    def newton_solve(self, u: DofVector, max_iters: int = 100) -> DofVector:
        '''Minimise the internal energy from the seed `u` via the shared Newton engine.

        The one-line body is the point: this solver's assembly (residual = energy
        gradient, tangent = energy Hessian) is exactly `EnergyProblem`'s, so the
        Newton loop is `NewtonSolve` rather than a second copy of it. The Dirichlet
        values on `u` are preserved; the increment is pinned to zero at those DOFs.
        '''
        problem = EnergyProblem(self.space, self.form, self.boundary_conditions)
        return NewtonSolve(max_iters=max_iters).solve(problem, u0=u)
