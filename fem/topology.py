import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fem.boundary import BoundaryConditions
from fem.numerics import calculate_smoothing_matrix
from fem.solver import Solver, LinearElastic
from fem.solution import Solution
from fem.typing import ElementField

if TYPE_CHECKING:
    from fem.mesh.femesh import FEMesh

logger = logging.getLogger(__name__)

class TopologyOptimizer:
    '''
    Density based topology optimization

    Creates a solver and iteratively updates density field to minimize some objective 
    for some equation and boundary conditions.
    '''
    def __init__(
        self,
        femesh: 'FEMesh',
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions,
        iters: int = 10,
        volume_frac: float = 1.0,
        smoothing_radius: float = 0.1,
    ) -> None:
        assert isinstance(equation, LinearElastic), \
            'TopologyOptimizer only supports LinearElastic equations'
        self.femesh = femesh
        # Narrowed once here: the assert above guarantees it, and every later use
        # of E goes through this rather than the Solver's loosely-typed equation.
        self.equation: LinearElastic = equation
        self.orig_equation: LinearElastic = equation.copy()
        self.solver = Solver(femesh, equation, boundary_conditions)
        self.solution = Solution(femesh, self.orig_equation.dim)

        self.iters = iters
        self.volume_frac = volume_frac

        self.rho: ElementField
        self.set_rho(np.full(len(self.femesh.elements), self.volume_frac))

        self.smoothing_matrix = calculate_smoothing_matrix(self.femesh, r=smoothing_radius)

    def set_rho(self, rho: ElementField) -> None:
        self.rho = rho
        self.equation.E = self.rho**3 * self.orig_equation.E

    def oc_density(
        self,
        sensitivity: ElementField,
        volume_frac: float,
        max_iters: int = 100,
        tol: float = 1e-8,
    ) -> ElementField:
        # sensitivity is the gradient of the compliance with respect to the density
        # Bisect on the Lagrange multiplier until the volume constraint is met.
        # Bounded iterations + a relative tolerance on the bracket: the previous
        # `while lo*(1+1e-15) < hi` never tightened while lo stayed at 0, so it
        # only stopped once hi had halved all the way down to zero.
        lo, hi = 0.0, 1e15 # search interval
        rho_new = self.rho
        for _ in range(max_iters):
            m = 0.5*(lo+hi)
            rho_new = self.rho * np.sqrt(sensitivity / m)
            rho_new = np.clip(rho_new, self.rho - 0.1, self.rho + 0.1) # change limit
            rho_new = np.clip(rho_new, 1e-6, 1)

            if self.solver.femesh.calculate_mean_value(rho_new) < volume_frac:
                hi = m
            else:
                lo = m

            if hi - lo <= tol * hi:
                break
        return rho_new

    def solve(
        self,
        objective_name: Literal['min_compliance', 'target_compliance'] = 'min_compliance',
        objective_args: Sequence[Any] | None = None,
        optimization_method: Literal['oc'] = 'oc',
        optimization_args: Sequence[Any] | None = None,
        on_iteration: Callable[[int, Solution], None] | None = None,
    ) -> Solution:

        objective_func, gradient_func = self._select_objective(objective_name)
        optimization_func = self._select_optimization(optimization_method, optimization_args)

        solution_list = []
        for iter in range(self.iters):
            # solve
            solution = self.solver.solve()
            solution.set_values('rho', self.rho)
            solution_list.append(solution)

            # log and, if the caller wants it, hand off to their own visualization --
            # this class has no business knowing how (or whether) to plot itself.
            self._log_iteration(iter, solution)
            if on_iteration is not None:
                on_iteration(iter, solution)

            # update rho
            smoothed_gradient = self.smoothing_matrix @ gradient_func(objective_args)
            updated_rho = optimization_func(smoothed_gradient)
            self.set_rho(updated_rho)

        self.solution = Solution.combine_solutions(solution_list)
        return self.solution

    def compliance(self, args: Sequence[Any] | None) -> float:
        return self.solver.solution.values['compliance'].sum()

    def compliance_gradient(self, args: Sequence[Any] | None) -> ElementField:
        return self.solver.solution.values['compliance'] * 3/self.rho

    def target_compliance_objective(self, args: Sequence[Any]) -> float:
        target = args[0]
        return (self.compliance(args) - target)**2

    def target_compliance_gradient(self, args: Sequence[Any]) -> ElementField:
        target = args[0]
        return self.compliance_gradient(args) * 2 * (self.compliance(args) - target)

    def _select_objective(self, objective_name: str) -> tuple[Callable[..., float], Callable[..., ElementField]]:
        if objective_name == 'min_compliance':
            return self.compliance, self.compliance_gradient
        elif objective_name == 'target_compliance':
            return self.target_compliance_objective, self.target_compliance_gradient
        else:
            raise ValueError(f'Invalid objective: {objective_name}')

    def _select_optimization(self, optimization_method: str, optimization_args: Sequence[Any] | None) -> Callable[[ElementField], ElementField]:
        if optimization_method == 'oc':
            return lambda gradient: self.oc_density(gradient, self.volume_frac)
        else:
            raise ValueError(f'Invalid optimization method: {optimization_method}')

    def _log_iteration(self, iter: int, solution: Solution) -> None:
        max_displacement = np.max(solution.values['u'], axis=0)
        compliance = solution.values['compliance'].sum()
        volume_fraction = self.solver.femesh.calculate_mean_value(self.rho)
        logger.info('Iteration %d: total compliance = %.4f, max displacement = %s, volume fraction = %.4f',
                    iter, compliance, max_displacement, volume_fraction)

    def _get_deformed_mesh(self, iter_idx: int = -1) -> 'FEMesh':
        try:
            u = self.solution.values['u_list'][iter_idx]
        except (KeyError, IndexError):
            u = self.solver.solution.values['u']
        return self.solver.solution.get_deformed_mesh(u)
