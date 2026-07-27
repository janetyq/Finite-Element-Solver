"""Density-based (SIMP) topology optimization, as a driver over the solve core.

The optimizer owns a solve strategy and rebuilds a fresh `Problem` each iteration
from the current density -- it never mutates a shared equation. The density scales
the modulus, `E(rho) = rho^p E_0`, which becomes a fresh `LinearElasticMaterial`
per iteration; the compliance and its sensitivity drive an optimality-criterion
update of the density.

The objective is an injected object (`MinCompliance`, `TargetCompliance`) rather
than a string resolved through a `_select_*` dispatch, and the optimization method
(the OC update) is a method here. That replaces the plugin-shaped machinery -- an
ignored args bag, an objective value that was never evaluated -- with the one
quantity that is actually used: the sensitivity.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fem.boundary import BoundaryConditions
from fem.forms import LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.numerics import calculate_smoothing_matrix
from fem.problem import LinearProblem
from fem.solution import ElasticSolution
from fem.solve import LinearSolve, SolveStrategy
from fem.equations import LinearElastic
from fem.space import FunctionSpace
from fem.typing import DofVector, ElementField, FieldValue


class Objective(Protocol):
    '''Maps a compliance field and density to a per-element sensitivity.'''

    def gradient(self, compliance: ElementField, rho: ElementField, penalty: float) -> ElementField:
        ...


@dataclass(frozen=True)
class MinCompliance:
    '''Minimize total compliance: the sensitivity is d(compliance)/d(rho).'''

    def gradient(self, compliance: ElementField, rho: ElementField, penalty: float) -> ElementField:
        # For E(rho) = rho^p E_0 the element compliance is linear in E, so its
        # derivative is p/rho times it. The exponent must match the one set_rho
        # raised rho to, hence penalty in both.
        return compliance * penalty / rho


@dataclass(frozen=True)
class TargetCompliance:
    '''Drive the total compliance toward `target` (a least-squares objective).'''
    target: float

    def gradient(self, compliance: ElementField, rho: ElementField, penalty: float) -> ElementField:
        residual = compliance.sum() - self.target
        return (compliance * penalty / rho) * 2 * residual


@dataclass(frozen=True, eq=False)
class TopologyHistory:
    '''The per-iteration series a topology optimization produces.

    Replaces the old `Solution.combine_solutions` `_list`-suffix convention: the
    quantities that vary across iterations are typed, discoverable lists rather
    than string keys probed with try/except.
    '''
    rho: list[ElementField]
    u: list[DofVector]
    von_mises: list[ElementField]   # one scalar per element, per iteration
    compliance: list[ElementField]


logger = logging.getLogger(__name__)


class TopologyOptimizer:
    '''Density-based topology optimization.

    Owns a solve strategy (any `SolveStrategy`, defaulting to `LinearSolve`) and
    iteratively updates a density field to minimize an objective for a
    linear-elastic problem under a volume constraint.
    '''

    def __init__(
        self,
        mesh: Mesh,
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions | None = None,
        iters: int = 10,
        volume_frac: float = 1.0,
        smoothing_radius: float = 0.1,
        penalty: float = 3.0,
        objective: Objective | None = None,
        strategy: SolveStrategy | None = None,
    ) -> None:
        assert isinstance(equation, LinearElastic), \
            'TopologyOptimizer only supports LinearElastic equations'
        self.mesh = mesh
        self.bc = boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        # The solid-material modulus every density scaling is measured against, and
        # the rest of the problem spec. Kept as data, not as a mutable equation.
        self.base_E: float | ElementField = equation.E
        self.nu = equation.nu
        self.source: FieldValue = equation.source

        self.iters = iters
        self.volume_frac = volume_frac
        # The SIMP exponent p in E(rho) = rho^p * E_0. Above 1 it makes intermediate
        # densities inefficient per unit volume, which drives the design toward
        # black-and-white; 3 is the standard choice.
        self.penalty = penalty
        self.objective: Objective = objective if objective is not None else MinCompliance()

        self.strategy: SolveStrategy = strategy if strategy is not None else LinearSolve()
        # Geometry-only space, for element volumes (the volume constraint).
        self.space = FunctionSpace(mesh, n_components=mesh.spatial_dim)
        self.rho: ElementField = np.full(len(self.mesh.elements), self.volume_frac)
        self.smoothing_matrix = calculate_smoothing_matrix(self.mesh, r=smoothing_radius)

        self._last: ElasticSolution | None = None   # most recent single-iteration solve
        self.history: TopologyHistory | None = None  # the per-iteration series

    @property
    def scaled_modulus(self) -> ElementField:
        '''The SIMP-scaled modulus E(rho) = rho^p * E_0 for the current density.'''
        return self.rho**self.penalty * self.base_E

    def set_rho(self, rho: ElementField) -> None:
        self.rho = rho

    def _volume_fraction(self, rho: ElementField) -> float:
        '''Volume-weighted mean of a per-element field.'''
        volumes = self.space.element_volumes
        return float((volumes * rho).sum() / volumes.sum())

    def _solve(self) -> ElasticSolution:
        '''Solve the elastic problem at the current density and recover compliance.

        A fresh material and problem each call -- the density scales the modulus,
        so there is no shared state to mutate. The numerical path (LinearProblem ->
        LinearSolve -> derived_fields) is the same one the Solver facade runs, reusing
        the optimizer's cached space rather than rebuilding geometry each iteration.
        '''
        form = LinearElasticForm(LinearElasticMaterial(self.scaled_modulus, self.nu))
        problem = LinearProblem(self.space, form, self.source, self.bc)
        u = self.strategy.solve(problem)

        self._last = ElasticSolution.from_solve(self.space, u, form)
        return self._last

    def oc_density(
        self,
        sensitivity: ElementField,
        volume_frac: float,
        max_iters: int = 100,
        tol: float = 1e-8,
    ) -> ElementField:
        # sensitivity is the gradient of the compliance with respect to the density.
        # Bisect on the Lagrange multiplier until the volume constraint is met.
        lo, hi = 0.0, 1e15  # search interval
        rho_new = self.rho
        for _ in range(max_iters):
            m = 0.5 * (lo + hi)
            rho_new = self.rho * np.sqrt(sensitivity / m)
            rho_new = np.clip(rho_new, self.rho - 0.1, self.rho + 0.1)  # change limit
            rho_new = np.clip(rho_new, 1e-6, 1)

            if self._volume_fraction(rho_new) < volume_frac:
                hi = m
            else:
                lo = m

            if hi - lo <= tol * hi:
                break
        return rho_new

    def solve(self, on_iteration: Callable[[int, ElasticSolution], None] | None = None) -> TopologyHistory:
        rho_series: list[ElementField] = []
        u_series: list[DofVector] = []
        von_mises_series: list[ElementField] = []
        compliance_series: list[ElementField] = []
        for i in range(self.iters):
            solution = self._solve()
            rho_series.append(self.rho)
            u_series.append(solution.u)
            # A scalar per iteration, not the whole stress state; `_last` keeps
            # the tensors from the final solve.
            von_mises_series.append(solution.von_mises)
            compliance_series.append(solution.compliance)

            # Log and, if the caller wants it, hand off to their own visualization --
            # this class has no business knowing how (or whether) to plot itself.
            self._log_iteration(i, solution)
            if on_iteration is not None:
                on_iteration(i, solution)

            sensitivity = self.objective.gradient(solution.compliance, self.rho, self.penalty)
            smoothed = self.smoothing_matrix @ sensitivity
            self.set_rho(self.oc_density(smoothed, self.volume_frac))

        self.history = TopologyHistory(rho_series, u_series, von_mises_series, compliance_series)
        return self.history

    def _log_iteration(self, iteration: int, solution: ElasticSolution) -> None:
        max_displacement = np.max(solution.u, axis=0)
        compliance = solution.compliance.sum()
        volume_fraction = self._volume_fraction(self.rho)
        logger.info('Iteration %d: total compliance = %.4f, max displacement = %s, volume fraction = %.4f',
                    iteration, compliance, max_displacement, volume_fraction)

    def deformed_mesh(self) -> Mesh:
        '''The deformed mesh from the most recent solve (for post-processing).'''
        if self._last is None:
            raise RuntimeError('no solve yet; call solve() first')
        return self._last.deformed_mesh()
