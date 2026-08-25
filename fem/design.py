"""Design optimization over the adjoint core: objective + constraint -> optimizer.

The general sibling of `TopologyOptimizer`: drives an arbitrary `QuantityOfInterest`
(compliance, a point deflection, ...) over a SIMP density field, using
`fem.sensitivity` for the gradient and the optimality-criteria (OC) update that
`TopologyOptimizer` shares.

Scope: SIMP density design of a linear-elastic problem under a single volume
constraint, the setting where OC applies. A general constrained solver over
`scipy.optimize` is noted in `attic/fem-adjoint-sensitivity-design-2026-08-18.md`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fem.boundary import BoundaryConditions
from fem.forms import PrecomputedForm
from fem.problem import LinearProblem
from fem.sensitivity import (
    Compliance,
    DensityField,
    QuantityOfInterest,
    SensitivityAnalysis,
)
from fem.space import FunctionSpace
from fem.typing import DofVector, ElementField, FieldValue, FloatArray, SparseMatrix

logger = logging.getLogger(__name__)


def optimality_criteria_update(
    rho: ElementField,
    sensitivity: ElementField,
    volumes: FloatArray,
    volume_frac: float,
    move: float = 0.1,
    max_iters: int = 100,
    tol: float = 1e-8,
) -> ElementField:
    '''One OC density step: bisect a Lagrange multiplier to meet the volume target.

    `sensitivity` is the objective's negative gradient with respect to density (positive
    where adding material helps), which the OC heuristic `rho * sqrt(sensitivity / m)`
    needs. Each candidate is capped by the `move` limit and clipped to `[1e-6, 1]`, and
    the multiplier `m` is bisected until the volume-weighted mean density meets
    `volume_frac`. Extracted here so `TopologyOptimizer` and `DesignOptimizer` share one
    update rather than two copies that could drift.
    '''
    if np.any(sensitivity < 0.0):
        raise ValueError(
            'the optimality-criteria update needs a nonnegative sensitivity (adding '
            'material must not raise the objective), which holds for compliance-type '
            'objectives. A signed objective such as a raw point displacement can go '
            'either way and is not compatible with this update; minimize a compliance '
            'or a squared quantity instead, or use a general gradient optimizer.'
        )
    lo, hi = 0.0, 1e15
    rho_new = rho
    for _ in range(max_iters):
        m = 0.5 * (lo + hi)
        rho_new = rho * np.sqrt(sensitivity / m)
        rho_new = np.clip(rho_new, rho - move, rho + move)
        rho_new = np.clip(rho_new, 1e-6, 1.0)
        if float((volumes * rho_new).sum() / volumes.sum()) < volume_frac:
            hi = m
        else:
            lo = m
        if hi - lo <= tol * hi:
            break
    return rho_new


@dataclass
class SIMPModel:
    '''A SIMP density design of a linear-elastic problem: geometry, material, supports.

    Bundles what a design iteration needs from a density field: the diluted `Problem` to
    solve (`E(rho) = rho^p E0`, the stiffness rescaled from a cached solid set) and the
    `DensityField` parameterization that differentiates it. The solid element stiffness
    and the constraints and load are built once; a step rescales the cached matrices
    rather than reassembling.

    An optional `sensitivity_filter` is the SIMP cone filter (`calculate_smoothing_matrix`):
    applied to the raw sensitivity in the optimizer, not here, since filtering is a
    property of density design rather than of the adjoint gradient.
    '''
    space: FunctionSpace
    base_E: float
    nu: float
    bc: BoundaryConditions
    source: FieldValue = None
    penalty: float = 3.0
    sensitivity_filter: SparseMatrix | None = None
    _density: DensityField = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._density = DensityField.create(
            self.space, np.ones(len(self.space.element_nodes)), self.base_E, self.nu, self.penalty,
        )

    @property
    def volumes(self) -> FloatArray:
        return self.space.element_volumes

    def parameterization(self, rho: ElementField) -> DensityField:
        return self._density.with_density(rho)

    def problem(self, rho: ElementField) -> LinearProblem:
        '''The elastic problem at density `rho`, its stiffness rescaled by `rho^p`.'''
        dilution = np.asarray(rho, dtype=float) ** self.penalty
        stiffness = PrecomputedForm(dilution[:, None, None] * self._density._K0)
        return LinearProblem(self.space, stiffness, self.source, self.bc)


@dataclass(frozen=True, eq=False)
class DesignHistory:
    '''The per-iteration series a design optimization produces.'''
    rho: list[ElementField]
    u: list[DofVector]
    objective: list[float]


class DesignOptimizer:
    '''Minimize a `QuantityOfInterest` over a SIMP density under a volume constraint.

    Each iteration solves the elastic problem at the current density, scores the
    objective, gets `dJ/drho` from the adjoint core (`SensitivityAnalysis`), filters the
    sensitivity if the model carries a filter, and takes an OC step. The gradient of a
    self-adjoint objective (compliance) costs no extra solve; a general one (a point
    deflection) costs a single adjoint solve reusing the forward factorization.
    '''

    def __init__(
        self,
        model: SIMPModel,
        objective: QuantityOfInterest | None = None,
        volume_frac: float = 0.5,
        iters: int = 30,
        move: float = 0.2,
    ) -> None:
        self.model = model
        self.objective: QuantityOfInterest = objective if objective is not None else Compliance()
        self.volume_frac = volume_frac
        self.iters = iters
        self.move = move
        self.rho: ElementField = np.full(len(model.space.element_nodes), volume_frac)
        self.history: DesignHistory | None = None

    def step(self) -> tuple[DofVector, float]:
        '''One iteration: solve, score, and advance the density. Returns `(u, J)`.'''
        problem = self.model.problem(self.rho)
        analysis = SensitivityAnalysis(problem)
        u = analysis.solve_forward()
        objective_value = self.objective.value(problem, u)

        parameterization = self.model.parameterization(self.rho)
        gradient = analysis.gradient(self.objective, parameterization, u)
        # OC wants the ascent sensitivity: positive where adding material lowers J.
        sensitivity = -gradient
        if self.model.sensitivity_filter is not None:
            sensitivity = self.model.sensitivity_filter @ sensitivity

        self.rho = optimality_criteria_update(
            self.rho, sensitivity, self.model.volumes, self.volume_frac, self.move,
        )
        return u, objective_value

    def solve(self, on_iteration: Callable[[int, ElementField, float], None] | None = None) -> DesignHistory:
        rho_series: list[ElementField] = []
        u_series: list[DofVector] = []
        objective_series: list[float] = []
        for i in range(self.iters):
            rho_before = self.rho
            u, objective_value = self.step()
            rho_series.append(rho_before)
            u_series.append(u)
            objective_series.append(objective_value)
            logger.info('Design iteration %d: objective = %.6g, volume fraction = %.4f',
                        i, objective_value,
                        float((self.model.volumes * rho_before).sum() / self.model.volumes.sum()))
            if on_iteration is not None:
                on_iteration(i, rho_before, objective_value)
        self.history = DesignHistory(rho_series, u_series, objective_series)
        return self.history
