"""Density (SIMP) design optimization over the adjoint core.

`SIMPModel` is the specification: a small-strain elastic `LinearProblem` whose
stiffness a density field dilutes as `E(rho) = rho^p E0`. `DesignOptimizer` is
the driver: each iteration solves the diluted problem, scores a `QuantityOfInterest`,
takes its gradient through `fem.sensitivity`, filters it, and moves the density by the
optimality-criteria (OC) update under a volume constraint.

Scope: a single volume constraint, the setting where OC applies. A general constrained
solver over `scipy.optimize` is noted in `attic/fem-adjoint-sensitivity-design-2026-08-18.md`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import KDTree

from fem.forms import LinearElasticForm, PrecomputedForm
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.problem import LinearProblem, Problem
from fem.sensitivity import (
    Compliance,
    DensityField,
    QuantityOfInterest,
    SensitivityAnalysis,
)
from fem.solution import ElasticSolution
from fem.space import FunctionSpace
from fem.typing import DofVector, ElementField, FloatArray, SparseMatrix

logger = logging.getLogger(__name__)


def calculate_smoothing_matrix(mesh: Mesh, r: float) -> SparseMatrix:
    '''Row-normalized cone weights over the element centers within radius `r`.

    The SIMP sensitivity filter: an element's smoothed sensitivity is a weighted
    mean of the sensitivities within `r` of it, under the weight `r - distance`
    falling linearly to zero at the radius. Filtering keeps the optimizer off
    checkerboard designs, and `r` sets the design's feature size.

    Sparse, off a KD-tree neighbour query, so the filter costs O(n_elements) when `r`
    tracks the element size. Rows sum to 1, except at `r = 0`, where every weight is
    zero and the row is too.
    '''
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    n_elements = len(centers)

    # Distinct pairs (i < j) within the radius, mirrored below; the diagonal is the
    # self-pair at distance zero, weight r.
    pairs = KDTree(centers).query_pairs(r, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    off_diagonal = r - np.linalg.norm(centers[i] - centers[j], axis=1)

    diagonal = np.arange(n_elements)
    rows = np.concatenate([i, j, diagonal])
    cols = np.concatenate([j, i, diagonal])
    weights = np.concatenate([off_diagonal, off_diagonal, np.full(n_elements, float(r))])

    # The 1e-16 keeps a weightless row (only at r = 0) at zero.
    row_sums = np.bincount(rows, weights=weights, minlength=n_elements)
    return csr_array(
        (weights / (row_sums[rows] + 1e-16), (rows, cols)),
        shape=(n_elements, n_elements),
    )


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
    `volume_frac`.
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


@dataclass(frozen=True)
class TargetCompliance:
    '''Drive the total compliance toward `target`: `J = (C - target)^2`.

    `dJ/du = 2 (C - target) f`, so the OC update accepts it only while `C > target`
    (a stiffer-than-target design has a signed sensitivity).
    '''
    target: float
    self_adjoint: bool = False

    def value(self, problem: Problem, u: DofVector) -> float:
        return (Compliance().value(problem, u) - self.target) ** 2

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        residual = Compliance().value(problem, u) - self.target
        return 2.0 * residual * Compliance().dJ_du(problem, u)


@dataclass
class SIMPModel:
    '''A SIMP density design over a small-strain elastic `LinearProblem`.

    `template` supplies the solid material (a `LinearElasticForm` with a scalar
    modulus), the supports, and the load, shared by every density. `problem(rho)` is the
    elastic problem at density `rho`, its stiffness rescaled by `rho^p` from one cached
    set of solid element matrices; `parameterization(rho)` is the `DensityField` that
    differentiates it.

    `sensitivity_filter` is the SIMP cone filter (`calculate_smoothing_matrix`), applied
    to the raw sensitivity by the optimizer.
    '''
    template: LinearProblem
    penalty: float = 3.0
    sensitivity_filter: SparseMatrix | None = None
    _material: LinearElasticMaterial = field(init=False, repr=False)
    _density: DensityField = field(init=False, repr=False)

    def __post_init__(self) -> None:
        operator = self.template.operator
        if not isinstance(operator, LinearElasticForm):
            raise ValueError(
                'SIMP rescales a small-strain elastic stiffness; the operator is '
                f'{type(operator).__name__}'
            )
        if not isinstance(operator.material.E, int | float):
            raise ValueError('SIMP scales one solid modulus; E must be a scalar')
        self._material = operator.material
        solid = operator.element_matrices(self.space.geometry)
        self._density = DensityField(
            space=self.space, nu=self._material.nu, _K0=solid,
            rho=np.ones(len(self.space.element_nodes)), penalty=self.penalty,
        )

    @property
    def space(self) -> FunctionSpace:
        return self.template.space

    @property
    def volumes(self) -> FloatArray:
        return self.space.element_volumes

    def scaled_modulus(self, rho: ElementField) -> ElementField:
        '''The SIMP-scaled modulus `E(rho) = rho^p E0`, one value per element.'''
        return np.asarray(rho, dtype=float) ** self.penalty * self._material.E

    def parameterization(self, rho: ElementField) -> DensityField:
        return self._density.with_density(rho)

    def problem(self, rho: ElementField) -> LinearProblem:
        '''The elastic problem at density `rho`, its stiffness rescaled by `rho^p`.'''
        dilution = np.asarray(rho, dtype=float) ** self.penalty
        stiffness = PrecomputedForm(dilution[:, None, None] * self._density._K0)
        return self.template.with_operator(stiffness)

    def solution(self, rho: ElementField, u: DofVector) -> ElasticSolution:
        '''The displacement `u` at density `rho` with the stress of the diluted material.'''
        # Stress wants the diluted material itself: sigma = D(E(rho)) eps.
        form = LinearElasticForm(LinearElasticMaterial(self.scaled_modulus(rho), self._material.nu))
        return ElasticSolution.from_solve(self.space, u, form)


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
    `solution` is the most recent iterate's `ElasticSolution`.
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
        self.solution: ElasticSolution | None = None
        self.history: DesignHistory | None = None

    def step(self) -> tuple[DofVector, float]:
        '''One iteration: solve, score, and advance the density. Returns `(u, J)`.'''
        problem = self.model.problem(self.rho)
        analysis = SensitivityAnalysis(problem)
        u = analysis.solve_forward()
        objective_value = self.objective.value(problem, u)
        self.solution = self.model.solution(self.rho, u)

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

    def run(self, on_iteration: Callable[[int, ElementField, float], None] | None = None) -> DesignHistory:
        '''Run every iteration and return the history; `on_iteration(i, rho, J)` is
        called after each.'''
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
