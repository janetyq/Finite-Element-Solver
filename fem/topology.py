"""Density-based (SIMP) topology optimization, as a driver over the solve core.

The optimizer owns a solve strategy and derives a fresh `Problem` each iteration
from the current density; it never mutates a shared one. The density scales the
modulus, `E(rho) = rho^p E_0`, and hence scales the element stiffness by the same
factor; the compliance and its sensitivity drive an optimality-criterion update of
the density. What the density does not touch (the constraints, the load, and the
element stiffness of the undiluted material) is built once and reused.

The objective is an injected object (`MinCompliance`, `TargetCompliance`) rather
than a string resolved through a `_select_*` dispatch, and the optimization method
(the OC update) is a method here. That replaces the plugin-shaped machinery (an
ignored args bag, an objective value that was never evaluated) with the one
quantity that is actually used: the sensitivity.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import KDTree

from fem.boundary import BoundaryConditions
from fem.design import optimality_criteria_update
from fem.forms import LinearElasticForm, PrecomputedForm
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.problem import LinearProblem
from fem.sensitivity import Compliance, DensityField, SensitivityAnalysis
from fem.solution import ElasticSolution
from fem.solve import LinearSolve, SolveStrategy
from fem.equations import LinearElastic
from fem.space import FunctionSpace
from fem.typing import DofVector, ElementField, FieldValue, FloatArray, SparseMatrix


def calculate_smoothing_matrix(mesh: Mesh, r: float) -> SparseMatrix:
    '''Row-normalized cone weights over the element centers within radius `r`.

    The SIMP sensitivity filter: an element's smoothed sensitivity is a weighted
    mean of the sensitivities within `r` of it, under the weight `r - distance`
    falling linearly to zero at the radius. Filtering the sensitivity keeps the
    optimizer off checkerboard designs, and `r` sets the design's feature size.

    Sparse, off a KD-tree neighbour query: an element couples only to the ones
    inside its radius, so only those pairs are stored. Under the usual choice of a
    radius tracking the element size, that is a bounded number of neighbours each
    and the filter costs O(n_elements); hold `r` fixed while refining and the
    neighbour count grows with the mesh, though the stored pairs stay a small
    fraction of all n^2 of them.

    Rows sum to 1, except at `r = 0`, where every weight is zero and the row is too.
    '''
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    n_elements = len(centers)

    # Distinct pairs (i < j) within the radius, so each coupling is found once and
    # mirrored below; the self-pairs query_pairs omits are the diagonal, at distance
    # zero and hence full weight r.
    pairs = KDTree(centers).query_pairs(r, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]
    off_diagonal = r - np.linalg.norm(centers[i] - centers[j], axis=1)

    diagonal = np.arange(n_elements)
    rows = np.concatenate([i, j, diagonal])
    cols = np.concatenate([j, i, diagonal])
    weights = np.concatenate([off_diagonal, off_diagonal, np.full(n_elements, float(r))])

    # The 1e-16 keeps a weightless row (only reachable at r = 0) at zero rather than
    # dividing by it.
    row_sums = np.bincount(rows, weights=weights, minlength=n_elements)
    return csr_array(
        (weights / (row_sums[rows] + 1e-16), (rows, cols)),
        shape=(n_elements, n_elements),
    )


class Objective(Protocol):
    '''Maps a compliance field and density to a per-element sensitivity.

    `chain` is the outer derivative `dJ/d(total compliance)`: the optimizer gets the
    per-element compliance sensitivity `dC/drho` from the adjoint core and multiplies it
    by this scalar to form the design sensitivity, so an objective that is a function of
    total compliance is stated by that one factor. `gradient` is the closed-form
    per-element sensitivity, kept as the self-adjoint reference the core reproduces.
    '''

    def gradient(self, compliance: ElementField, rho: ElementField, penalty: float) -> ElementField:
        ...

    def chain(self, total_compliance: float) -> float:
        ...


@dataclass(frozen=True)
class MinCompliance:
    '''Minimize total compliance: the sensitivity is d(compliance)/d(rho).'''

    def gradient(self, compliance: ElementField, rho: ElementField, penalty: float) -> ElementField:
        # For E(rho) = rho^p E_0 the element compliance is linear in E, so its
        # derivative is p/rho times it. The exponent must match the one set_rho
        # raised rho to, hence penalty in both.
        return compliance * penalty / rho

    def chain(self, total_compliance: float) -> float:
        # J = C, so dJ/dC = 1: the design sensitivity is the compliance sensitivity.
        return 1.0


@dataclass(frozen=True)
class TargetCompliance:
    '''Drive the total compliance toward `target` (a least-squares objective).'''
    target: float

    def gradient(self, compliance: ElementField, rho: ElementField, penalty: float) -> ElementField:
        residual = compliance.sum() - self.target
        return (compliance * penalty / rho) * 2 * residual

    def chain(self, total_compliance: float) -> float:
        # J = (C - target)^2, so dJ/dC = 2 (C - target).
        return 2.0 * (total_compliance - self.target)


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

        # The two iteration-invariant halves of the solve, built once here.
        #
        # `_solid_stiffness` is the element stiffness at the undiluted modulus E_0.
        # D is linear in E, so the density enters it as a plain factor: scaling E by
        # rho^p scales each element matrix by rho^p, and an iteration rescales these
        # rather than re-contracting B^T D B over the mesh.
        #
        # `_problem` carries the constraints and the load, which the density does not
        # reach; an iteration derives its own from this one with `with_operator`.
        self._solid_stiffness: FloatArray = LinearElasticForm(
            LinearElasticMaterial(self.base_E, self.nu)
        ).element_matrices(self.space.geometry)
        self._problem = LinearProblem(
            self.space, PrecomputedForm(self._solid_stiffness), self.source, self.bc,
        )
        # The adjoint-core parameterization, sharing the solid stiffness just built:
        # `_solve` computes each iteration's sensitivity through it rather than from a
        # hand-written formula. Only its density changes per iteration.
        self._density_param = DensityField(
            space=self.space, nu=self.nu, _K0=self._solid_stiffness,
            rho=self.rho, penalty=self.penalty,
        )

        self._last: ElasticSolution | None = None   # most recent single-iteration solve
        self._analysis: SensitivityAnalysis | None = None  # the factored system of _last
        self.history: TopologyHistory | None = None  # the per-iteration series

    @property
    def dilution(self) -> ElementField:
        '''The SIMP factor rho^p for the current density.

        Stated once because it applies twice over: to the modulus, and (since the
        element stiffness is linear in the modulus) to the element stiffness by the
        same factor. Two spellings of rho^p could drift into descending different
        gradients, exactly as two literal 3s once could.
        '''
        return self.rho**self.penalty

    @property
    def scaled_modulus(self) -> ElementField:
        '''The SIMP-scaled modulus E(rho) = rho^p * E_0 for the current density.'''
        return self.dilution * self.base_E

    def set_rho(self, rho: ElementField) -> None:
        self.rho = rho

    def _volume_fraction(self, rho: ElementField) -> float:
        '''Volume-weighted mean of a per-element field.'''
        volumes = self.space.element_volumes
        return float((volumes * rho).sum() / volumes.sum())

    def _solve(self) -> ElasticSolution:
        '''Solve the elastic problem at the current density and recover compliance.

        A fresh problem each call, derived from the iteration-invariant one: the
        density scales the modulus, so there is no shared state to mutate. The
        numerical path (LinearProblem -> LinearSolve -> derived_fields) is the same
        one the Solver facade runs, over the optimizer's cached space, constraints,
        and solid-material element stiffness.
        '''
        stiffness = PrecomputedForm(self.dilution[:, None, None] * self._solid_stiffness)
        # Solved through the adjoint core so the factored system is kept for the
        # sensitivity: compliance is self-adjoint, so the gradient reuses this forward
        # solution as its adjoint field with no second solve. `LinearSolve` and this both
        # eliminate the Dirichlet block through `DiscreteSystem`, so `u` is unchanged.
        self._analysis = SensitivityAnalysis(self._problem.with_operator(stiffness))
        u = self._analysis.solve_forward()

        # Stress recovery wants the diluted material itself, not the element matrices
        # it would produce: the stress in an element is D(E(rho)) times its strain.
        form = LinearElasticForm(LinearElasticMaterial(self.scaled_modulus, self.nu))
        self._last = ElasticSolution.from_solve(self.space, u, form)
        return self._last

    def _compliance_sensitivity(self, solution: ElasticSolution) -> ElementField:
        '''The per-element compliance sensitivity `dC/drho`, from the adjoint core.

        Computed through `SensitivityAnalysis` with a `Compliance` quantity of interest
        and the `DensityField` parameterization, replacing the hand-written `p/rho * c`
        formula with the general adjoint pass. Compliance is self-adjoint, so this reuses
        the forward solution as its adjoint field and adds no solve. The core returns the
        true gradient (negative, since stiffening lowers compliance); the optimizer wants
        the positive ascent sensitivity, so the sign is flipped here.
        '''
        assert self._analysis is not None  # set by the _solve that produced `solution`
        parameterization = self._density_param.with_density(self.rho)
        gradient = self._analysis.gradient(Compliance(), parameterization, solution.u)
        return -gradient

    def oc_density(
        self,
        sensitivity: ElementField,
        volume_frac: float,
        max_iters: int = 100,
        tol: float = 1e-8,
    ) -> ElementField:
        # sensitivity is the ascent sensitivity (positive where adding material lowers
        # the objective). The bisection on the Lagrange multiplier lives in
        # `fem.design.optimality_criteria_update`, shared with `DesignOptimizer`.
        return optimality_criteria_update(
            self.rho, sensitivity, self.space.element_volumes, volume_frac,
            move=0.1, max_iters=max_iters, tol=tol,
        )

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

            # Log and, if the caller wants it, hand off to their own visualization;
            # this class has no business knowing how (or whether) to plot itself.
            self._log_iteration(i, solution)
            if on_iteration is not None:
                on_iteration(i, solution)

            sensitivity = self._compliance_sensitivity(solution) * self.objective.chain(
                solution.compliance.sum())
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
