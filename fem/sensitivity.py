"""Adjoint sensitivity: the gradient of a scalar output with respect to many parameters.

Given a solved linear problem `K u = f`, a scalar quantity of interest `J(u)`, and a
parameter field `p` the operator depends on, this computes `dJ/dp` for every parameter
at the cost of one extra solve, not one solve per parameter. The derivation lives in
`attic/adjoint-method-derivation-2026-08-18.md`; the three moving parts are:

    adjoint solve:   Kᵀ λ = (∂J/∂u)ᵀ                      (one solve, reuses K's factorization)
    gradient:        dJ/dp = ∂J/∂p − λᵀ (∂K/∂p) u          (a cheap product per parameter)

The design splits cleanly in three, none of them per-PDE:

- `QuantityOfInterest` owns `∂J/∂u` (the adjoint load) and can score a state.
- `Parameterization` owns `∂K/∂p` and turns the pair `(u, λ)` into the gradient.
- `SensitivityAnalysis` owns one factored `DiscreteSystem`, so the forward and adjoint
  solves share the factorization. For a symmetric `K` (Poisson, small-strain elasticity)
  the adjoint reuses it directly; `Kᵀ = K`.

`TopologyOptimizer`'s compliance sensitivity is the self-adjoint special case of this:
`J = fᵀu` gives the adjoint load `f`, so `λ = u` and no second solve is needed. That
case is the correctness anchor (`tests/test_sensitivity.py` checks the general path
reproduces it). Only linear, symmetric problems are handled here; the nonlinear tangent
path is future work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fem.backends import Backend
from fem.forms import LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.problem import Problem
from fem.space import FunctionSpace
from fem.system import DiscreteSystem
from fem.typing import DofVector, ElementField, FloatArray


def _element_dof_vectors(space: FunctionSpace, vector: DofVector) -> FloatArray:
    '''Gather a global DOF vector into per-element blocks `(n_elements, N*n_components)`.

    The interleaved node-major order the element matrices are written in: the same
    reshape `ElasticSolution.from_solve` uses, so the local vector lines up with the
    columns of a `LinearElasticForm` element matrix.
    '''
    per_node = np.asarray(vector, dtype=float).reshape(-1, space.n_components)
    return per_node[space.element_nodes].reshape(len(space.element_nodes), -1)


# -- quantities of interest ----------------------------------------------------

class QuantityOfInterest(Protocol):
    '''A scalar output `J(u)` and its derivative `∂J/∂u`, the adjoint load.

    `self_adjoint` marks the case where the adjoint load equals the forward load, so
    `λ = u` and the adjoint solve is skipped (compliance, under homogeneous Dirichlet
    supports). Left `False` by default; a QoI sets it only when the identity holds.
    '''

    @property
    def self_adjoint(self) -> bool: ...

    def value(self, problem: Problem, u: DofVector) -> float: ...

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        '''The adjoint load `(∂J/∂u)ᵀ`, a full DOF vector.'''
        ...


@dataclass(frozen=True)
class Compliance:
    '''Total compliance `J = fᵀu`: the structure's strain energy under its load.

    Self-adjoint: `∂J/∂u = f` is the forward load, so `λ = u` and no adjoint solve is
    needed. Valid where the supports are homogeneous Dirichlet (zero prescribed
    displacement), which is the topology-optimization setting and the one
    `SensitivityAnalysis` guards for before taking the shortcut.
    '''
    self_adjoint: bool = True

    def value(self, problem: Problem, u: DofVector) -> float:
        return float(problem.load @ u)

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        return problem.load


@dataclass(frozen=True)
class PointValue:
    '''A single DOF of the solution, `J = u[dof]`: a point displacement or potential.

    Not self-adjoint: the adjoint load is a unit vector at `dof` (a dummy unit load
    there), so the adjoint solve reads how sensitive that one DOF is to the design
    everywhere, the classic unit-load method.
    '''
    dof: int
    self_adjoint: bool = False

    def value(self, problem: Problem, u: DofVector) -> float:
        return float(u[self.dof])

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        e = np.zeros_like(np.asarray(u, dtype=float))
        e[self.dof] = 1.0
        return e


# -- parameterizations ---------------------------------------------------------

class Parameterization(Protocol):
    '''What the operator depends on, and how `∂K/∂p` turns `(u, λ)` into the gradient.'''

    @property
    def size(self) -> int: ...

    def gradient(self, problem: Problem, u: DofVector, lam: DofVector) -> FloatArray:
        '''The per-parameter gradient contribution `−λᵀ (∂K/∂p) u` (with `∂f/∂p = 0`).'''
        ...


@dataclass(frozen=True)
class _ElementModulusParameterization:
    '''Shared core for the two per-element modulus parameterizations.

    Both scale a per-element solid stiffness `K0_e` by a factor that depends on the
    design, so `∂K/∂p_e = c_e K0_e` is element-local and the gradient contribution is
    `−c_e (λ_eᵀ K0_e u_e)`. Subclasses supply `K0` (the reference element stiffness) and
    the factor `c_e`.
    '''
    space: FunctionSpace
    nu: float
    _K0: FloatArray = field(repr=False)

    def _element_quadratic(self, u: DofVector, lam: DofVector) -> ElementField:
        '''The element bilinear form `λ_eᵀ K0_e u_e`, one scalar per element.'''
        u_el = _element_dof_vectors(self.space, u)
        lam_el = _element_dof_vectors(self.space, lam)
        return np.einsum('ei,eij,ej->e', lam_el, self._K0, u_el)


@dataclass(frozen=True)
class DensityField(_ElementModulusParameterization):
    '''SIMP density: `E(ρ) = ρ^p E0`, so the stiffness scales by `ρ^p` per element.

    `∂K/∂ρ_e = p ρ_e^{p-1} K0_e` with `K0_e` the solid-material element stiffness, so the
    gradient contribution is `−p ρ_e^{p-1} (λ_eᵀ K0_e u_e)`. Build one with `create` and
    advance the density each optimizer step with `with_density`, which shares the cached
    `K0` rather than reassembling it.

    No sensitivity filter here: filtering the raw sensitivity is a property of density
    topology optimization, not of the adjoint, so it lives in the optimization loop.
    '''
    rho: ElementField = field(default_factory=lambda: np.zeros(0))
    penalty: float = 3.0

    @classmethod
    def create(
        cls, space: FunctionSpace, rho: ElementField, base_E: float, nu: float,
        penalty: float = 3.0,
    ) -> 'DensityField':
        K0 = LinearElasticForm(LinearElasticMaterial(base_E, nu)).element_matrices(space.geometry)
        return cls(space=space, nu=nu, _K0=K0, rho=np.asarray(rho, dtype=float), penalty=penalty)

    def with_density(self, rho: ElementField) -> 'DensityField':
        '''The same parameterization at a new density, sharing the cached `K0`.'''
        return DensityField(
            space=self.space, nu=self.nu, _K0=self._K0,
            rho=np.asarray(rho, dtype=float), penalty=self.penalty,
        )

    @property
    def size(self) -> int:
        return len(self.rho)

    def gradient(self, problem: Problem, u: DofVector, lam: DofVector) -> FloatArray:
        quad = self._element_quadratic(u, lam)
        return -self.penalty * self.rho ** (self.penalty - 1) * quad


@dataclass(frozen=True)
class ModulusField(_ElementModulusParameterization):
    '''A per-element Young's modulus, differentiated directly (no SIMP penalty).

    `K` is linear in `E`, so `∂K/∂E_e = K0_e` at unit modulus and the gradient is
    `−(λ_eᵀ K0_e u_e)`. The parameterization for inverse problems that recover a modulus
    field from measured response.
    '''
    E: ElementField = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def create(cls, space: FunctionSpace, E: ElementField, nu: float) -> 'ModulusField':
        # Unit-modulus solid stiffness: K_e(E_e) = E_e * K0_e, so ∂K/∂E_e = K0_e.
        K0 = LinearElasticForm(LinearElasticMaterial(1.0, nu)).element_matrices(space.geometry)
        return cls(space=space, nu=nu, _K0=K0, E=np.asarray(E, dtype=float))

    def with_modulus(self, E: ElementField) -> 'ModulusField':
        return ModulusField(space=self.space, nu=self.nu, _K0=self._K0, E=np.asarray(E, dtype=float))

    @property
    def size(self) -> int:
        return len(self.E)

    def gradient(self, problem: Problem, u: DofVector, lam: DofVector) -> FloatArray:
        return -self._element_quadratic(u, lam)


# -- the driver ----------------------------------------------------------------

class SensitivityAnalysis:
    '''Forward solve, adjoint solve, and gradient, sharing one factorization.

    Owns a `DiscreteSystem` built from the problem's constant tangent, so the forward
    solve and every adjoint solve reuse the same factored free block. For a symmetric
    operator (the linear problems here) that factorization serves the adjoint directly.

    The adjoint carries homogeneous Dirichlet data (`λ` is zero at the fixed DOFs),
    which is what `DiscreteSystem.solve_homogeneous` supplies while reusing the
    factorization. A self-adjoint QoI under homogeneous supports skips the adjoint solve
    entirely and takes `λ = u`.
    '''

    def __init__(self, problem: Problem, backend: Backend | None = None) -> None:
        self.problem = problem
        _, _, fixed_values = problem.constraints
        # The λ = u shortcut is exact only when the forward supports are homogeneous;
        # a prescribed nonzero displacement breaks the self-adjoint identity.
        self._homogeneous_supports = bool(np.allclose(fixed_values, 0.0))
        self._system = DiscreteSystem(problem.tangent(None), problem.constraints, backend)

    def solve_forward(self) -> DofVector:
        '''The forward solution `u` of `K u = f`.'''
        return self._system.solve(self.problem.load)

    def adjoint(self, qoi: QuantityOfInterest, u: DofVector) -> DofVector:
        '''The adjoint field `λ` solving `Kᵀ λ = (∂J/∂u)ᵀ` with homogeneous supports.'''
        if qoi.self_adjoint and self._homogeneous_supports:
            return u
        return self._system.solve_homogeneous(qoi.dJ_du(self.problem, u))

    def gradient(
        self, qoi: QuantityOfInterest, parameterization: Parameterization, u: DofVector,
    ) -> FloatArray:
        '''`dJ/dp` for every parameter: one adjoint solve, then a product per parameter.'''
        lam = self.adjoint(qoi, u)
        return parameterization.gradient(self.problem, u, lam)
