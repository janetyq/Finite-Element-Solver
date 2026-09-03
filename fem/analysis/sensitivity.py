"""Adjoint sensitivity: the gradient of a scalar output with respect to many parameters.

Given a solved linear problem `K u = f`, a scalar quantity of interest `J(u)`, and a
parameter field `p` the operator depends on, this computes `dJ/dp` for every parameter
at the cost of one extra solve, not one solve per parameter. The derivation lives in
`attic/adjoint-method-derivation-2026-08-18.md`; the three moving parts are:

    adjoint solve:   Kᵀ λ = (∂J/∂u)ᵀ                      (one solve, reuses K's factorization)
    gradient:        dJ/dp = ∂J/∂p − λᵀ (∂K/∂p) u          (a cheap product per parameter)

Three parts, none per-PDE:

- `QuantityOfInterest` owns `∂J/∂u` (the adjoint load) and can score a state.
- `Parameterization` owns `∂K/∂p` and turns the pair `(u, λ)` into the gradient.
- `SensitivityAnalysis` owns one factored `DiscreteSystem`, so the forward and adjoint
  solves share the factorization. For a symmetric `K` (Poisson, small-strain elasticity)
  the adjoint reuses it directly; `Kᵀ = K`.

Compliance in SIMP design is the self-adjoint special case:
`J = fᵀu` gives the adjoint load `f`, so `λ = u` and no second solve is needed. Only
linear, symmetric problems are handled here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fem.algebra.backends import Backend
from fem.field import NodalField
from fem.physics.forms import LinearElasticForm
from fem.physics.materials import LinearElasticMaterial
from fem.problem import Problem
from fem.space import FunctionSpace
from fem.algebra.solve import backend_for
from fem.algebra.system import DiscreteSystem
from fem.typing import BoolArray, DofVector, ElementValues, FloatArray, IntArray


def _element_dof_vectors(space: FunctionSpace, vector: DofVector) -> FloatArray:
    '''Gather a global DOF vector into per-element blocks `(n_elements, N*n_components)`.

    The interleaved node-major order the element matrices are written in: the same
    reshape `ElasticSolution.from_solve` uses, so the local vector lines up with the
    columns of a `LinearElasticForm` element matrix.
    '''
    return NodalField(space, vector).element_values.reshape(len(space.element_nodes), -1)


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
    '''Total compliance `J = fᵀu`, twice the strain energy when the load is the
    conditions' alone; with an operator load such as a thermal one, `f` includes it.

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
    '''The field at a point, `J = u_h(x)`: a point displacement or potential.

    `component` picks the component of a vector field. `J` is linear in the DOFs,
    `J = phi(x) . u_e` over the element holding `x`, so `dJ/du` is the shape functions
    at `x` scattered to that element's DOFs: a dummy unit load there, the classic
    unit-load method. At a node it is the unit vector at that node's DOF. Not
    self-adjoint.
    '''
    point: FloatArray
    component: int = 0
    self_adjoint: bool = False

    def value(self, problem: Problem, u: DofVector) -> float:
        self._check_component(problem.space)
        values = NodalField(problem.space, u).evaluate(self.point)   # (1,) or (1, n_components)
        return float(values.reshape(1, -1)[0, self.component])

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        return self._weights(problem.space)

    def _check_component(self, space: FunctionSpace) -> None:
        if not 0 <= self.component < space.n_components:
            raise IndexError(
                f'component {self.component} of a field with {space.n_components} components')

    def _weights(self, space: FunctionSpace) -> DofVector:
        '''`dJ/du`: the shape functions at `point`, on the DOFs of the element holding it.'''
        self._check_component(space)
        (element,), reference = space.mesh.locate(np.atleast_2d(np.asarray(self.point, dtype=float)))
        phi = space.element_type.shape_values(reference)[0]           # (N,)
        weights = np.zeros(space.n_dofs)
        dofs = space.dof_indices(space.element_nodes[element])        # node-major
        weights[dofs[self.component::space.n_components]] = phi
        return weights


# -- stress-based quantities of interest ---------------------------------------
#
# The adjoint load for a stress functional is the stress-recovery map run backward.
# The forward map is `s = D B u_e` per element (in-plane Voigt stress from element
# displacements); a stress measure `m(s)` then aggregates over elements. The adjoint
# load is `∂J/∂u = Σ_e (∂agg/∂m_e)(∂m_e/∂s_e)(D_e B_e)`, scattered to the global DOFs.
# Only the plane-strain 2D case is handled, the setting the elastic demos use.
#
# Scope: these supply `∂J/∂u` (the adjoint load) for a fixed material. Used with
# `SensitivityAnalysis` over a parameter the stress does not explicitly depend on (an
# applied load, or a design whose stress is read off a fixed reference material), the
# gradient is complete. Over a parameter the stress does depend on directly (the same
# modulus the measure evaluates the stress with), there is an additional explicit term
# `∂J/∂p` the adjoint pass does not add; that coupling, the basis of stress-constrained
# design, is a documented follow-up (see BACKLOG.md).


def _element_dof_indices(space: FunctionSpace) -> IntArray:
    '''(n_elements, N*n_components) global DOF index of each element's local DOFs.

    The interleaved node-major order `_element_dof_vectors` gathers in, so a per-element
    vector in that order scatters back through these indices.
    '''
    nc = space.n_components
    nodes = np.asarray(space.element_nodes)
    return (nodes[:, :, None] * nc + np.arange(nc)).reshape(len(nodes), -1)


def _require_plane_strain(space: FunctionSpace) -> None:
    '''Guard the stress quantities of interest, which are plane-strain 2D only.

    The von Mises here lives in three Voigt components with `sigma_zz = nu(sxx + syy)`;
    a 3D space carries six, which `_Q` and its `dQ/ds` chain do not handle, so the measure
    would return a silently wrong gradient. Refuse instead. A 3D measure is future work.
    '''
    if space.spatial_dim != 2:
        raise NotImplementedError(
            f'Stress quantities of interest are plane-strain 2D only; the space is '
            f'{space.spatial_dim}D. A 3D measure needs the full six-component Voigt path.'
        )


def _require_no_operator_load(problem: Problem) -> None:
    '''Guard the stress quantities of interest against an eigenstrain.

    They measure `D B u`, the stress of the displacement alone, and a problem whose
    operator carries its own load (a thermal strain) has a stress `D (eps - eps*)`
    they do not see. Refuse rather than report the wrong one.
    '''
    if problem.operator_load is not None:
        raise NotImplementedError(
            'the stress quantities of interest measure the stress of the displacement '
            'alone and take no eigenstrain yet; the problem\'s operator carries a load'
        )


@dataclass(frozen=True)
class _VonMisesStress:
    '''Per-element von Mises stress and its derivative with respect to `u`, plane strain.

    Shared machinery for the stress quantities of interest. `region` optionally restricts
    attention to a subset of elements (a mask over all elements); the aggregation weights
    are volume-weighted over whatever the region selects.

    Plane strain: `sigma_zz = nu (sigma_xx + sigma_yy)`, so von Mises is a function of the
    three in-plane Voigt components alone, and its gradient carries the `sigma_zz`
    dependence through the chain rule. `evaluate` and `derivative` are consistent by
    construction (both from `Q = von_mises^2`), which the finite-difference test pins.
    '''
    space: FunctionSpace
    material: LinearElasticMaterial
    region: BoolArray | None = None

    def __post_init__(self) -> None:
        _require_plane_strain(self.space)

    def _DB(self) -> FloatArray:
        '''(n_elements, 3, N*nc): the map `u_e -> in-plane Voigt stress`, D_e B_e.

        B is the volume-weighted mean over the element's rule, so the stress this
        measures is the element mean (the centroid value on a straight P2 element),
        the same one `ElasticSolution.stress` reports.
        '''
        from fem.physics.forms import strain_displacement
        geometry = self.space.geometry
        weights = geometry.weight_detJ / geometry.weight_detJ.sum(axis=1, keepdims=True)
        B = np.einsum('eq,eqsk->esk', weights, strain_displacement(geometry.grad_phi))
        D = self.material.constitutive_matrices(geometry.reference_dim, geometry.n_elements)
        return np.einsum('est,etk->esk', D, B)

    def _voigt_stress(self, u: DofVector) -> FloatArray:
        u_el = _element_dof_vectors(self.space, u)                       # (n_el, N*nc)
        return np.einsum('esk,ek->es', self._DB(), u_el)                 # (n_el, 3): xx, yy, xy

    def von_mises(self, u: DofVector) -> ElementValues:
        s = self._voigt_stress(u)
        return np.sqrt(np.maximum(self._Q(s), 0.0))

    def _Q(self, s: FloatArray) -> ElementValues:
        '''von Mises squared from the in-plane Voigt stress, plane strain.'''
        sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
        szz = self.material.nu * (sxx + syy)
        # vm^2 = 1/2[(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2] + 3 sxy^2, expanded.
        return sxx**2 + syy**2 + szz**2 - sxx * syy - syy * szz - szz * sxx + 3.0 * sxy**2

    def _dvm_du(self, u: DofVector) -> tuple[ElementValues, FloatArray]:
        '''Per-element `(von_mises, d(von_mises)/d u_e)`; the latter is `(n_el, N*nc)`.'''
        DB = self._DB()
        s = np.einsum('esk,ek->es', DB, _element_dof_vectors(self.space, u))
        sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
        nu = self.material.nu
        szz = nu * (sxx + syy)
        vm = np.sqrt(np.maximum(self._Q(s), 0.0))

        # dQ/ds, total derivative through sigma_zz = nu(sxx + syy): the first terms are
        # dQ/ds at fixed szz, the nu(...) terms add (dQ/dszz)(dszz/ds) = nu * dQ/dszz.
        dQ = np.empty_like(s)
        dQ[:, 0] = 2 * sxx - syy - szz + nu * (2 * szz - syy - sxx)
        dQ[:, 1] = 2 * syy - sxx - szz + nu * (2 * szz - sxx - syy)
        dQ[:, 2] = 6.0 * sxy
        # dvm/ds = dQ/ds / (2 vm), since vm = sqrt(Q); at vm = 0, dQ/ds is also 0, so 0 there.
        safe = vm > 0
        dvm_ds = np.zeros_like(s)
        dvm_ds[safe] = dQ[safe] / (2.0 * vm[safe, None])
        dvm_du = np.einsum('esk,es->ek', DB, dvm_ds)                     # (n_el, N*nc)
        return vm, dvm_du

    def _weights(self) -> FloatArray:
        '''Volume weights over the region, summing to 1 (a volume-weighted mean).'''
        volumes = self.space.element_volumes
        mask = np.ones(len(volumes), dtype=bool) if self.region is None else np.asarray(self.region)
        w = np.where(mask, volumes, 0.0)
        return w / w.sum()

    def _scatter(self, per_element_load: FloatArray, n_dofs: int) -> DofVector:
        '''Scatter per-element DOF loads `(n_el, N*nc)` into a global `(n_dofs,)` vector.'''
        dofs = _element_dof_indices(self.space)
        load = np.zeros(n_dofs)
        np.add.at(load, dofs.ravel(), per_element_load.ravel())
        return load


@dataclass(frozen=True)
class MeanStress:
    '''Volume-weighted mean von Mises stress over a region (all elements by default).

    A smooth, differentiable stress measure. Not self-adjoint: the adjoint load is the
    stress-recovery map run backward, so the adjoint solve reads how each design change
    moves the mean stress.
    '''
    space: FunctionSpace
    material: LinearElasticMaterial
    region: BoolArray | None = None
    self_adjoint: bool = False

    def __post_init__(self) -> None:
        _require_plane_strain(self.space)

    def _stress(self) -> _VonMisesStress:
        return _VonMisesStress(self.space, self.material, self.region)

    def value(self, problem: Problem, u: DofVector) -> float:
        _require_no_operator_load(problem)
        stress = self._stress()
        return float(stress._weights() @ stress.von_mises(u))

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        _require_no_operator_load(problem)
        stress = self._stress()
        _, dvm_du = stress._dvm_du(u)
        per_element = stress._weights()[:, None] * dvm_du
        return stress._scatter(per_element, len(np.asarray(u)))


@dataclass(frozen=True)
class SoftMaxStress:
    '''A smooth approximation of the peak von Mises stress: the volume-weighted p-norm.

    `J = (Σ_e w_e vm_e^p)^{1/p}` with `w_e` the volume weights over the region. As `p`
    grows this approaches the maximum, but stays differentiable, so it is the usual stand-
    in for a peak-stress constraint (a true max is not differentiable). `p = 8` is a
    common default: large enough to track the peak, small enough to stay well conditioned.
    '''
    space: FunctionSpace
    material: LinearElasticMaterial
    region: BoolArray | None = None
    p: float = 8.0
    self_adjoint: bool = False

    def __post_init__(self) -> None:
        _require_plane_strain(self.space)

    def _stress(self) -> _VonMisesStress:
        return _VonMisesStress(self.space, self.material, self.region)

    def value(self, problem: Problem, u: DofVector) -> float:
        _require_no_operator_load(problem)
        stress = self._stress()
        w = stress._weights()
        vm = stress.von_mises(u)
        return float((w @ vm**self.p) ** (1.0 / self.p))

    def dJ_du(self, problem: Problem, u: DofVector) -> DofVector:
        _require_no_operator_load(problem)
        stress = self._stress()
        w = stress._weights()
        vm, dvm_du = stress._dvm_du(u)
        total = float(w @ vm**self.p)
        if total <= 0.0:
            return np.zeros(len(np.asarray(u)))
        # dJ/dvm_e = J^{1-p} w_e vm_e^{p-1}, then chain through dvm_e/du.
        dJ_dvm = total ** (1.0 / self.p - 1.0) * w * vm ** (self.p - 1.0)
        per_element = dJ_dvm[:, None] * dvm_du
        return stress._scatter(per_element, len(np.asarray(u)))


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

    def _element_quadratic(self, u: DofVector, lam: DofVector) -> ElementValues:
        '''The element bilinear form `λ_eᵀ K0_e u_e`, one scalar per element.'''
        u_el = _element_dof_vectors(self.space, u)
        lam_el = _element_dof_vectors(self.space, lam)
        return np.einsum('ei,eij,ej->e', lam_el, self._K0, u_el)


@dataclass(frozen=True)
class DensityParameterization(_ElementModulusParameterization):
    '''SIMP density: `E(ρ) = ρ^p E0`, so the stiffness scales by `ρ^p` per element.

    `∂K/∂ρ_e = p ρ_e^{p-1} K0_e` with `K0_e` the solid-material element stiffness, so the
    gradient contribution is `−p ρ_e^{p-1} (λ_eᵀ K0_e u_e)`. Build one with `create` and
    advance the density each optimizer step with `with_density`, which shares the cached
    `K0` rather than reassembling it.

    No sensitivity filter here: filtering the raw sensitivity is a property of density
    topology optimization, not of the adjoint, so it lives in the optimization loop.
    '''
    rho: ElementValues = field(default_factory=lambda: np.zeros(0))
    penalty: float = 3.0

    @classmethod
    def create(
        cls, space: FunctionSpace, rho: ElementValues, base_E: float, nu: float,
        penalty: float = 3.0,
    ) -> 'DensityParameterization':
        K0 = LinearElasticForm(LinearElasticMaterial(base_E, nu)).element_matrices(space.geometry)
        return cls(space=space, nu=nu, _K0=K0, rho=np.asarray(rho, dtype=float), penalty=penalty)

    def with_density(self, rho: ElementValues) -> 'DensityParameterization':
        '''The same parameterization at a new density, sharing the cached `K0`.'''
        return DensityParameterization(
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
class ModulusParameterization(_ElementModulusParameterization):
    '''A per-element Young's modulus, differentiated directly (no SIMP penalty).

    `K` is linear in `E`, so `∂K/∂E_e = K0_e` at unit modulus and the gradient is
    `−(λ_eᵀ K0_e u_e)`. The parameterization for inverse problems that recover a modulus
    field from measured response.
    '''
    E: ElementValues = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def create(cls, space: FunctionSpace, E: ElementValues, nu: float) -> 'ModulusParameterization':
        # Unit-modulus solid stiffness: K_e(E_e) = E_e * K0_e, so ∂K/∂E_e = K0_e.
        K0 = LinearElasticForm(LinearElasticMaterial(1.0, nu)).element_matrices(space.geometry)
        return cls(space=space, nu=nu, _K0=K0, E=np.asarray(E, dtype=float))

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
    through `DiscreteSystem.solve_homogeneous` on the same factorization. A self-adjoint QoI under homogeneous supports skips the adjoint solve
    entirely and takes `λ = u`.
    '''

    def __init__(self, problem: Problem, backend: Backend | None = None) -> None:
        self.problem = problem
        _, _, fixed_values = problem.constraints
        # The λ = u shortcut is exact only when the forward supports are homogeneous;
        # a prescribed nonzero displacement breaks the self-adjoint identity.
        self._homogeneous_supports = bool(np.allclose(fixed_values, 0.0))
        self._system = DiscreteSystem(
            problem.tangent(None), problem.constraints, backend_for(problem, backend))

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
        '''`dJ/dp` for every parameter: one adjoint solve, then a product per parameter.

        The parameterizations assume the load does not depend on the parameters. An
        operator load does (a thermal load scales with the modulus), so a problem
        carrying one is refused rather than given a wrong gradient.
        '''
        if self.problem.operator_load is not None:
            raise NotImplementedError(
                'the parameterizations take the load as independent of the parameters, '
                'and the problem\'s operator carries a load of its own (an eigenstrain); '
                'its derivative is not implemented'
            )
        lam = self.adjoint(qoi, u)
        return parameterization.gradient(self.problem, u, lam)
