"""The `Problem`: the assembly-ready statement a solve strategy consumes.

A `Problem` is the resolved view of a physics composition, built for one mesh: a
space, an operator (`Form`), the load terms, and boundary conditions. It answers
the questions a solver needs: `constraints` (which DOFs are fixed), `load` (the
right-hand side), `residual(u)`, `tangent(u)`, and, where the operator has an energy,
`energy(u)`. Below it, `DiscreteSystem` sees only a matrix and a partition, so a
solve strategy never learns which PDE it is solving. After the solve, `solution(u)`
packages the DOF vector as the typed `Solution` its physics recovers (stress for
elasticity, flux for Poisson).

The operator is one `Form`, a sum of terms: the physics form the problem was stated
with plus, for each Robin condition, `kappa` times the boundary mass over that
region's facets. The load is a tuple of `fem.loads.Load` terms: the volume source, one
`Traction` per Neumann condition, one per Robin value, and any extra terms (a
`PointLoad`). Energy, residual, and tangent then read

    term        energy         residual      tangent
    operator    Π(u)           R(u)          ∂R/∂u
    load        -fᵀ u          -f            0

with the operator's terms handled by the space (`assemble`, `assemble_residual`,
`assemble_tangent`, `total_energy`) and the load's by `fem.loads.total_load`.

A transient problem also has a mass side, `mass` = density times the space's consistent
mass matrix, and optionally a `damping` matrix (`RayleighDamping`), which the
integrators and modal analysis pair with the tangent. A source or boundary value may be
a `TimeDependent` field; `load` and `constraints` are their values at `t = 0`, and
`load_at(t)` / `constraints_at(t)` re-evaluate them, which the integrators call per
step. `at(t)` is the steady snapshot with every value fixed at `t`, which a steady solve
(`solve(t=...)`) or an estimator works on.

`LinearProblem` is the case whose operator has a constant tangent (every term a
`BilinearForm`): the matrix is assembled once and held, and the residual is affine.
Everything that needs one fixed operator (a direct solve, the integrators, an
eigenproblem, SIMP) requires it. Both own their constraints, resolved from the BC spec
once; a driver that remeshes builds a new `Problem`. Named PDEs are `Equation`s
(`fem.equations`), whose `problem` builds one of these.
"""
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fem.boundary import BoundaryConditions, ResolvedBC
from fem.forms import Form, LinearForm, MaskedMassForm
from fem.loads import FixedLoad, Load, Source, Traction, as_load, total_load
from fem.regions import field_at
from fem.solution import FieldSolution
from fem.space import FunctionSpace
from fem.typing import Constraints, DofVector, FieldValue, FloatArray, Operator

if TYPE_CHECKING:
    from fem.backends import Backend
    from fem.solve import SolveStrategy

__all__ = ['Problem', 'LinearProblem', 'RayleighDamping', 'Source']


@dataclass(frozen=True)
class RayleighDamping:
    '''Proportional damping C = alpha M + beta K: `alpha` damps the low modes (mass
    proportional), `beta` the high ones (stiffness proportional). The modal damping
    ratio is ζ(ω) = alpha / (2ω) + beta ω / 2.'''
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError(f'damping coefficients must be non-negative, got {self}')

    def matrix(self, mass: Operator, stiffness: Operator) -> Operator:
        return self.alpha * mass + self.beta * stiffness


class Problem:
    '''R(u) = 0: an operator, a load, and boundary conditions on one space.

    `source` is the volume load: a field, a `LinearForm`, or a `Source`, kept as the
    normalized load term beside the assembled load so a residual estimator can read
    the pointwise source it needs. `loads` are extra load terms (a `PointLoad`) added
    to those the boundary conditions supply. `bc` is the mesh-independent spec the
    constraints were resolved from, and `resolved` that resolution on this space.
    `density` scales the mass matrix (`mass`); it is the coefficient on the
    time-derivative term. `damping` is the `RayleighDamping` a second-order integrator
    reads, or None.
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form,
        source: FieldValue | LinearForm | Source = None,
        bc: BoundaryConditions | None = None,
        density: float = 1.0,
        loads: tuple[Load, ...] = (),
        damping: RayleighDamping | None = None,
    ) -> None:
        if density <= 0:
            raise ValueError(f'density must be positive, got {density}')
        self.space = space
        self.density = density
        self.damping = damping
        self.bc = bc if bc is not None else BoundaryConditions()
        self._resolved = self.bc.resolve(space.nodes, space.n_components)

        # A Robin condition contributes to both sides: κ∫_Γ u·v on the operator and
        # ∫_Γ g·v on the load, each over the region's facets. The operator half is a
        # term of the operator, so every consumer of `operator` sees it; `physics` is
        # the form the problem was stated with, for a driver that rebuilds it.
        self.physics = operator
        self._boundary_terms: tuple[Form, ...] = tuple(
            robin.kappa * MaskedMassForm(space.n_components, robin.facet_mask)
            for robin in self._resolved.robin
        )
        self.operator: Form = self._with_boundary_terms(operator)

        # Callers pass only the volume source; the BC resolution supplies the traction
        # terms, one masked boundary integral per condition so a traction stays on its
        # own facets instead of spreading through a shared corner node.
        self.source = as_load(source, space.n_components)
        boundary_loads: list[Load] = []
        for neumann in self._resolved.neumann:
            boundary_loads.append(
                Traction.over(space, neumann.facet_mask, neumann.node_idxs, neumann.value))
        for robin in self._resolved.robin:
            boundary_loads.append(Traction.over(space, robin.facet_mask, robin.node_idxs, robin.value))
        self.loads: tuple[Load, ...] = (
            (() if self.source is None else (self.source,)) + tuple(boundary_loads) + tuple(loads)
        )
        # Set by `at(t)`: a snapshot answers no time-dependent question.
        self._frozen = False
        self._b = total_load(self.loads, space, 0.0)
        # A constant tangent is assembled on first use, not here. Stating a problem
        # is cheap; assembling its operator is the expensive half, and a problem can
        # be built without ever being solved: a topology optimization iteration
        # derives its own operator from a template whose own operator is never assembled.
        self._A: Operator | None = None
        self._M: Operator | None = None
        self._C: Operator | None = None

    @property
    def is_time_dependent(self) -> bool:
        '''Whether the source, an extra load, or any boundary value is a
        `TimeDependent` field. A snapshot from `at(t)` is not.'''
        if self._frozen:
            return False
        return any(term.is_time_dependent for term in self.loads) or self.bc.is_time_dependent

    def at(self, t: float) -> 'Problem':
        '''This problem with every time-dependent value fixed at time `t`: a steady
        problem sharing the space, operator, and boundary integrals.'''
        if not self.is_time_dependent:
            return self
        snapshot = copy.copy(self)
        snapshot._frozen = True
        snapshot.loads = tuple(
            FixedLoad(term.vector(self.space, t)) if term.is_time_dependent else term
            for term in self.loads
        )
        # The source stays a pointwise field (an estimator reads it), fixed at `t`.
        if isinstance(self.source, LinearForm):
            snapshot.source = self.source.at(t)
        elif isinstance(self.source, Source):
            snapshot.source = Source(field_at(self.source.field, t))
        snapshot._resolved = self._resolved.at(t)
        snapshot._b = total_load(snapshot.loads, self.space, t)
        return snapshot

    def load_at(self, t: float) -> DofVector:
        '''The load at time `t`; `load` itself when nothing depends on time.'''
        if not self.is_time_dependent:
            return self._b
        return total_load(self.loads, self.space, t)

    def constraints_at(self, t: float) -> Constraints:
        '''The Dirichlet partition at time `t`: the same DOFs as `constraints`, with
        the prescribed values of a `TimeDependent` condition taken at `t`.'''
        if self._frozen:
            return self.constraints
        r = self._resolved.at(t)
        return (r.free_idxs, r.fixed_idxs, r.fixed_values)

    @property
    def mass(self) -> Operator:
        '''`density` times the consistent mass matrix, assembled on first use and held.'''
        if self._M is None:
            self._M = self.density * self.space.mass_matrix
        return self._M

    @property
    def damping_matrix(self) -> Operator | None:
        '''`C = alpha M + beta K` for a problem with `damping`, else None. Needs a
        constant tangent, assembled on first use and held.'''
        if self.damping is None:
            return None
        if self._C is None:
            self._C = self.damping.matrix(self.mass, self.tangent())
        return self._C

    @property
    def is_linear(self) -> bool:
        '''Whether the operator's tangent is constant.'''
        return self.operator.constant_tangent

    @property
    def has_energy(self) -> bool:
        '''Whether the residual is the gradient of an energy.

        Only the line search reads this: with an energy it scores a step by Π(u),
        without one by ½‖r‖². The Newton iteration itself is the same either way.
        '''
        return self.operator.has_energy

    @property
    def resolved(self) -> ResolvedBC:
        return self._resolved

    @property
    def constraints(self) -> Constraints:
        r = self._resolved
        return (r.free_idxs, r.fixed_idxs, r.fixed_values)

    @property
    def load(self) -> DofVector:
        return self._b

    def with_operator(self, operator: Form) -> 'Problem':
        '''The same problem stated with a different physics operator.

        Which DOFs are constrained and what the load is follow from the boundary
        conditions and the source, neither of which the operator enters, so a driver
        re-solving under a rebuilt operator (a topology optimization iteration
        rescaling the stiffness) keeps them rather than resolving the BCs and
        reassembling the load per solve. The Robin terms of the operator are kept
        alongside the new physics.

        A new problem rather than a mutation: the two share the constraints and load
        they agree on, and nothing here writes to either.
        '''
        derived = copy.copy(self)
        derived.physics = operator
        derived.operator = self._with_boundary_terms(operator)
        # The copy carries this problem's assembled operator and damping, which are
        # precisely what the derived one must not answer with.
        derived._A = None
        derived._C = None
        return derived

    def _with_boundary_terms(self, physics: Form) -> Form:
        operator = physics
        for term in self._boundary_terms:
            operator = operator + term
        return operator

    def tangent(self, u: DofVector | None = None) -> Operator:
        '''∂R/∂u at `u`; with `u` omitted, the constant tangent of a linear problem.'''
        if self.is_linear:
            # Assembled once, on the first call, and held: the operator is constant,
            # so a Newton loop or a time-stepper asking repeatedly pays for one assembly.
            if self._A is None:
                self._A = self.space.assemble(self.operator)
            return self._A
        if u is None:
            raise ValueError(
                f'{type(self.operator).__name__} has a state-dependent tangent; evaluate '
                'it at a u. A solve that needs one fixed operator needs a LinearProblem.'
            )
        return self.space.assemble_tangent(self.operator, u)

    def internal_residual(self, u: DofVector) -> DofVector:
        '''The internal force at `u`, the residual without the load: the operator's
        residual, boundary terms included. Kept apart from `load` so a strategy can
        scale the load against it.'''
        if self.is_linear:
            return self.tangent() @ u
        return self.space.assemble_residual(self.operator, u)

    def residual(self, u: DofVector) -> DofVector:
        return self.internal_residual(u) - self._b

    def energy(self, u: DofVector) -> float:
        '''The potential Π(u) whose gradient is `residual(u)`, for an operator with an energy.'''
        if not self.has_energy:
            raise TypeError(f'{type(self.operator).__name__} has no energy to minimise')
        if self.is_linear:
            return float(0.5 * u @ (self.tangent() @ u) - self._b @ u)
        return float(self.space.total_energy(self.operator, u) - self._b @ u)

    def solution(self, u: DofVector) -> FieldSolution:
        '''The typed `Solution` the operator recovers: an `ElasticSolution` for an
        operator that recovers stress, a `ScalarFieldSolution` for one naming a flux
        (Poisson's gradient), else a bare `FieldSolution` (a projection).'''
        return self.operator.solution(self.space, u)

    def solve(
        self,
        strategy: 'SolveStrategy | None' = None,
        backend: 'Backend | None' = None,
        u0: DofVector | None = None,
        t: float | None = None,
    ) -> FieldSolution:
        '''Solve and package the result as the typed `Solution` for this operator.

        `strategy` None is `default_strategy`: `LinearSolve` for a constant tangent,
        line-searched `NewtonSolve` otherwise, over `backend`. A strategy carries its
        own backend, so the two are not given together. `u0` seeds an iterative
        strategy. A time-dependent problem is solved as its snapshot `at(t)`, so `t`
        is required for one; an integrator steps it instead.
        '''
        if strategy is not None and backend is not None:
            raise ValueError('a strategy carries its own backend; pass one or the other')
        if self.is_time_dependent:
            if t is None:
                raise ValueError(
                    'the problem has a time-dependent source or boundary value; pass t= '
                    'for a steady solve at that time, or step it with an integrator'
                )
            return self.at(t).solve(strategy, backend, u0)
        if strategy is None:
            from fem.solve import default_strategy
            strategy = default_strategy(self, backend)
        return self.solution(strategy.solve(self, u0))

    def near_null_space(self) -> FloatArray | None:
        '''The operator's AMG near-kernel over all DOFs, or None.

        `LinearSolve` hands it to an `IterativeBackend`, so an elastic solve composed
        by hand converges as well as one through the facade.
        '''
        return self.operator.near_null_space(self.space)


class LinearProblem(Problem):
    '''a(u, v) = L(v): a `Problem` whose operator has a constant tangent.

    The type every consumer that needs one fixed operator asks for. `tangent()` with
    no state is the assembled matrix, held after the first call.
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form,
        source: FieldValue | LinearForm | Source = None,
        bc: BoundaryConditions | None = None,
        density: float = 1.0,
        loads: tuple[Load, ...] = (),
        damping: RayleighDamping | None = None,
    ) -> None:
        if not operator.constant_tangent:
            raise TypeError(
                f'{type(operator).__name__} has a state-dependent tangent; state it as a '
                'Problem and solve it with NewtonSolve.'
            )
        super().__init__(space, operator, source, bc, density, loads, damping)

    def with_operator(self, operator: Form) -> 'LinearProblem':
        if not operator.constant_tangent:
            raise TypeError(f'{type(operator).__name__} has a state-dependent tangent')
        derived = super().with_operator(operator)
        assert isinstance(derived, LinearProblem)
        return derived
