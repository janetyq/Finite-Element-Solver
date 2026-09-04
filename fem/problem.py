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
region's facets. The load is the sum of the resolution's `fem.loads.Load` terms (the
volume source, one `BoundaryLoad` per Neumann condition, one per Robin value, and any
`PointLoad`) plus any load the operator's own physics contributes (`operator_load`, the
thermal load of a heated elastic body). Energy, residual, and tangent then read

    term        energy         residual      tangent
    operator    Π(u)           R(u)          ∂R/∂u
    load        -fᵀ u          -f            0

with the operator's terms handled by the space (`assemble`, `assemble_residual`,
`assemble_tangent`, `total_energy`) and the load's by `ResolvedConditions.load_at`.

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
eigenproblem, SIMP) requires it. Both own their `Conditions`, resolved on the space
once; a driver that remeshes builds a new `Problem`. Named PDEs are `Equation`s
(`fem.physics.equations`), whose `problem` builds one of these.

`Problem.solve` imports `fem.algebra.solve` lazily for `default_strategy`: the strategies
consume a `Problem`, so the edge points up and stays function-local.
"""
import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from fem.algebra.backends import Backend
from fem.conditions import Conditions, Initial, ResolvedConditions
from fem.field import NodalField
from fem.physics.forms import Form
from fem.loads import Load, Source
from fem.post.solution import FieldSolution
from fem.space import FunctionSpace
from fem.typing import Constraints, DofVector, FloatArray, Operator

S = TypeVar('S', bound=FieldSolution)     # the solution the operator packages
S2 = TypeVar('S2', bound=FieldSolution)
P = TypeVar('P', bound='Problem[Any]')      # the problem's own type, for copies

if TYPE_CHECKING:
    from fem.boundary import Robin
    from fem.algebra.solve import SolveStrategy

__all__ = ['Problem', 'LinearProblem', 'RayleighDamping']


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


class Problem(Generic[S]):
    '''R(u) = 0: an operator with conditions on one space.

    `conditions` is the mesh-independent `Conditions` (boundary conditions, the volume
    source, point loads) and `resolved` its resolution on this space: the constraints,
    the operator terms a Robin condition adds, and the load terms, whose sum is `load`.
    `source` and `loads` are the resolution's; the source is kept as a pointwise field
    so a residual estimator can read it. `density` scales the mass matrix (`mass`); it is the
    coefficient on the time-derivative term. `damping` is the `RayleighDamping` a second-order integrator
    reads, or None. `time_orders` is the set of time-derivative orders the problem has
    a meaning for (an `Equation`'s; every order for a problem composed by hand), which
    `solve` and the integrators check.

    `S` is the typed `Solution` the operator packages (`Form[S]`), so `solve` and
    `solution` return it: `Problem[ElasticSolution]` for an elastic operator.
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form[S],
        conditions: Conditions | None = None,
        *,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
        time_orders: frozenset[int] = frozenset({0, 1, 2}),
    ) -> None:
        if density <= 0:
            raise ValueError(f'density must be positive, got {density}')
        self.space = space
        self.density = density
        self.damping = damping
        self.time_orders = frozenset(time_orders)
        self.conditions = conditions if conditions is not None else Conditions()
        self._resolved = self.conditions.resolve(space)

        # The Robin terms are part of the operator, so every consumer of `operator`
        # sees them; `physics` is the form the problem was stated with, for a driver
        # that rebuilds it.
        self.physics: Form[S] = operator
        self._boundary_terms: tuple[Form, ...] = self._resolved.operator_terms
        self.operator: Form[S] = self._with_boundary_terms(operator)
        # The conditions' load and the operator's are kept apart: a driver restating
        # the problem under a new operator replaces the second and keeps the first.
        self._conditions_load = self._resolved.load_at(0.0)
        self._operator_load = space.assemble_loads(self.operator)
        self._b = self._sum_loads(self._conditions_load)
        # A constant tangent is assembled on first use, not here. Stating a problem
        # is cheap; assembling its operator is the expensive half, and a problem can
        # be built without ever being solved: a topology optimization iteration
        # derives its own operator from a template whose own operator is never assembled.
        self._A: Operator | None = None
        self._M: Operator | None = None
        self._C: Operator | None = None

    @property
    def is_time_dependent(self) -> bool:
        '''Whether the source, a load, or any boundary value is a `TimeDependent`
        field. A snapshot from `at(t)` is not.'''
        return self._resolved.is_time_dependent

    def at(self: P, t: float) -> P:
        '''This problem with every time-dependent value fixed at time `t`: a steady
        problem sharing the space, operator, and boundary integrals.'''
        if not self.is_time_dependent:
            return self
        snapshot = copy.copy(self)
        snapshot._resolved = self._resolved.at(t)
        snapshot._conditions_load = snapshot._resolved.load_at(t)
        snapshot._b = snapshot._sum_loads(snapshot._conditions_load)
        return snapshot

    def _total_load(self, t: float) -> DofVector:
        '''The conditions' load at `t` plus the operator's own.'''
        return self._sum_loads(self._resolved.load_at(t))

    def _sum_loads(self, conditions_load: DofVector) -> DofVector:
        '''`conditions_load` plus the operator's own load; the vector itself without one.'''
        if self._operator_load is None:
            return conditions_load
        return conditions_load + self._operator_load

    @property
    def operator_load(self) -> DofVector | None:
        '''The load the operator's own physics contributes, or None, such as the
        thermal load of a heated elastic body. Assembled once from `Form.element_loads`
        and included in `load` with the loads from the conditions.'''
        return self._operator_load

    def with_load_factor(self: P, factor: float) -> P:
        '''This problem with its whole loading scaled by `factor`: the assembled load
        vector and the prescribed Dirichlet values alike, so `factor` walks the
        proportional loading path from rest (0) to the stated problem (1) and beyond.
        What quasi-static continuation (`QuasiStaticStepping`) solves at each step.

        A snapshot sharing the space, operator, and constraint partition; only the
        values scale. The load *terms* (`loads`, `source`) are left as stated, the way
        `at(t)` leaves them: `load` and `constraints` are the scaled data a solve
        reads. The initial state is not a loading and keeps its scale. A problem with
        time-dependent values has no one loading to scale; fix it first with `at(t)`.
        '''
        if self.is_time_dependent:
            raise ValueError(
                'with_load_factor scales one fixed loading; a time-dependent problem '
                'has a different one per t. Take a snapshot first: problem.at(t).'
            )
        snapshot = copy.copy(self)
        resolved = self._resolved
        snapshot._resolved = replace(resolved, fixed_values=factor * resolved.fixed_values)
        snapshot._conditions_load = factor * self._conditions_load
        snapshot._b = factor * self._b
        return snapshot

    def load_at(self, t: float) -> DofVector:
        '''The load at time `t`; `load` itself when nothing depends on time.'''
        if not self.is_time_dependent:
            return self._b
        return self._total_load(t)

    def constraints_at(self, t: float) -> Constraints:
        '''The Dirichlet partition at time `t`: the same DOFs as `constraints`, with
        the prescribed values of a `TimeDependent` condition taken at `t`.'''
        return self._resolved.constraints_at(t)

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
    def resolved(self) -> ResolvedConditions:
        return self._resolved

    def robin_flux(self, u: DofVector | NodalField, condition: 'Robin | None' = None,
                   t: float = 0.0) -> float | FloatArray:
        '''∫_Γ (κu − g) over the region of a Robin `condition` of this problem at state
        `u`: the flux leaving the domain through it (the heat a convective film sheds),
        with a `TimeDependent` g taken at `t`. A float for a scalar problem,
        `(n_components,)` for a vector one. `condition` may be left out when the
        problem has exactly one.

        Read off the same region-restricted boundary mass the condition assembles, so
        it is the exact discrete integral, not a quadrature of the recovered gradient.
        '''
        robins = self.conditions.robin
        if condition is None:
            if len(robins) != 1:
                raise ValueError(
                    f'robin_flux needs the condition to integrate over: the problem has '
                    f'{len(robins)} Robin conditions')
            i = 0
        elif condition in robins:
            i = robins.index(condition)
        else:
            raise ValueError(f'{condition} is not a Robin condition of this problem')
        robin = self._resolved.robin
        load = self._resolved.robin_loads[i]
        n = self.space.n_components
        dofs = np.asarray(u, dtype=float)
        integral = np.asarray(load.boundary_mass @ (robin[i].kappa * dofs)).reshape(-1, n).sum(axis=0)
        integral = integral - load.vector(self.space, t).reshape(-1, n).sum(axis=0)
        return float(integral[0]) if n == 1 else integral

    @property
    def source(self) -> Source | None:
        '''The volume source as a pointwise field, or None.'''
        return self._resolved.source

    @property
    def loads(self) -> tuple[Load, ...]:
        '''The conditions' load terms, whose sum plus `operator_load` is `load`.'''
        return self._resolved.loads

    @property
    def constraints(self) -> Constraints:
        return self._resolved.constraints

    @property
    def u0(self) -> NodalField:
        '''The state a solve starts from: the conditions' `Initial`, else the Dirichlet
        lift. An integrator steps from it and `NewtonSolve` iterates from it.'''
        return self._resolved.u0

    @property
    def v0(self) -> NodalField:
        '''The time derivative of the state at `t = 0`, zero unless the `Initial` gave one.'''
        return self._resolved.v0

    @property
    def load(self) -> DofVector:
        return self._b

    def with_operator(self, operator: Form[S2]) -> 'Problem[S2]':
        '''The same problem stated with a different physics operator.

        Which DOFs are constrained and what the conditions' load is follow from the
        boundary conditions and the source, neither of which the operator enters, so a
        driver re-solving under a rebuilt operator (a topology optimization iteration
        rescaling the stiffness) keeps them rather than resolving the conditions and
        reassembling the load per solve. The operator's own load is the new operator's.
        The Robin terms of the operator are kept alongside the new physics.

        A new problem rather than a mutation: the two share the constraints and load
        they agree on, and nothing here writes to either.
        '''
        derived = self._copy()
        derived._restate(operator)
        return derived

    def _copy(self) -> 'Problem[Any]':
        '''A shallow copy whose type parameter is left open, for `with_operator`.'''
        return copy.copy(self)

    def _restate(self, operator: Form[Any]) -> None:
        '''Rebind a copy to `operator`; its type parameter is the operator's.'''
        self.physics = operator
        self.operator = self._with_boundary_terms(operator)
        # The copy carries the original's assembled operator and damping, which are
        # precisely what the derived one must not answer with, and the original
        # operator's load, which the new operator's replaces. The conditions' load is
        # kept as it was assembled.
        self._A = None
        self._C = None
        self._operator_load = self.space.assemble_loads(self.operator)
        self._b = self._sum_loads(self._conditions_load)

    def _with_boundary_terms(self, physics: Form[S2]) -> Form[S2]:
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

    def solution(self, u: DofVector) -> S:
        '''The typed `Solution` the operator recovers: an `ElasticSolution` for an
        operator that recovers stress, a `DiffusionSolution` for one naming a flux
        (Poisson's gradient), else a bare `FieldSolution` (a projection).'''
        return self.operator.solution(self.space, u)

    def solve(
        self,
        strategy: 'SolveStrategy | None' = None,
        backend: Backend | None = None,
        initial: Initial | None = None,
        t: float | None = None,
    ) -> S:
        '''Solve and package the result as the typed `Solution` for this operator.

        `strategy` is how the problem is iterated (`LinearSolve`, `NewtonSolve`); None is
        `default_strategy`, `LinearSolve` for a constant tangent and line-searched
        `NewtonSolve` otherwise. `backend` is how each linear system on the way is solved
        (direct by default); the two are independent choices. `initial` seeds an
        iterative strategy in place of the conditions' own `Initial`. A time-dependent problem is solved as its snapshot `at(t)`, so `t` is
        required for one; an integrator steps it instead.
        '''
        if 0 not in self.time_orders:
            raise TypeError(
                f'a steady solve needs time order 0; this problem allows {sorted(self.time_orders)}. '
                'The steady state of the heat or wave equation is Poisson(coefficient=...).'
            )
        if self.is_time_dependent:
            if t is None:
                raise ValueError(
                    'the problem has a time-dependent source or boundary value; pass t= '
                    'for a steady solve at that time, or step it with an integrator'
                )
            return self.at(t).solve(strategy, backend, initial)
        if strategy is None:
            from fem.algebra.solve import default_strategy
            strategy = default_strategy(self)
        return self.solution(strategy.solve(self, initial=initial, backend=backend))

    def near_null_space(self) -> FloatArray | None:
        '''The operator's AMG near-kernel over all DOFs, or None.

        `LinearSolve` hands it to an `IterativeBackend`, so an elastic solve composed
        by hand converges as well as one through the facade.
        '''
        return self.operator.near_null_space(self.space)


class LinearProblem(Problem[S]):
    '''a(u, v) = L(v): a `Problem` whose operator has a constant tangent.

    The type every consumer that needs one fixed operator asks for. `tangent()` with
    no state is the assembled matrix, held after the first call.
    '''

    def __init__(
        self,
        space: FunctionSpace,
        operator: Form[S],
        conditions: Conditions | None = None,
        *,
        density: float = 1.0,
        damping: RayleighDamping | None = None,
        time_orders: frozenset[int] = frozenset({0, 1, 2}),
    ) -> None:
        if not operator.constant_tangent:
            raise TypeError(
                f'{type(operator).__name__} has a state-dependent tangent; state it as a '
                'Problem and solve it with NewtonSolve.'
            )
        super().__init__(space, operator, conditions, density=density, damping=damping,
                         time_orders=time_orders)

    def with_operator(self, operator: Form[S2]) -> 'LinearProblem[S2]':
        if not operator.constant_tangent:
            raise TypeError(f'{type(operator).__name__} has a state-dependent tangent')
        derived = self._copy()
        derived._restate(operator)
        return derived

    def _copy(self) -> 'LinearProblem[Any]':
        return copy.copy(self)
