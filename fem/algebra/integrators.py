"""Time integrators: a scheme applied to a semi-discrete `Problem`.

Heat is first order (M u' + K u = b(t)), wave is second (M u'' + C u' + K u = b(t)), so
there is one integrator family per order. Each forms a constant effective operator from the
problem's mass and stiffness, factors it once through `DiscreteSystem`, and steps by
updating only the right-hand side, re-evaluating a time-dependent load
(`Problem.load_at`) each step. `dt` and the step count live here; the initial state
is the problem's (`Problem.initial` and `initial_rate`, from the conditions' `Initial`),
unless `solve` is handed a `u0` (and `v0`) to continue from, a `NodalField` or its DOF
vector. The result is a `TransientSolution` that packages any step as the typed steady
solution (`history[i]`).

The wave path uses Newmark rather than a 2N first-order block: its effective operator
`M + β dt² K` is SPD and N-sized, so it stays inside the CG/preconditioning path.
"""
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from fem.algebra.backends import Backend
from fem.field import NodalField
from fem.problem import Problem
from fem.post.solution import FieldSolution, TransientSolution, WaveSolution
from fem.algebra.system import DiscreteSystem
from fem.typing import DofVector

S = TypeVar('S', bound=FieldSolution)


def _require_order(problem: Problem, order: int, what: str, use: str) -> None:
    '''Refuse a problem whose equation has no meaning at this time order.'''
    if order not in problem.time_orders:
        raise TypeError(
            f'{what}; this problem allows time orders {sorted(problem.time_orders)}. Use {use}.'
        )


def _initial_state(problem: Problem, u0: DofVector | NodalField | None,
                   v0: DofVector | NodalField | None = None) -> tuple[DofVector, DofVector]:
    '''`(u, du/dt)` at `t = 0`: the problem's own unless overridden, checked against
    the Dirichlet data either way.'''
    u = problem.initial.dofs if u0 is None else np.asarray(u0, dtype=float)
    v = problem.initial_rate.dofs if v0 is None else np.asarray(v0, dtype=float)
    problem.resolved.check_initial(u, v)
    return u.copy(), v.copy()


def _history(problem: Problem[S], t_values: list[float], u_values: list[DofVector],
             dudt_values: list[DofVector] | None = None) -> TransientSolution[S]:
    '''Package a time series into the matching transient solution type.'''
    t, dofs = np.asarray(t_values), np.array(u_values)
    if dudt_values is not None:
        return WaveSolution(problem.space, t, dofs, np.array(dudt_values), operator=problem.operator)
    return TransientSolution(problem.space, t, dofs, operator=problem.operator)


@dataclass(frozen=True)
class ThetaMethod:
    '''First-order integrator for M u' + K u = b.

    θ = ½ is Crank–Nicolson (second-order accurate, the default); θ = 1 is backward
    Euler. The step is (M + θ dt K) u_{n+1} = (M − (1−θ) dt K) u_n + dt ((1−θ) b_n + θ b_{n+1}),
    whose LHS is constant, so it is factored once and reused. A time-dependent Dirichlet
    value is prescribed at each step's end time.
    '''

    dt: float
    steps: int
    theta: float = 0.5

    def solve(self, problem: Problem[S], u0: DofVector | NodalField | None = None, *,
              backend: Backend | None = None) -> TransientSolution[S]:
        '''Step from the problem's initial state, or from `u0` (a `NodalField` or its
        DOF vector) to continue a series; `backend` solves the factored-once step
        operator.'''
        _require_order(problem, 1, 'ThetaMethod integrates a first-order system', 'Heat')
        u, _ = _initial_state(problem, u0)
        M = problem.mass
        K = problem.tangent(None)
        dt, theta = self.dt, self.theta

        system = DiscreteSystem(M + theta * dt * K, problem.constraints, backend)
        rhs_operator = M - (1 - theta) * dt * K

        b = problem.load_at(0.0)
        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        for i in range(self.steps):
            t_next = dt * (i + 1)
            b_next = problem.load_at(t_next)
            rhs = rhs_operator @ u + dt * ((1 - theta) * b + theta * b_next)
            u = system.solve(rhs, fixed_values=problem.constraints_at(t_next)[2])
            b = b_next
            t_values.append(t_next)
            u_values.append(u.copy())
        return _history(problem, t_values, u_values)


@dataclass(frozen=True)
class NewmarkMethod:
    '''Second-order integrator for M u'' + C u' + K u = b, with C the problem's
    `damping_matrix` (none by default).

    β = ¼, γ = ½ is the average-acceleration scheme: unconditionally stable and, for
    a linear undamped system, energy-conserving. It solves for the acceleration
    against the SPD operator M + γ dt C + β dt² K, an N-sized system factored once. Constant
    Dirichlet displacement means zero velocity and acceleration at the fixed nodes,
    so those DOFs are pinned to zero in the acceleration solve: the ordinary
    constraint, no lifting into a 2N block. A time-dependent load is re-evaluated at
    each step's end time; time-dependent Dirichlet data (prescribed motion) is not
    supported, since it needs the prescribed velocity and acceleration as well.
    '''

    dt: float
    steps: int
    beta: float = 0.25
    gamma: float = 0.5

    def solve(self, problem: Problem[S], u0: DofVector | NodalField | None = None,
              v0: DofVector | NodalField | None = None, *,
              backend: Backend | None = None) -> WaveSolution[S]:
        '''Step from the problem's initial state and rate, or from `(u0, v0)` (fields
        or DOF vectors) to continue a series; `backend` solves the factored-once
        effective operator.'''
        _require_order(problem, 2, 'NewmarkMethod integrates a second-order system',
                       'Wave or an elastic equation')
        if problem.conditions.has_time_dependent_dirichlet:
            raise NotImplementedError(
                'NewmarkMethod takes a time-dependent load but not time-dependent '
                'Dirichlet data, which would need the prescribed velocity and acceleration'
            )
        M = problem.mass
        K = problem.tangent(None)
        b = problem.load_at(0.0)
        free, fixed, _ = problem.constraints
        u, v = _initial_state(problem, u0, v0)

        dt, beta, gamma = self.dt, self.beta, self.gamma
        accel_constraints = (free, fixed, np.zeros(len(fixed)))

        # Initial acceleration from M a0 = b − C v0 − K u0, pinned to zero at fixed DOFs.
        C = problem.damping_matrix

        def damping(velocity: DofVector) -> DofVector:
            '''The damping force C v, zero without a damping matrix.'''
            return np.zeros_like(velocity) if C is None else C @ velocity

        a = DiscreteSystem(M, accel_constraints, backend).solve(b - damping(v) - K @ u)
        effective_operator = M + beta * dt**2 * K
        if C is not None:
            effective_operator = effective_operator + gamma * dt * C
        effective = DiscreteSystem(effective_operator, accel_constraints, backend)

        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        dudt_values: list[DofVector] = [v.copy()]
        for i in range(self.steps):
            t_next = dt * (i + 1)
            u_pred = u + dt * v + dt**2 / 2 * (1 - 2 * beta) * a
            v_pred = v + dt * (1 - gamma) * a
            a = effective.solve(problem.load_at(t_next) - damping(v_pred) - K @ u_pred)
            u = u_pred + beta * dt**2 * a
            v = v_pred + gamma * dt * a
            t_values.append(t_next)
            u_values.append(u.copy())
            dudt_values.append(v.copy())
        history = _history(problem, t_values, u_values, dudt_values)
        assert isinstance(history, WaveSolution)
        return history


def wave_energy(problem: Problem, u: DofVector, v: DofVector) -> float:
    '''Total wave energy ½(uᵀ K u + vᵀ M v), with K the c²-scaled stiffness and M
    the problem's mass.

    The quantity average-acceleration Newmark conserves for a linear system, so it
    is a usable integrator diagnostic. The consistent mass matrix is load-bearing:
    pairing a lumped kinetic term with the exact potential one makes the total swing
    as energy sloshes between them, a pure measurement artifact.
    '''
    M = problem.mass
    K = problem.tangent(None)
    return float(0.5 * (u @ K @ u + v @ M @ v))
