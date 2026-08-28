"""Time integrators: a scheme applied to a semi-discrete `Problem`.

Heat is first order (M u' + K u = b(t)), wave is second (M u'' + C u' + K u = b(t)), so
there is one integrator family per order. Each forms a constant effective operator from the
problem's mass and stiffness, factors it once through `DiscreteSystem`, and steps by
updating only the right-hand side, re-evaluating a time-dependent load
(`Problem.load_at`) each step. `dt` and the step count live here; initial conditions
come in through `solve`, as DOF vectors (`FunctionSpace.interpolate`). The result is a
`TransientSolution` that packages any step as the typed steady solution (`at(i)`).

The wave path uses Newmark rather than a 2N first-order block: its effective operator
`M + β dt² K` is SPD and N-sized, so it stays inside the CG/preconditioning path.
"""
import numpy as np

from fem.backends import Backend
from fem.problem import Problem
from fem.solution import TransientSolution, WaveSolution
from fem.system import DiscreteSystem
from fem.typing import DofVector


def _require_order(problem: Problem, order: int, what: str, use: str) -> None:
    '''Refuse a problem whose equation has no meaning at this time order.'''
    if order not in problem.time_orders:
        raise TypeError(
            f'{what}; this problem allows time orders {sorted(problem.time_orders)}. Use {use}.'
        )


def _history(problem: Problem, t_values: list[float], u_values: list[DofVector],
             dudt_values: list[DofVector] | None = None) -> TransientSolution:
    '''Package a time series into the matching transient solution type.'''
    t = np.asarray(t_values)
    if dudt_values is not None:
        return WaveSolution(problem.space, t, u_values, dudt_values, problem=problem)
    return TransientSolution(problem.space, t, u_values, problem=problem)


class ThetaMethod:
    '''First-order integrator for M u' + K u = b.

    θ = ½ is Crank–Nicolson (second-order accurate, the default); θ = 1 is backward
    Euler. The step is (M + θ dt K) u_{n+1} = (M − (1−θ) dt K) u_n + dt ((1−θ) b_n + θ b_{n+1}),
    whose LHS is constant, so it is factored once and reused. A time-dependent Dirichlet
    value is prescribed at each step's end time.
    '''

    def __init__(self, dt: float, steps: int, theta: float = 0.5,
                 backend: Backend | None = None) -> None:
        self.dt = dt
        self.steps = steps
        self.theta = theta
        self.backend = backend

    def solve(self, problem: Problem, u0: DofVector) -> TransientSolution:
        _require_order(problem, 1, 'ThetaMethod integrates a first-order system', 'Heat')
        M = problem.mass
        K = problem.tangent(None)
        dt, theta = self.dt, self.theta

        system = DiscreteSystem(M + theta * dt * K, problem.constraints, self.backend)
        rhs_operator = M - (1 - theta) * dt * K

        u = np.asarray(u0, dtype=float)
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

    def __init__(self, dt: float, steps: int, beta: float = 0.25, gamma: float = 0.5,
                 backend: Backend | None = None) -> None:
        self.dt = dt
        self.steps = steps
        self.beta = beta
        self.gamma = gamma
        self.backend = backend

    def solve(self, problem: Problem, u0: DofVector, v0: DofVector) -> WaveSolution:
        _require_order(problem, 2, 'NewmarkMethod integrates a second-order system',
                       'Wave or an elastic equation')
        if problem.bc.is_time_dependent:
            raise NotImplementedError(
                'NewmarkMethod takes a time-dependent load but not time-dependent '
                'Dirichlet data, which would need the prescribed velocity and acceleration'
            )
        M = problem.mass
        K = problem.tangent(None)
        b = problem.load_at(0.0)
        free, fixed, fixed_values = problem.constraints

        u = np.asarray(u0, dtype=float)
        v = np.asarray(v0, dtype=float)
        # An initial state that disagrees with the constraints is a modelling error:
        # the solve would otherwise jump to satisfy them at the first step.
        if not np.allclose(u[fixed], fixed_values):
            raise ValueError('u0 disagrees with the Dirichlet values at fixed nodes')
        if not np.allclose(v[fixed], 0):
            raise ValueError('v0 must be zero at Dirichlet-fixed nodes')

        dt, beta, gamma = self.dt, self.beta, self.gamma
        accel_constraints = (free, fixed, np.zeros(len(fixed)))

        # Initial acceleration from M a0 = b − C v0 − K u0, pinned to zero at fixed DOFs.
        C = problem.damping_matrix

        def damping(velocity: DofVector) -> DofVector:
            '''The damping force C v, zero without a damping matrix.'''
            return np.zeros_like(velocity) if C is None else C @ velocity

        a = DiscreteSystem(M, accel_constraints, self.backend).solve(b - damping(v) - K @ u)
        effective_operator = M + beta * dt**2 * K
        if C is not None:
            effective_operator = effective_operator + gamma * dt * C
        effective = DiscreteSystem(effective_operator, accel_constraints, self.backend)

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
