"""Time integrators: a scheme applied to a semi-discrete `Problem`.

Heat is first order (M u' + r_int(u) = b(t)), wave is second
(M u'' + C u' + r_int(u) = b(t)), so there is one integrator family per order. Both
step the same two ways, dispatching on `Problem.is_linear`:

- A constant tangent gives a constant effective operator from the problem's mass and
  stiffness, factored once through `DiscreteSystem`; each step updates only the
  right-hand side.
- A state-dependent tangent (an `EnergyForm`, finite-strain elasticity) gives an
  effective residual per step, solved by a plain Newton iteration seeded from the
  scheme's own predictor, its tangent (the mass shift plus `K_T(u)`) factored per
  iteration. `dt` bounds how nonlinear one step is, so no line search is needed;
  exhausting `newton_max_iters` raises `NewtonDivergence` advising a smaller `dt`.

A time-dependent load (`Problem.load_at`) is re-evaluated each step either way. `dt`
and the step count live here; the initial state is the problem's (`Problem.u0` and
`v0`, from the conditions' `Initial`), unless `solve` is handed an `initial=` to
continue from, an `Initial` over a previous step, say. The result is a
`TransientSolution` that packages any step as the typed steady solution
(`solution[i]`).

The wave path uses Newmark rather than a 2N first-order block: its effective operator
`M + β dt² K` is SPD and N-sized, so it stays inside the CG/preconditioning path.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from fem.algebra.solve import NewtonDivergence
from fem.algebra.system import DiscreteSystem
from fem.conditions import Initial
from fem.post.solution import FieldSolution, TransientSolution, WaveSolution
from fem.problem import Problem
from fem.typing import DofVector, Operator

S = TypeVar('S', bound=FieldSolution)

# The effective residual and its tangent at a trial state: what one step's Newton reads.
_EffectiveSystem = Callable[[DofVector], tuple[DofVector, Operator]]
# One step of a scheme, built once per solve and closed over the linear or nonlinear
# path: `(u, b, b_next, t_next) -> u_next` for a first-order scheme, and
# `(u, v, a, t_next) -> (u, v, a)` for a second-order one, which carries its own state.
_FirstOrderStep = Callable[[DofVector, DofVector, DofVector, float], DofVector]
_SecondOrderStep = Callable[
    [DofVector, DofVector, DofVector, float], tuple[DofVector, DofVector, DofVector]
]


def _require_order(problem: Problem, order: int, what: str, use: str) -> None:
    '''Refuse a problem whose equation has no meaning at this time order.'''
    if order not in problem.time_orders:
        raise TypeError(
            f'{what}; this problem allows time orders {sorted(problem.time_orders)}. Use {use}.'
        )


def _initial_state(problem: Problem, initial: Initial | None) -> tuple[DofVector, DofVector]:
    '''`(u, du/dt)` at `t = 0`: the problem's own, or `initial` resolved on its space
    and checked against the Dirichlet data.'''
    u0, v0 = ((problem.u0, problem.v0) if initial is None
              else problem.resolved.resolve_initial(initial))
    return u0.dofs.copy(), v0.dofs.copy()


def _history(problem: Problem[S], t_values: list[float], u_values: list[DofVector],
             dudt_values: list[DofVector] | None = None) -> TransientSolution[S]:
    '''Package a time series into the matching transient solution type.'''
    t, dofs = np.asarray(t_values), np.array(u_values)
    if dudt_values is not None:
        return WaveSolution(problem.space, t, dofs, np.array(dudt_values), operator=problem.operator)
    return TransientSolution(problem.space, t, dofs, operator=problem.operator)


def _internal_and_tangent(problem: Problem, u: DofVector) -> tuple[DofVector, Operator]:
    '''`(internal_residual(u), tangent(u))` from one assembly pass.

    `Problem.residual_and_tangent` evaluates the operator once for both, which an
    energy form would otherwise do twice; the load it subtracts is added back, since a
    step's effective residual carries its own load at the step's end time.
    '''
    residual, tangent = problem.residual_and_tangent(u)
    return residual + problem.load, tangent


def _newton_per_step(
    problem: Problem, seed: DofVector, effective: _EffectiveSystem,
    tol: float, max_iters: int,
) -> DofVector:
    '''The state solving one step's effective residual, by plain Newton from `seed`.

    No line search: `dt` bounds how far the predictor is from the answer, so the full
    step converges where an unseeded solve of the same physics would need globalizing.
    The increment is pinned to zero at the fixed DOFs (the seed carries their
    prescribed values), so every solve here is the homogeneous one. Convergence is the
    relative step norm `NewtonSolve` uses, checked before the step is applied.
    '''
    u = seed.copy()
    step_norm = np.inf
    for _ in range(max_iters):
        residual, tangent = effective(u)
        system = DiscreteSystem(tangent, problem.partition, problem.backend)
        step = system.solve_homogeneous(-residual)
        step_norm = float(np.linalg.norm(step))
        if step_norm < tol * max(1.0, float(np.linalg.norm(u))):
            return u
        u = u + step
    raise NewtonDivergence(
        f'the step did not converge in {max_iters} Newton iterations: the last increment '
        f'had norm {step_norm:.3e} against a tolerance of {tol:.1e} relative to the state. '
        f'Reduce dt, which is what bounds the nonlinearity of one step, or raise '
        f'newton_max_iters; the last iterate is on the exception as `u`.',
        u, max_iters, step_norm,
    )


def _seeded_at(problem: Problem, predictor: DofVector, t: float) -> DofVector:
    '''The predictor with the values prescribed at `t` written into the fixed DOFs, so
    the Newton increment can stay homogeneous.'''
    seed = predictor.copy()
    seed[problem.partition.fixed] = problem.fixed_values_at(t)
    return seed


@dataclass(frozen=True)
class ThetaMethod:
    '''First-order integrator for M u' + r_int(u) = b.

    θ = ½ is Crank–Nicolson (second-order accurate, the default); θ = 1 is backward
    Euler. The step is
    M (u_{n+1} − u_n)/dt + θ r_int(u_{n+1}) + (1−θ) r_int(u_n) = (1−θ) b_n + θ b_{n+1}.
    With a constant tangent that is (M + θ dt K) u_{n+1} = (M − (1−θ) dt K) u_n +
    dt ((1−θ) b_n + θ b_{n+1}), whose LHS is factored once and reused; with a
    state-dependent tangent it is solved for u_{n+1} by Newton, tangent M/dt + θ K_T(u),
    seeded from the previous step. A time-dependent Dirichlet value is prescribed at
    each step's end time on either path.

    `newton_tol` and `newton_max_iters` govern that inner iteration and are read only
    on the nonlinear path.
    '''

    dt: float
    steps: int
    theta: float = 0.5
    newton_tol: float = 1e-8
    newton_max_iters: int = 20

    def solve(self, problem: Problem[S], *, initial: Initial | None = None) -> TransientSolution[S]:
        '''Step from the problem's `u0`, or from `initial` to continue a series; every
        linear system on the way is solved with the problem's `backend`.'''
        _require_order(problem, 1, 'ThetaMethod integrates a first-order system', 'Heat')
        u, _ = _initial_state(problem, initial)
        M = problem.mass
        step = self._linear_step(problem, M) if problem.is_linear else self._nonlinear_step(problem, M)

        b = problem.load_at(0.0)
        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        for i in range(self.steps):
            t_next = self.dt * (i + 1)
            b_next = problem.load_at(t_next)
            u = step(u, b, b_next, t_next)
            b = b_next
            t_values.append(t_next)
            u_values.append(u.copy())
        return _history(problem, t_values, u_values)

    def _linear_step(self, problem: Problem, M: Operator) -> '_FirstOrderStep':
        '''One step against the constant effective operator, factored once here.'''
        dt, theta = self.dt, self.theta
        K = problem.tangent()
        system = DiscreteSystem(M + theta * dt * K, problem.partition, problem.backend)
        rhs_operator = M - (1 - theta) * dt * K

        def step(u: DofVector, b: DofVector, b_next: DofVector, t_next: float) -> DofVector:
            rhs = rhs_operator @ u + dt * ((1 - theta) * b + theta * b_next)
            return system.solve(rhs, fixed_values=problem.fixed_values_at(t_next))

        return step

    def _nonlinear_step(self, problem: Problem, M: Operator) -> '_FirstOrderStep':
        '''One step by Newton on the effective residual, seeded from the last state.'''
        dt, theta = self.dt, self.theta

        def step(u: DofVector, b: DofVector, b_next: DofVector, t_next: float) -> DofVector:
            # Everything the old state contributes, evaluated once for the whole step.
            known = (M @ u / dt - (1 - theta) * problem.internal_residual(u)
                     + (1 - theta) * b + theta * b_next)

            def effective(w: DofVector) -> tuple[DofVector, Operator]:
                internal, K_T = _internal_and_tangent(problem, w)
                return M @ w / dt + theta * internal - known, M / dt + theta * K_T

            return _newton_per_step(problem, _seeded_at(problem, u, t_next), effective,
                                    self.newton_tol, self.newton_max_iters)

        return step


@dataclass(frozen=True)
class NewmarkMethod:
    '''Second-order integrator for M u'' + C u' + r_int(u) = b, with C the problem's
    `damping_matrix` (none by default).

    β = ¼, γ = ½ is the average-acceleration scheme: unconditionally stable and, for
    a linear undamped system, energy-conserving. With a constant tangent it solves for
    the acceleration against the SPD operator M + γ dt C + β dt² K, an N-sized system
    factored once. With a state-dependent tangent it solves for the displacement
    instead, by Newton on
    R(u) = M ü(u) + r_int(u) − b(t_{n+1}) with tangent M/(β dt²) + K_T(u), where the
    Newmark relations give ü and u̇ from u; the mass shift keeps the tangent
    well conditioned for a small `dt`, and the predictor seeds the iteration. Damping
    is refused there: `RayleighDamping` is built on one constant stiffness.

    Constant Dirichlet displacement means zero velocity and acceleration at the fixed
    nodes, so those DOFs are pinned to zero in the acceleration solve: the ordinary
    constraint, no lifting into a 2N block. A time-dependent load is re-evaluated at
    each step's end time; time-dependent Dirichlet data (prescribed motion) is not
    supported, since it needs the prescribed velocity and acceleration as well.
    `newton_tol` and `newton_max_iters` govern the inner iteration of the nonlinear path.
    '''

    dt: float
    steps: int
    beta: float = 0.25
    gamma: float = 0.5
    newton_tol: float = 1e-8
    newton_max_iters: int = 20

    def solve(self, problem: Problem[S], *, initial: Initial | None = None) -> WaveSolution[S]:
        '''Step from the problem's `u0` and `v0`, or from `initial` to continue a
        series; every linear system on the way is solved with the problem's `backend`.'''
        _require_order(problem, 2, 'NewmarkMethod integrates a second-order system',
                       'Wave or an elastic equation')
        if problem.conditions.has_time_dependent_dirichlet:
            raise NotImplementedError(
                'NewmarkMethod takes a time-dependent load but not time-dependent '
                'Dirichlet data, which would need the prescribed velocity and acceleration'
            )
        if not problem.is_linear and problem.damping is not None:
            raise TypeError(
                'NewmarkMethod does not damp a state-dependent tangent: RayleighDamping '
                'is C = alpha M + beta K over one constant stiffness, which '
                f'{type(problem.operator).__name__} does not have. Drop the damping, or '
                'state the problem with a linear operator.'
            )
        M = problem.mass
        b = problem.load_at(0.0)
        u, v = _initial_state(problem, initial)

        # Initial acceleration from M a0 = b − C v0 − r_int(u0), pinned to zero at fixed DOFs.
        C = problem.damping_matrix
        damped = b if C is None else b - C @ v
        a = DiscreteSystem(M, problem.partition, problem.backend).solve_homogeneous(
            damped - problem.internal_residual(u))

        step = (self._linear_step(problem, M, C) if problem.is_linear
                else self._nonlinear_step(problem, M))

        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        dudt_values: list[DofVector] = [v.copy()]
        for i in range(self.steps):
            t_next = self.dt * (i + 1)
            u, v, a = step(u, v, a, t_next)
            t_values.append(t_next)
            u_values.append(u.copy())
            dudt_values.append(v.copy())
        history = _history(problem, t_values, u_values, dudt_values)
        assert isinstance(history, WaveSolution)
        return history

    def _predictors(
        self, u: DofVector, v: DofVector, a: DofVector,
    ) -> tuple[DofVector, DofVector]:
        '''The Newmark predictors `(u_pred, v_pred)`: the state at the step's end
        without the new acceleration's contribution.'''
        dt, beta, gamma = self.dt, self.beta, self.gamma
        return u + dt * v + dt**2 / 2 * (1 - 2 * beta) * a, v + dt * (1 - gamma) * a

    def _linear_step(
        self, problem: Problem, M: Operator, C: Operator | None,
    ) -> '_SecondOrderStep':
        '''One step against the constant effective operator, factored once here, solved
        for the acceleration.'''
        dt, beta, gamma = self.dt, self.beta, self.gamma
        K = problem.tangent()
        effective_operator = M + beta * dt**2 * K
        if C is not None:
            effective_operator = effective_operator + gamma * dt * C
        effective = DiscreteSystem(effective_operator, problem.partition, problem.backend)

        def step(
            u: DofVector, v: DofVector, a: DofVector, t_next: float,
        ) -> tuple[DofVector, DofVector, DofVector]:
            u_pred, v_pred = self._predictors(u, v, a)
            rhs = problem.load_at(t_next) - K @ u_pred
            if C is not None:
                rhs = rhs - C @ v_pred
            a_next = effective.solve_homogeneous(rhs)
            return u_pred + beta * dt**2 * a_next, v_pred + gamma * dt * a_next, a_next

        return step

    def _nonlinear_step(self, problem: Problem, M: Operator) -> '_SecondOrderStep':
        '''One step by Newton on the effective residual in the displacement, seeded from
        the predictor. Undamped: the constructor refuses a damped nonlinear problem.'''
        dt, beta, gamma = self.dt, self.beta, self.gamma
        shift = 1.0 / (beta * dt**2)

        def step(
            u: DofVector, v: DofVector, a: DofVector, t_next: float,
        ) -> tuple[DofVector, DofVector, DofVector]:
            u_pred, v_pred = self._predictors(u, v, a)
            b_next = problem.load_at(t_next)

            def effective(w: DofVector) -> tuple[DofVector, Operator]:
                internal, K_T = _internal_and_tangent(problem, w)
                acceleration = shift * (w - u_pred)
                return M @ acceleration + internal - b_next, shift * M + K_T

            u_next = _newton_per_step(problem, _seeded_at(problem, u_pred, t_next), effective,
                                      self.newton_tol, self.newton_max_iters)
            a_next = shift * (u_next - u_pred)
            return u_next, v_pred + gamma * dt * a_next, a_next

        return step


def wave_energy(problem: Problem, u: DofVector, v: DofVector) -> float:
    '''Total energy: the stored energy at `u` plus the kinetic energy ½ vᵀ M v, with M
    the problem's mass.

    The stored half is ½ uᵀ K u for a constant tangent (K the c²-scaled stiffness) and
    the operator's own Π(u) for a state-dependent one, which is the quantity a
    finite-strain body exchanges with its kinetic energy. Load potential is not
    included: this is the energy average-acceleration Newmark conserves on an
    unforced system, so it is a usable integrator diagnostic. The consistent mass
    matrix is load-bearing: pairing a lumped kinetic term with the exact potential one
    makes the total swing as energy sloshes between them, a pure measurement artifact.
    '''
    kinetic = 0.5 * float(v @ problem.mass @ v)
    if problem.is_linear:
        return kinetic + 0.5 * float(u @ problem.tangent() @ u)
    return kinetic + float(problem.space.total_energy(problem.operator, u))
