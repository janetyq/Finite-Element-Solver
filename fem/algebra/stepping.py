"""Quasi-static continuation: walk a nonlinear problem along its equilibrium path.

A strongly nonlinear solve can sit too far from its seed for Newton: from rest, the
solution at full load may be past the region the tangent predicts, and the iteration
diverges or the line search crawls. `QuasiStaticStepping` walks there instead: it
splits the loading into steps, solves the steady equilibrium at each level with the
previous level's solution as the seed, and returns the whole path as a
`TransientSolution`. Each step is then an easy solve (continuation), and the series
is the load-deflection story a nonlinear analysis usually wants, not just its end
state.

Quasi-static means inertia is negligible: no mass matrix, no rates. `t` is a dial on
the loading, not physical time — for a problem with `TimeDependent` values it is
their parameter (each step solves the snapshot `problem.at(t)`), and for a steady
problem it is the proportional load factor (`problem.with_load_factor(t)`), ramped
from rest to `t_end`.

A step that diverges is retried at half the increment, from the last converged state,
until it converges or the increment falls below `t_end / steps / 2^max_bisections`;
every converged level is kept in the history, inserted midpoints included. Exhausting
the bisections raises `SteppingDivergence`, which carries the path walked so far.

That is load control: λ is prescribed per step and the state solved for, so it cannot
pass a limit point (a fold, where equilibrium past the peak exists only at lower loads
— a snap-through). `ArcLengthStepping` is the same loop under arc-length control: λ
becomes an unknown alongside the state, and what each increment prescribes instead is
its own length `Δs` through the (u, λ) space, tied down by one scalar constraint. The
path can then turn back in λ, which is exactly what a fold requires and what no choice
of load steps can produce. It returns a `PathSolution`, the series plus the sign of
`det K_T` at each state, which flips where stability is lost.
"""
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from fem.algebra.backends import det_sign
from fem.algebra.solve import (
    BacktrackingLineSearch,
    LineSearchFailure,
    NewtonDivergence,
    NewtonSolve,
)
from fem.algebra.system import DiscreteSystem
from fem.conditions import Initial
from fem.field import NodalField
from fem.post.solution import FieldSolution, PathSolution, TransientSolution
from fem.problem import Problem
from fem.typing import DofIndices, DofVector

S = TypeVar('S', bound=FieldSolution)


class SteppingDivergence(RuntimeError):
    '''A continuation strategy could not converge an increment even at its smallest
    step.

    `history` is the path walked so far (every converged state, as the strategy would
    have returned it: a `TransientSolution` from `QuasiStaticStepping`, a
    `PathSolution` from `ArcLengthStepping`) and `t` the last level reached, the load
    factor either way, so a caller can read how far the walk got, plot it, or continue
    with a different strategy from its end.'''

    def __init__(self, message: str, history: TransientSolution, t: float) -> None:
        super().__init__(message)
        self.history = history
        self.t = t


@dataclass(frozen=True)
class QuasiStaticStepping:
    '''Steady equilibria along the load path: `steps` levels from rest to `t_end`.

    Each level is solved by `newton` (line-searched by default) seeded with the
    previous level's solution, so the walk stays inside Newton's convergence region
    even where a single solve at full load would not. A level that still diverges is
    bisected toward the last converged one, at most `max_bisections` times per
    increment. The problem must have a steady meaning (time order 0): each step is an
    equilibrium, not a transient.

    `solve` returns a `TransientSolution` over the levels reached (step 0 the initial
    state at `t = 0`), each `history[i]` the typed steady solution at its level, so
    the load-deflection curve is read straight off the series. The final state is
    `history[-1]`.
    '''

    steps: int = 10
    t_end: float = 1.0
    newton: NewtonSolve = NewtonSolve(line_search=BacktrackingLineSearch())
    max_bisections: int = 8

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f'steps must be at least 1, got {self.steps}')
        if self.t_end <= 0:
            raise ValueError(f't_end must be positive, got {self.t_end}')

    def solve(self, problem: Problem[S], *, initial: Initial | None = None) -> TransientSolution[S]:
        '''Walk `problem` from rest to `t_end` and return the path.

        `initial` seeds the first step in place of the problem's own `u0`. On failure
        past the bisection budget, raises `SteppingDivergence` carrying the partial path.
        '''
        if 0 not in problem.time_orders:
            raise TypeError(
                f'quasi-static stepping solves a steady equilibrium per level; this '
                f'problem allows time orders {sorted(problem.time_orders)}. Step it '
                'with an integrator instead.'
            )
        seed = (problem.u0 if initial is None
                else problem.resolved.resolve_initial(initial, check=False)[0])
        u = seed.dofs.copy()
        # Step 0 is the state at t = 0, so the fixed DOFs carry that level's values
        # (zero under proportional loading), not the stated problem's lift.
        u[problem.partition.fixed] = self._at(problem, 0.0).fixed_values

        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        min_increment = self.t_end / self.steps / 2 ** self.max_bisections
        pending = [self.t_end * (i + 1) / self.steps for i in reversed(range(self.steps))]
        t = 0.0
        while pending:
            target = pending[-1]
            try:
                u = self.newton.solve(self._at(problem, target),
                                      initial=Initial(NodalField(problem.space, u)))
            except (NewtonDivergence, LineSearchFailure) as failure:
                midpoint = 0.5 * (t + target)
                if midpoint - t < min_increment:
                    raise SteppingDivergence(
                        f'the step from t = {t:g} toward {target:g} still diverged at '
                        f'the smallest increment ({min_increment:g}); the walk stopped '
                        f'at t = {t:g} of {self.t_end:g}. Raise max_bisections or '
                        f'steps, or loosen the Newton tolerance; the partial path is '
                        f'on the exception as `history`.',
                        self._history(problem, t_values, u_values), t,
                    ) from failure
                pending.append(midpoint)
                continue
            pending.pop()
            t = target
            t_values.append(t)
            u_values.append(u.copy())
        return self._history(problem, t_values, u_values)

    @staticmethod
    def _at(problem: Problem, t: float) -> Problem:
        '''The steady problem at level `t`: the values of a time-dependent problem
        taken at `t`, else the proportional loading scaled to `t`.'''
        if problem.is_time_dependent:
            return problem.at(t)
        return problem.with_load_factor(t)

    @staticmethod
    def _history(problem: Problem[S], t_values: list[float],
                 u_values: list[DofVector]) -> TransientSolution[S]:
        return TransientSolution(problem.space, np.asarray(t_values), np.array(u_values),
                                 operator=problem.operator)


@dataclass(frozen=True)
class ArcLengthStepping:
    '''The equilibrium path of `r(u, λ) = r_int(u) − λ f_ext = 0`, folds included.

    Arc-length (Riks/Crisfield) continuation treats λ as an unknown beside the state
    and prescribes instead how far each increment travels through `(u, λ)` space. Each
    increment is a predictor along the tangent direction, scaled to the step length
    `Δs`, then a corrector that restores equilibrium while staying on the sphere
    `‖Δu‖ = Δs` (the cylindrical Crisfield constraint, measured over the free DOFs).
    The corrector is a bordered Newton: one factorization of `K_T` per iteration, two
    back-substitutions, `K_T δu_r = −r` and `K_T δu_t = f_ext`, and `δu = δu_r + δλ δu_t`
    with `δλ` the root of the constraint that keeps the direction of travel. Because
    nothing fixes λ, the path may turn back in it, which is how a limit point is
    traversed and why `QuasiStaticStepping` (load control) stops there.

    `initial_step` is `Δs` as a fraction of the reference response `‖K_T(u_0)^-1 f_ext‖`,
    so it means the same on any structure and any unit system: 0.1 walks a linear
    problem to `λ = 0.1` on its first step. `Δs` then adapts by
    `sqrt(n_target / n_taken)` per increment, clamped by `step_scale_bounds` and never
    growing past `max_step_factor` times its initial value; an increment whose
    corrector fails (no real root of the constraint, or `max_iters` without
    convergence) is halved and restarted from the last converged state, at most
    `max_retries` times before `SteppingDivergence` is raised with the partial path.
    Stopping is `max_steps` increments, or `lambda_max` / `displacement_max` (the
    largest free-DOF magnitude) if given, whichever comes first.

    Force control: λ scales the assembled load only, so a problem with nonzero
    prescribed Dirichlet values is refused rather than having them silently held while
    the load scales. Dead loads, so the tangent needs no load correction. The tangent
    is genuinely indefinite past a critical point and the bordered solve needs it
    unshifted, so no regularization is applied and the problem's backend must handle an
    indefinite operator, which the direct default does.

    `solve` returns a `PathSolution`: the load factors as `t` (step 0 at `λ = 0`), the
    states, and the sign of `det K_T` at each of them, whose flips mark the limit and
    bifurcation points crossed.
    '''

    initial_step: float = 0.1
    max_steps: int = 50
    lambda_max: float | None = None
    displacement_max: float | None = None
    tol: float = 1e-8
    max_iters: int = 20
    n_target: int = 4
    max_retries: int = 8
    step_scale_bounds: tuple[float, float] = (0.5, 2.0)
    max_step_factor: float = 4.0

    def __post_init__(self) -> None:
        if self.initial_step <= 0:
            raise ValueError(f'initial_step must be positive, got {self.initial_step}')
        if self.max_steps < 1:
            raise ValueError(f'max_steps must be at least 1, got {self.max_steps}')
        if self.max_iters < 1:
            raise ValueError(f'max_iters must be at least 1, got {self.max_iters}')
        if self.n_target < 1:
            raise ValueError(f'n_target must be at least 1, got {self.n_target}')
        if self.max_retries < 0:
            raise ValueError(f'max_retries cannot be negative, got {self.max_retries}')

    def solve(self, problem: Problem[S], *, initial: Initial | None = None) -> PathSolution[S]:
        '''Trace `problem`'s equilibrium path from rest and return it.

        `initial` seeds the first predictor in place of the problem's own `u0`. On
        failure past the retry budget, raises `SteppingDivergence` carrying the partial
        `PathSolution`.
        '''
        self._check(problem)
        free = problem.partition.free
        f_ext = problem.load
        seed = (problem.u0 if initial is None
                else problem.resolved.resolve_initial(initial, check=False)[0])
        u = seed.dofs.copy()
        # Step 0 is the unloaded state, and force control means the prescribed values
        # are zero there and everywhere after it.
        u[problem.partition.fixed] = 0.0
        lam = 0.0

        # The factorization at a converged state serves both that increment's predictor
        # and the stability flag recorded for it; the corrector's last iteration hands
        # the next one over, so no state is factored twice.
        system = DiscreteSystem(problem.tangent(u), problem.partition, problem.backend)
        lambdas: list[float] = [0.0]
        states: list[DofVector] = [u.copy()]
        stability: list[int] = [self._stability(system)]

        reference = float(np.linalg.norm(system.solve_homogeneous(f_ext)[free]))
        if reference == 0.0:
            raise ValueError(
                'the reference load produces no displacement, so there is no path to '
                'trace; check the load and the constraints'
            )
        step = self.initial_step * reference
        largest_step = self.max_step_factor * step
        previous: DofVector | None = None

        while len(lambdas) <= self.max_steps:
            for _ in range(self.max_retries + 1):
                increment = self._increment(problem, u, lam, step, system, previous)
                if increment is not None:
                    break
                step *= 0.5
            else:
                raise SteppingDivergence(
                    f'the increment from λ = {lam:g} still failed after {self.max_retries} '
                    f'halvings of the arc length (down to {step:g}); the path stopped at '
                    f'λ = {lam:g} after {len(lambdas) - 1} increments. Loosen tol, raise '
                    f'max_iters or max_retries, or start from a smaller initial_step; the '
                    f'partial path is on the exception as `history`.',
                    self._path(problem, lambdas, states, stability), lam,
                )
            u, lam, du, iterations, system = increment
            lambdas.append(lam)
            states.append(u.copy())
            stability.append(self._stability(system))
            previous = du[free]
            low, high = self.step_scale_bounds
            scale = float(np.clip(np.sqrt(self.n_target / iterations), low, high))
            step = min(largest_step, step * scale)
            if self._target_reached(lam, u, free):
                break
        return self._path(problem, lambdas, states, stability)

    @staticmethod
    def _check(problem: Problem) -> None:
        '''The problems arc-length control has a meaning for.'''
        if 0 not in problem.time_orders:
            raise TypeError(
                f'arc-length continuation traces steady equilibria; this problem allows '
                f'time orders {sorted(problem.time_orders)}. Step it with an integrator '
                'instead.'
            )
        if problem.is_time_dependent:
            raise ValueError(
                'arc-length continuation scales one fixed reference load; a problem with '
                'time-dependent values has a different one per t. Take a snapshot first: '
                'problem.at(t).'
            )
        if np.any(problem.fixed_values != 0.0):
            raise ValueError(
                'arc-length continuation is force control: λ scales the load, and a '
                'nonzero prescribed Dirichlet value would be held at full size while it '
                'does, which is neither the stated problem nor a proportional path. State '
                'the loading as a traction or a source, or step it with '
                'QuasiStaticStepping, which scales both.'
            )
        if not np.any(problem.load):
            raise ValueError(
                'the problem has no load for λ to scale; arc-length continuation needs a '
                'reference load (a source, a traction, or a point load)'
            )

    def _increment(
        self, problem: Problem, u0: DofVector, lam0: float, step: float,
        system: DiscreteSystem, previous: DofVector | None,
    ) -> tuple[DofVector, float, DofVector, int, DiscreteSystem] | None:
        '''One increment of arc length `step` from the converged state `(u0, lam0)`.

        `(u, λ, Δu, iterations, system)` at the new equilibrium, or None when this
        increment failed and the caller should halve `step` and try again. `system` is
        the factorization at the state reached, which the next predictor and the
        stability flag both read. `previous` is the last converged increment over the
        free DOFs, whose direction this one continues.
        '''
        free = problem.partition.free
        f_ext = problem.load

        # Predictor: the tangent direction scaled to the arc length, signed to keep
        # going the way the path was going rather than doubling back on itself.
        du_t = system.solve_homogeneous(f_ext)
        tangent_norm = float(np.linalg.norm(du_t[free]))
        if tangent_norm == 0.0:
            return None
        forward = previous is None or float(previous @ du_t[free]) >= 0.0
        dlam = (1.0 if forward else -1.0) * step / tangent_norm
        du = dlam * du_t
        u = u0 + du
        lam = lam0 + dlam

        for iteration in range(1, self.max_iters + 1):
            # One assembly pass gives both. `residual_and_tangent` reports the residual
            # at full load, r_int(u) - f_ext; the path's is r_int(u) - λ f_ext, which is
            # that plus the (1 - λ) of the load it already subtracted.
            at_full_load, tangent = problem.residual_and_tangent(u)
            residual = at_full_load + (1.0 - lam) * f_ext
            try:
                discrete = DiscreteSystem(tangent, problem.partition, problem.backend)
                du_r = discrete.solve_homogeneous(-residual)
                du_t = discrete.solve_homogeneous(f_ext)
            except RuntimeError:
                # The tangent is singular at a limit point, and the backend says so
                # rather than answering. A shorter increment lands beside it instead.
                return None
            root = self._constraint_root(du[free], du_r[free], du_t[free], step)
            if root is None:
                return None
            correction = du_r + root * du_t
            # Checked before it is applied, as NewtonSolve does, so a sub-tolerance
            # correction is never added: on a linear problem the predictor is already
            # exact and the path is traced to round-off.
            if (float(np.linalg.norm(correction)) < self.tol * max(1.0, float(np.linalg.norm(u)))
                    and abs(root) < self.tol * max(1.0, abs(lam))):
                return u, lam, du, iteration, discrete
            du = du + correction
            u = u + correction
            lam = lam + root
        return None

    @staticmethod
    def _constraint_root(
        du: DofVector, du_r: DofVector, du_t: DofVector, step: float,
    ) -> float | None:
        '''δλ from `‖Δu + δu_r + δλ δu_t‖ = Δs`, every vector already restricted to the
        free DOFs.

        A quadratic in δλ; of its two roots, the one whose resulting increment has the
        larger cosine with the incoming `Δu` continues the path rather than reversing
        along it. None when there is no real root, which means this arc length overshoots
        the path's curvature and the increment must be retried shorter.
        '''
        base = du + du_r
        a = float(du_t @ du_t)
        if a == 0.0:
            return None
        b = 2.0 * float(base @ du_t)
        c = float(base @ base) - step * step
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        offset = np.sqrt(discriminant)
        incoming = float(np.linalg.norm(du))

        def cosine(root: float) -> float:
            candidate = base + root * du_t
            scale = float(np.linalg.norm(candidate)) * incoming
            return float(candidate @ du) / scale if scale > 0.0 else 0.0

        return max(((-b + offset) / (2.0 * a), (-b - offset) / (2.0 * a)), key=cosine)

    def _target_reached(self, lam: float, u: DofVector, free: DofIndices) -> bool:
        '''Whether a stopping target has been crossed at the state just recorded.'''
        if self.lambda_max is not None and lam >= self.lambda_max:
            return True
        return (self.displacement_max is not None
                and float(np.abs(u[free]).max()) >= self.displacement_max)

    @staticmethod
    def _stability(system: DiscreteSystem) -> int:
        '''The sign of `det K_T,ff` at a converged state, 0 where the backend cannot
        say (it formed no factorization to read).'''
        sign = det_sign(system.factorization)
        return 0 if sign is None else sign

    @staticmethod
    def _path(problem: Problem[S], lambdas: list[float], states: list[DofVector],
              stability: list[int]) -> PathSolution[S]:
        return PathSolution(problem.space, np.asarray(lambdas), np.array(states),
                            np.asarray(stability, dtype=int), operator=problem.operator)
