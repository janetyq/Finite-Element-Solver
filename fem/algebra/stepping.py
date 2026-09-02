"""Quasi-static continuation: walk a nonlinear problem up its load path.

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

This is load control: it cannot pass a limit point (a fold, where equilibrium past
the peak exists only at lower loads — a snap-through). Tracing past one needs
arc-length control of the same loop, which is the planned sequel.
"""
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from fem.algebra.backends import Backend
from fem.algebra.solve import (
    BacktrackingLineSearch, LineSearchFailure, NewtonDivergence, NewtonSolve,
)
from fem.conditions import Initial
from fem.field import NodalField
from fem.post.solution import FieldSolution, TransientSolution
from fem.problem import Problem
from fem.typing import DofVector

S = TypeVar('S', bound=FieldSolution)


class SteppingDivergence(RuntimeError):
    '''`QuasiStaticStepping` could not converge a step even at its smallest increment.
    `history` is the path walked so far (every converged level, as the strategy would
    have returned it) and `t` the last converged level, so a caller can read how far
    the walk got, plot it, or continue with a different strategy from its end.'''

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

    def solve(self, problem: Problem[S], *, initial: Initial | None = None,
              backend: Backend | None = None) -> TransientSolution[S]:
        '''Walk `problem` from rest to `t_end` and return the path.

        `initial` seeds the first step in place of the problem's own `u0`; `backend`
        solves each Newton tangent. On failure past the bisection budget, raises
        `SteppingDivergence` carrying the partial path.
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
        _, fixed0, values0 = self._at(problem, 0.0).constraints
        u[fixed0] = values0

        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        min_increment = self.t_end / self.steps / 2 ** self.max_bisections
        pending = [self.t_end * (i + 1) / self.steps for i in reversed(range(self.steps))]
        t = 0.0
        while pending:
            target = pending[-1]
            try:
                u = self.newton.solve(self._at(problem, target),
                                      initial=Initial(NodalField(problem.space, u)),
                                      backend=backend)
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
