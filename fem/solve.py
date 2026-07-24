"""Solve strategies: they consume a `Problem` and return the DOF vector.

Every strategy sits on the one algebra atom, `DiscreteSystem` (matrix + Dirichlet
partition + factor-once solve), and knows nothing about which PDE produced the
`Problem`. `LinearSolve` assembles once and solves once; `NewtonSolve` iterates.
The two are one engine: a `LinearProblem` has a constant tangent and an affine
residual, so `NewtonSolve` reaches its solution in a single step from any seed --
`LinearSolve` is that step done directly, skipping the residual evaluation.
"""
from typing import Protocol

import numpy as np

from fem.problem import Problem
from fem.system import DiscreteSystem
from fem.typing import DofVector


class SolveStrategy(Protocol):
    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector: ...


class LinearSolve:
    '''Assemble once, solve once: for a `Problem` with a state-independent tangent.'''

    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector:
        return DiscreteSystem(problem.tangent(None), problem.constraints).solve(problem.load)


class NewtonSolve:
    '''Newton's method on r(u) = 0, re-factoring the tangent each iteration.

    The increment is pinned to zero at the fixed DOFs -- the seed already carries
    their Dirichlet values -- and `DiscreteSystem` eliminates them, so the tangent
    needs no special-casing. Convergence is checked before the step is applied, so a
    sub-tolerance increment is never added: on a `LinearProblem` the first step is
    exact and the second is zero, so the exact answer is reached in one applied step.
    '''

    def __init__(self, max_iters: int = 100, tol: float = 1e-6) -> None:
        self.max_iters = max_iters
        self.tol = tol

    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector:
        free, fixed, fixed_values = problem.constraints
        u = np.zeros(problem.space.n_dofs) if u0 is None else np.asarray(u0, dtype=float).copy()
        u[fixed] = fixed_values

        step_constraints = (free, fixed, np.zeros(len(fixed)))
        for _ in range(self.max_iters):
            system = DiscreteSystem(problem.tangent(u), step_constraints)
            step = system.solve(-problem.residual(u))
            if np.linalg.norm(step) < self.tol:
                break
            u = u + step
        return u
