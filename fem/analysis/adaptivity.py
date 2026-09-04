"""Adaptive mesh refinement: a driver that refines where the error is largest.

The outer loop re-solves on progressively finer meshes. It owns the mesh, states the
problem on it through `problem_for`, solves with a strategy, reads an error estimate,
refines the marked elements, and repeats. Linear and energy problems alike: the
builder and the strategy are the caller's.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.analysis.estimators import ErrorEstimator
from fem.algebra.solve import SolveStrategy
from fem.post.solution import FieldSolution

S = TypeVar('S', bound=FieldSolution)   # the solution each round packages

if TYPE_CHECKING:
    from collections.abc import Callable

    from fem.problem import Problem
    from fem.typing import ElementValues

logger = logging.getLogger(__name__)


class AdaptiveRefinement(Generic[S]):
    '''Refine where the error estimate is largest, re-solving on each new mesh.

    `problem_for(mesh)` states the problem on any mesh (`equation.problem(mesh,
    bc)`); its boundary conditions must be geometric, since
    they are resolved afresh on every mesh, and the problem it builds carries the
    backend each round solves with. `strategy` None is `default_strategy` for
    each round's problem. `estimator` is an `ErrorEstimator` or a bare callable of
    `(problem, solution)`. After `run`, `mesh`, `problem`, and `solution` are the
    final round's.
    '''

    def __init__(
        self,
        mesh: Mesh,
        problem_for: Callable[[Mesh], Problem[S]],
        estimator: ErrorEstimator | Callable[[Problem, FieldSolution], ElementValues],
        strategy: SolveStrategy | None = None,
        max_triangles: int = 1000,
        max_iters: int = 20,
        refine_fraction: float = 0.9,
    ) -> None:
        self.mesh = mesh
        self.problem_for = problem_for
        self.strategy = strategy
        self._estimate = estimator.estimate if isinstance(estimator, ErrorEstimator) else estimator
        self.max_triangles = max_triangles
        self.max_iters = max_iters
        self.refine_fraction = refine_fraction
        self.problem: Problem[S] | None = None
        self.solution: S | None = None

    def _solve(self) -> S:
        assert self.problem is not None
        self.solution = self.problem.solve(strategy=self.strategy)
        return self.solution

    def run(self) -> S:
        '''Refine and re-solve until a budget is hit; return the final solution.

        Elements whose estimate is within `refine_fraction` of the largest are
        refined each round.
        '''
        self.problem = self.problem_for(self.mesh)
        self.problem.conditions.check_remeshable()
        solution = self._solve()

        # RedGreenRefiner is stateful (it tracks the current mesh and returns the
        # refined one), so it is built once and kept in step with the loop's mesh.
        refiner = RedGreenRefiner(self.mesh)
        for _ in range(self.max_iters):
            if len(self.mesh.elements) >= self.max_triangles:
                break

            residuals = np.asarray(self._estimate(self.problem, solution), dtype=float)
            if len(residuals) != len(self.mesh.elements):
                raise ValueError(
                    f'estimator returned {len(residuals)} values for '
                    f'{len(self.mesh.elements)} elements'
                )
            refine_idxs = np.flatnonzero(residuals >= self.refine_fraction * residuals.max())
            if len(refine_idxs) == 0:
                break

            self.mesh = refiner.refine([int(i) for i in refine_idxs])
            self.problem = self.problem_for(self.mesh)
            solution = self._solve()

        return solution
