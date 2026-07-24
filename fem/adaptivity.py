"""Adaptive mesh refinement: a driver that refines where the error is largest.

The outer loop that re-solves on progressively finer meshes, sitting *above* the
Solver it drives rather than inside it -- the same shape as TopologyOptimizer,
where the old version was a method on the thing it drove. It owns a Solver, reads
an error estimate each round, refines the marked elements, and advances the solver
onto the new mesh via `Solver.remesh` (which rebuilds the space and re-resolves the
boundary conditions from their mesh-independent spec). Holding the Solver is what
lets the driver call `check_remeshable` on the BC spec up front -- a bare
problem-factory would hide it.
"""
import logging
from collections.abc import Callable

import numpy as np

from fem.mesh.refinement import RedGreenRefiner
from fem.solution import Solution
from fem.solver import Solver
from fem.typing import ElementField

logger = logging.getLogger(__name__)


class AdaptiveRefinement:
    '''Refine where the error estimate is largest, re-solving on each new mesh.'''

    def __init__(
        self,
        solver: Solver,
        estimator: Callable[[Solver], ElementField],
        max_triangles: int = 1000,
        max_iters: int = 20,
        refine_fraction: float = 0.9,
    ) -> None:
        self.solver = solver
        # estimator(solver) -> per-element error. It takes the solver rather than a
        # stored array because the estimate must be recomputed every round: once
        # elements are split, the previous array is both stale and the wrong length,
        # so indexing it selects unrelated elements -- the "bug somewhere" the old
        # in-Solver loop used to carry.
        self.estimator = estimator
        self.max_triangles = max_triangles
        self.max_iters = max_iters
        self.refine_fraction = refine_fraction

    def run(self) -> Solution:
        '''Refine and re-solve until a budget is hit; return the final solution.

        Elements whose estimate is within `refine_fraction` of the largest are
        refined each round.
        '''
        self.solver.boundary_conditions.check_remeshable()

        # RedGreenRefiner is stateful -- it tracks the current mesh and returns the
        # refined one -- so it is built once and kept in step with the solver's mesh.
        refiner = RedGreenRefiner(self.solver.mesh)
        solution = self.solver.solve()  # solve the initial mesh; the estimator may read solver.solution
        for _ in range(self.max_iters):
            if len(self.solver.mesh.elements) >= self.max_triangles:
                break

            residuals = np.asarray(self.estimator(self.solver), dtype=float)
            if len(residuals) != len(self.solver.mesh.elements):
                raise ValueError(
                    f'estimator returned {len(residuals)} values for '
                    f'{len(self.solver.mesh.elements)} elements'
                )
            refine_idxs = np.flatnonzero(residuals >= self.refine_fraction * residuals.max())
            if len(refine_idxs) == 0:
                break

            self.solver.remesh(refiner.refine([int(i) for i in refine_idxs]))
            solution = self.solver.solve()

        return solution
