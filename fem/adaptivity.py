"""Adaptive mesh refinement: a driver that refines where the error is largest.

The outer loop re-solves on progressively finer meshes. It owns a solver, reads an
error estimate each round, refines the marked elements, and advances the solver onto
the new mesh through `remesh`. The solver is a `RefinableSolver`, so this drives the
linear and nonlinear facades alike.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np

from fem.boundary import BoundaryConditions
from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.estimators import ErrorEstimator
from fem.solution import FieldSolution
from fem.space import FunctionSpace

if TYPE_CHECKING:
    from collections.abc import Callable

    from fem.typing import ElementField

logger = logging.getLogger(__name__)


class RefinableSolver(Protocol):
    '''What this driver needs from the solver it advances across meshes: a mesh, a
    mesh-independent BC spec, `remesh` to rebuild derived state, and `solve`. Both
    `Solver` and `EnergySolver` satisfy it.'''
    mesh: Mesh
    space: FunctionSpace
    boundary_conditions: BoundaryConditions
    solution: FieldSolution | None

    def remesh(self, mesh: Mesh) -> None: ...

    def solve(self) -> FieldSolution: ...


class AdaptiveRefinement:
    '''Refine where the error estimate is largest, re-solving on each new mesh.'''

    def __init__(
        self,
        solver: RefinableSolver,
        estimator: ErrorEstimator | Callable[[RefinableSolver], ElementField],
        max_triangles: int = 1000,
        max_iters: int = 20,
        refine_fraction: float = 0.9,
    ) -> None:
        self.solver = solver
        # An `ErrorEstimator` or a bare callable of the solver. It takes the solver
        # rather than a stored array because the estimate is recomputed every round.
        self.estimator = estimator
        self._estimate = estimator.estimate if isinstance(estimator, ErrorEstimator) else estimator
        self.max_triangles = max_triangles
        self.max_iters = max_iters
        self.refine_fraction = refine_fraction

    def run(self) -> FieldSolution:
        '''Refine and re-solve until a budget is hit; return the final solution.

        Elements whose estimate is within `refine_fraction` of the largest are
        refined each round.
        '''
        self.solver.boundary_conditions.check_remeshable()

        # RedGreenRefiner is stateful (it tracks the current mesh and returns the
        # refined one), so it is built once and kept in step with the solver's mesh.
        refiner = RedGreenRefiner(self.solver.mesh)
        solution = self.solver.solve()  # solve the initial mesh; the estimator may read solver.solution
        for _ in range(self.max_iters):
            if len(self.solver.mesh.elements) >= self.max_triangles:
                break

            residuals = np.asarray(self._estimate(self.solver), dtype=float)
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
