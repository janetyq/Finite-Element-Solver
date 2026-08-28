"""The assembled linear system: a matrix, its Dirichlet partition, and a solve.

`DiscreteSystem` owns the operator `A` and the DOF partition, eliminates the
constrained DOFs rather than penalising them, and factors the free-free block once
through a `Backend`, so repeated solves with different right-hand sides (a
time-stepper, a Newton loop with a fixed tangent) reuse the factorization.

The `backend` is `DirectBackend` (sparse LU, the default) or `IterativeBackend`
(AMG-preconditioned CG, opt-in for SPD systems); it owns the algebra and its storage
format.
"""
import numpy as np

from fem.algebra.backends import Backend, DirectBackend
from fem.typing import Constraints, DofVector, FloatArray, Operator


class DiscreteSystem:
    '''A x = b with the Dirichlet DOFs eliminated and the free block factored once.'''

    def __init__(
        self,
        A: Operator,
        constraints: Constraints,
        backend: Backend | None = None,
    ) -> None:
        free, fixed, fixed_values = constraints
        self.n_dofs = A.shape[0]
        self.free = np.asarray(free, dtype=int)
        self.fixed = np.asarray(fixed, dtype=int)
        self.fixed_values = np.asarray(fixed_values, dtype=float)

        # The free-free block is what actually gets solved; the free-fixed block
        # moves the known Dirichlet values to the right-hand side. The backend
        # factors (or preconditions) the former now, so each solve() reuses that
        # setup; the default direct backend is a sparse LU.
        backend = backend if backend is not None else DirectBackend()
        self._free_fixed = A[np.ix_(self.free, self.fixed)]
        self._factorization = backend.prepare(A[np.ix_(self.free, self.free)])

    def solve(self, b: DofVector, fixed_values: FloatArray | None = None) -> DofVector:
        '''Solve for x given a right-hand side b, reusing the factorization.

        `fixed_values` replaces the Dirichlet data for this solve (a time-stepper
        whose prescribed values change per step); the default is the system's own.
        '''
        values = self.fixed_values if fixed_values is None else np.asarray(fixed_values, dtype=float)
        x = np.zeros(self.n_dofs)
        x[self.fixed] = values
        b_free = b[self.free] - self._free_fixed @ values
        x[self.free] = self._factorization.solve(b_free)
        return x

    def solve_homogeneous(self, b: DofVector) -> DofVector:
        '''Solve with the fixed DOFs pinned to zero, reusing the factorization.

        The Dirichlet data is zero rather than the operator's own `fixed_values`, so the
        free-fixed lifting term drops out and only the free block is solved. This is the
        solve an adjoint problem needs: the adjoint field vanishes on the supported DOFs,
        whatever displacement the forward problem prescribes there. Reuses the same
        factored free block as `solve`, so it costs one back-substitution, not a refactor.
        '''
        x = np.zeros(self.n_dofs)
        x[self.free] = self._factorization.solve(b[self.free])
        return x
