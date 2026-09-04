"""The assembled linear system: a matrix, its Dirichlet partition, and a solve.

`DiscreteSystem` owns the operator `A` and the DOF partition, eliminates the
constrained DOFs rather than penalising them, and factors the free-free block once
through a `Backend`, so repeated solves with different right-hand sides and Dirichlet
values (a time-stepper, a Newton loop with a fixed tangent, the snapshots of one
problem) reuse the factorization. The prescribed values are given per solve, not
held: the system is the operator and the partition, which is what the factorization
depends on, and nothing more.

The `backend` is `DirectBackend` (sparse LU, the default) or `IterativeBackend`
(AMG-preconditioned CG, opt-in for SPD systems); it owns the algebra and its storage
format. A `LinearProblem` builds and holds one of these (`problem.system`) over its
constant tangent; the integrators build their own over the step operator.
"""
import numpy as np

from fem.algebra.backends import Backend, DirectBackend
from fem.typing import DofIndices, DofVector, FloatArray, Operator


class DiscreteSystem:
    '''A x = b with the Dirichlet DOFs eliminated and the free block factored once.'''

    def __init__(
        self,
        A: Operator,
        free: DofIndices,
        fixed: DofIndices,
        backend: Backend | None = None,
    ) -> None:
        self.n_dofs = A.shape[0]
        self.free = np.asarray(free, dtype=int)
        self.fixed = np.asarray(fixed, dtype=int)

        # The free-free block is what actually gets solved; the free-fixed block
        # moves the known Dirichlet values to the right-hand side. The backend
        # factors (or preconditions) the former now, so each solve() reuses that
        # setup; the default direct backend is a sparse LU.
        backend = backend if backend is not None else DirectBackend()
        self._free_fixed = A[np.ix_(self.free, self.fixed)]
        self._factorization = backend.prepare(A[np.ix_(self.free, self.free)])

    def solve(self, b: DofVector, fixed_values: FloatArray) -> DofVector:
        '''Solve for x given a right-hand side b and the values prescribed at the fixed
        DOFs, reusing the factorization.'''
        values = np.asarray(fixed_values, dtype=float)
        x = np.zeros(self.n_dofs)
        x[self.fixed] = values
        b_free = b[self.free] - self._free_fixed @ values
        x[self.free] = self._factorization.solve(b_free)
        return x

    def solve_homogeneous(self, b: DofVector) -> DofVector:
        '''Solve with the fixed DOFs pinned to zero, reusing the factorization.

        The free-fixed lifting term drops out and only the free block is solved. This
        is the solve an adjoint problem needs (the adjoint field vanishes on the
        supported DOFs, whatever displacement the forward problem prescribes there) and
        the one a Newton increment needs. It costs one back-substitution, not a refactor.
        '''
        x = np.zeros(self.n_dofs)
        x[self.free] = self._factorization.solve(b[self.free])
        return x
