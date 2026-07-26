"""The assembled linear system: a matrix, its Dirichlet partition, and a solve.

`DiscreteSystem` is the seam between assembly (which produces a matrix) and algebra
(which solves it). It owns the operator `A` and the DOF partition, eliminates the
constrained DOFs rather than penalising them, and -- the reason it is an object
rather than a function -- factors the free-free block *once* (through a `LinearAlgebra` backend), so repeated
solves with different right-hand sides reuse that factorization. A time-stepper whose
LHS is constant across steps, or a Newton loop with a fixed tangent, pays the setup
only once and a cheap solve per subsequent right-hand side.

*How* the free-free block is solved is the injected `backend`: `DirectBackend` (sparse
LU, the default, robust for any operator) or `IterativeBackend` (AMG-preconditioned CG,
opt-in for SPD systems). This class only eliminates the Dirichlet DOFs and hands the
free-free block off; the backend owns the algebra, including whatever storage format it
wants (`splu` needs CSC, CG needs CSR), so neither leaks in here.
"""
import numpy as np

from fem.linalg import DirectBackend, LinearAlgebra
from fem.typing import Constraints, DofVector, Operator


class DiscreteSystem:
    '''A x = b with the Dirichlet DOFs eliminated and the free block factored once.'''

    def __init__(
        self,
        A: Operator,
        constraints: Constraints,
        backend: LinearAlgebra | None = None,
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
        self._solver = backend.factor(A[np.ix_(self.free, self.free)])

    def solve(self, b: DofVector) -> DofVector:
        '''Solve for x given a right-hand side b, reusing the factorization.'''
        x = np.zeros(self.n_dofs)
        x[self.fixed] = self.fixed_values
        b_free = b[self.free] - self._free_fixed @ self.fixed_values
        x[self.free] = self._solver.solve(b_free)
        return x
