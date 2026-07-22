"""The assembled linear system: a matrix, its Dirichlet partition, and a solve.

`DiscreteSystem` is the seam between assembly (which produces a matrix) and algebra
(which solves it). It owns the operator `A` and the DOF partition, eliminates the
constrained DOFs rather than penalising them, and -- the reason it is an object
rather than a function -- factors the free-free block *once*, so repeated solves
with different right-hand sides reuse the factorization. A time-stepper whose LHS
is constant across steps, or a Newton loop with a fixed tangent, pays the O(n^3)
factorization only once and O(n^2) per subsequent solve.

Sparse factorization via `scipy.sparse.linalg.splu`; `csc_array` accepts a dense or
sparse free-free block interchangeably, so this class is agnostic to how the operator
was assembled. It is the single place the dense -> sparse migration touches the solve.
"""
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu

from fem.typing import Constraints, DofVector, Operator


class DiscreteSystem:
    '''A x = b with the Dirichlet DOFs eliminated and the free block factored once.'''

    def __init__(self, A: Operator, constraints: Constraints) -> None:
        free, fixed, fixed_values = constraints
        self.n_dofs = A.shape[0]
        self.free = np.asarray(free, dtype=int)
        self.fixed = np.asarray(fixed, dtype=int)
        self.fixed_values = np.asarray(fixed_values, dtype=float)

        # The free-free block is what actually gets solved; the free-fixed block
        # moves the known Dirichlet values to the right-hand side. LU-factor the
        # former now (as CSC, which splu wants) so each solve() is a cheap
        # triangular back-substitution reusing the factorization.
        self._free_fixed = A[np.ix_(self.free, self.fixed)]
        self._lu = splu(csc_array(A[np.ix_(self.free, self.free)]))

    def solve(self, b: DofVector) -> DofVector:
        '''Solve for x given a right-hand side b, reusing the factorization.'''
        x = np.zeros(self.n_dofs)
        x[self.fixed] = self.fixed_values
        b_free = b[self.free] - self._free_fixed @ self.fixed_values
        x[self.free] = self._lu.solve(b_free)
        return x
