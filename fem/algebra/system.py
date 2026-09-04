"""The assembled linear system: a matrix, its Dirichlet partition, and a solve.

`Partition` is which DOFs are free and which are fixed: structure, decided once when
the conditions are resolved on a space and shared by every snapshot of a problem. The
values prescribed at the fixed DOFs are data of a different lifetime (they move with
time or a load factor, as the load vector does) and are never part of it.

`DiscreteSystem` owns the operator `A` and its `Partition`, eliminates the constrained
DOFs rather than penalising them, and factors the free-free block once through a
`Backend`, so repeated solves with different right-hand sides and Dirichlet values (a
time-stepper, a Newton loop with a fixed tangent, the snapshots of one problem) reuse
the factorization. The prescribed values are given per solve, not held: the system is
the operator and the partition, which is what the factorization depends on, and
nothing more.

The `backend` is `DirectBackend` (sparse LU, the default) or `IterativeBackend`
(AMG-preconditioned CG, opt-in for SPD systems); it owns the algebra and its storage
format. A `LinearProblem` builds and holds one of these (`problem.system`) over its
constant tangent; the integrators build their own over the step operator.
"""
from dataclasses import dataclass

import numpy as np

from fem.algebra.backends import Backend, DirectBackend
from fem.typing import DofIndices, DofVector, FloatArray, Operator


@dataclass(frozen=True, eq=False)
class Partition:
    '''The DOFs of a system split into `free` (solved for) and `fixed` (prescribed).

    `free` and `fixed` are index arrays that together cover `range(n_dofs)` once; the
    prescribed values are not here, since they change where the partition does not.
    `eliminate` is the one thing done with a partition: the blocks of an operator a
    solve reads, the free-free block that is factored and the free-fixed block that
    moves the prescribed values to the right-hand side.
    '''
    free: DofIndices
    fixed: DofIndices
    n_dofs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, 'free', np.asarray(self.free, dtype=int))
        object.__setattr__(self, 'fixed', np.asarray(self.fixed, dtype=int))
        if len(self.free) + len(self.fixed) != self.n_dofs:
            raise ValueError(
                f'{len(self.free)} free and {len(self.fixed)} fixed DOFs do not partition '
                f'{self.n_dofs}'
            )

    def __eq__(self, other: object) -> bool:
        # By content: the generated dataclass comparison would compare the index
        # arrays elementwise and fail to reduce them to one truth value.
        if not isinstance(other, Partition):
            return NotImplemented
        return (self.n_dofs == other.n_dofs and np.array_equal(self.free, other.free)
                and np.array_equal(self.fixed, other.fixed))

    __hash__ = None  # type: ignore[assignment]

    @property
    def n_free(self) -> int:
        return len(self.free)

    def eliminate(self, A: Operator) -> tuple[Operator, Operator]:
        '''`(A_ff, A_fc)`: the free-free block of `A` and its free-fixed block.'''
        return A[np.ix_(self.free, self.free)], A[np.ix_(self.free, self.fixed)]


class DiscreteSystem:
    '''A x = b with the Dirichlet DOFs eliminated and the free block factored once.'''

    def __init__(
        self,
        A: Operator,
        partition: Partition,
        backend: Backend | None = None,
    ) -> None:
        if A.shape[0] != partition.n_dofs:
            raise ValueError(f'a {A.shape[0]}-DOF operator against a partition of {partition.n_dofs}')
        self.partition = partition

        # The free-free block is what actually gets solved; the free-fixed block
        # moves the known Dirichlet values to the right-hand side. The backend
        # factors (or preconditions) the former now, so each solve() reuses that
        # setup; the default direct backend is a sparse LU.
        backend = backend if backend is not None else DirectBackend()
        free_free, self._free_fixed = partition.eliminate(A)
        self._factorization = backend.prepare(free_free)

    def solve(self, b: DofVector, fixed_values: FloatArray) -> DofVector:
        '''Solve for x given a right-hand side b and the values prescribed at the fixed
        DOFs, reusing the factorization.'''
        free, fixed = self.partition.free, self.partition.fixed
        values = np.asarray(fixed_values, dtype=float)
        x = np.zeros(self.partition.n_dofs)
        x[fixed] = values
        b_free = b[free] - self._free_fixed @ values
        x[free] = self._factorization.solve(b_free)
        return x

    def solve_homogeneous(self, b: DofVector) -> DofVector:
        '''Solve with the fixed DOFs pinned to zero, reusing the factorization.

        The free-fixed lifting term drops out and only the free block is solved. This
        is the solve an adjoint problem needs (the adjoint field vanishes on the
        supported DOFs, whatever displacement the forward problem prescribes there) and
        the one a Newton increment needs. It costs one back-substitution, not a refactor.
        '''
        free = self.partition.free
        x = np.zeros(self.partition.n_dofs)
        x[free] = self._factorization.solve(b[free])
        return x
