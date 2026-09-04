"""How the free-free block gets solved: the linear-algebra backend.

`DiscreteSystem` owns the Dirichlet elimination but not how the remaining free-free
block is solved. That choice is a `Backend`: a `SolveStrategy` picks linear vs.
Newton, a `Backend` picks direct vs. iterative, and they compose.

What a caller touches, versus what is plumbing:

    Use:       DirectBackend() | IterativeBackend() | MinresBackend()  <- the public choice
    Extend:    Backend, Factorization                  <- implement these to add one
    Internal:  _CGSolver, _MinresSolver, _pyamg_csr   <- never named outside here

`DirectBackend` LU-factors the block (`splu`) and back-substitutes: robust for any
nonsingular system, indefinite ones included, but its fill-in on a 3D mesh grows
super-linearly and caps the reachable resolution. `IterativeBackend` runs
preconditioned conjugate gradients with an algebraic-multigrid preconditioner
(`pyamg`); CG is SPD-only, so it is opt-in (Poisson, small-strain elasticity, mass,
the time-stepping operators), and on a large 3D system it is O(n) where the direct
factorization is not.

`MinresBackend` is the iterative path for symmetric indefinite systems, which CG
cannot take: a harmonic operator `K - w^2 M` above the first natural frequency, or a
Newton tangent away from a convex minimum.

Both `prepare` an operator into a `Factorization`, an object that has factored or
preconditioned one matrix and solves it against many right-hand sides.
`DiscreteSystem` builds one per operator and reuses it, so a time-stepper or Newton
loop with a constant operator pays the setup once. The `Backend` is the recipe and
the `Factorization` the bound solver, since the matrix actually solved (the eliminated
free-free block) is built inside `DiscreteSystem`.

The AMG preconditioner is `pyamg`'s smoothed aggregation (see BACKLOG.md for a
geometric alternative).
"""
from typing import Protocol

import numpy as np
import pyamg
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import cg, minres, splu

from fem.typing import DofVector, FloatArray, Operator

__all__ = [
    'Backend', 'Factorization', 'DirectBackend', 'IterativeBackend', 'MinresBackend',
]


# -- the two contracts: implement both to add a backend ------------------------

class Factorization(Protocol):
    '''A matrix factored or preconditioned once, solving many right-hand sides.'''

    def solve(self, b: DofVector) -> DofVector: ...


class Backend(Protocol):
    '''The recipe for turning an assembled operator into a reusable `Factorization`.'''

    def prepare(self, A: Operator) -> Factorization: ...


# -- the backends a caller picks from ------------------------------------------

class DirectBackend:
    '''Sparse LU factorization via `splu`. The default: robust for any operator.'''

    def prepare(self, A: Operator) -> Factorization:
        # splu wants CSC; the SuperLU it returns already satisfies Factorization
        # (its .solve reuses the factorization), so no wrapper is needed.
        # The ordering matters more than anything else about the factorization: scipy's
        # default COLAMD is a column ordering for unsymmetric LU, and on a symmetric FEM
        # matrix it fills 2-3x more than minimum degree on the structure of A + A^T
        # (13.1M against 4.7M L+U entries on a 2D P1 Poisson block of 89k DOFs), which is
        # a 2-4x factorization and back-substitution time across 2D P1, 2D P2, and 3D.
        return splu(csc_array(A), permc_spec='MMD_AT_PLUS_A')


class IterativeBackend:
    '''AMG-preconditioned CG for SPD systems. Opt-in per solve; the default is direct.

    `near_null_space` is the AMG near-kernel `B` (shape `(n_free, n_modes)`): the
    low-energy modes the smoother cannot damp, which the coarse levels must
    represent to converge well. The constant (pyamg's default) suffices for scalar
    Poisson; vector elasticity wants the rigid-body modes, restricted to the free
    DOFs so it aligns with the free-free block this backend is handed.
    '''

    def __init__(
        self,
        rtol: float = 1e-10,
        maxiter: int | None = None,
        near_null_space: FloatArray | None = None,
    ) -> None:
        self.rtol = rtol
        self.maxiter = maxiter
        self.near_null_space = near_null_space

    def with_near_null_space(self, B: FloatArray) -> 'IterativeBackend':
        '''A copy carrying near-kernel `B`, leaving this instance's config intact.

        Config is immutable, so the elasticity-aware layer derives a mode-carrying
        backend rather than mutating the one a caller handed in.
        '''
        return IterativeBackend(self.rtol, self.maxiter, B)

    def prepare(self, A: Operator) -> Factorization:
        A_csr = _pyamg_csr(A)
        ml = pyamg.smoothed_aggregation_solver(A_csr, B=self.near_null_space)
        return _CGSolver(A_csr, ml.aspreconditioner(), self.rtol, self.maxiter)


class MinresBackend:
    '''MINRES for symmetric indefinite systems: the iterative path CG cannot take.

    CG needs a positive-definite operator; MINRES needs only symmetry. This is the
    iterative backend for the operators `DirectBackend` factors but `IterativeBackend`
    (CG) rejects: a harmonic operator `K - w^2 M` above the first natural frequency, a
    Newton tangent away from a convex minimum, or a saddle-point block. It gives the O(n)
    iterative path a symmetric-indefinite entry rather than forcing a direct factorization
    at 3D scale.

    Unpreconditioned by default. A useful preconditioner for an indefinite system is
    problem-specific (block/Schur for a saddle-point system; see BACKLOG.md), so it is
    injected rather than assumed. MINRES requires any preconditioner be SPD, which a
    block-diagonal one is; a generic ILU is not, and breaks the method's short recurrence.
    '''

    def __init__(
        self,
        rtol: float = 1e-10,
        maxiter: int | None = None,
        preconditioner: Operator | None = None,
    ) -> None:
        self.rtol = rtol
        self.maxiter = maxiter
        self.preconditioner = preconditioner

    def prepare(self, A: Operator) -> Factorization:
        return _MinresSolver(csr_array(A), self.preconditioner, self.rtol, self.maxiter)


# -- internal plumbing: not part of the public surface -------------------------

class _CGSolver:
    '''Preconditioned CG bound to one operator and its AMG preconditioner.

    Holds the AMG hierarchy built by `IterativeBackend.prepare`, so each `solve`
    reuses it rather than re-coarsening. Raises when CG reports non-convergence or an
    illegal input through a nonzero `info`.
    '''

    def __init__(self, A: csr_array, preconditioner, rtol: float, maxiter: int | None) -> None:
        self._A = A
        self._M = preconditioner
        self._rtol = rtol
        self._maxiter = maxiter

    def solve(self, b: DofVector) -> DofVector:
        x, info = cg(self._A, b, rtol=self._rtol, atol=0.0, maxiter=self._maxiter, M=self._M)
        if info != 0:
            reason = (
                f'did not converge in {info} iterations'
                if info > 0 else f'illegal input or breakdown (info={info})'
            )
            raise RuntimeError(
                f'CG failed to solve the free-free block: {reason}. The operator must '
                'be symmetric positive-definite for CG; use DirectBackend for indefinite '
                'systems (Newton tangents, energy Hessians away from a minimum).'
            )
        return np.asarray(x)


class _MinresSolver:
    '''MINRES bound to one symmetric (possibly indefinite) operator.

    Holds the operator and an optional SPD preconditioner. Raises on non-convergence,
    as `_CGSolver` does.
    '''

    def __init__(
        self, A: csr_array, preconditioner: Operator | None, rtol: float, maxiter: int | None,
    ) -> None:
        self._A = A
        self._M = preconditioner
        self._rtol = rtol
        self._maxiter = maxiter

    def solve(self, b: DofVector) -> DofVector:
        # pyright resolves `minres` to the same-named scipy submodule, not the function
        # (a stub quirk cg does not share); it is callable at runtime.
        x, info = minres(  # pyright: ignore[reportCallIssue]
            self._A, b, rtol=self._rtol, maxiter=self._maxiter, M=self._M)
        if info != 0:
            reason = (
                f'did not converge in {info} iterations'
                if info > 0 else f'illegal input or breakdown (info={info})'
            )
            raise RuntimeError(
                f'MINRES failed to solve the free-free block: {reason}. MINRES needs a '
                'symmetric operator (indefinite is allowed); check symmetry, add an SPD '
                'preconditioner, or use DirectBackend.'
            )
        return np.asarray(x)


def _pyamg_csr(A: Operator) -> csr_array:
    '''CSR with 32-bit indices, which pyamg's compiled kernels require.

    Our assembly scatters from 64-bit DOF-index arrays, so the assembled operators
    carry `int64` indptr/indices; pyamg's strength-of-connection kernel is only
    compiled for `int32` and mis-binds a 64-bit array. The matrices here are far
    from needing 64-bit addressing, so the downcast is safe.
    '''
    A_csr = csr_array(A)
    return csr_array(
        (A_csr.data, A_csr.indices.astype(np.int32), A_csr.indptr.astype(np.int32)),
        shape=A_csr.shape,
    )
