"""How the free-free block gets solved: the linear-algebra backend.

`DiscreteSystem` owns the Dirichlet elimination (the partition of the DOFs and
the lifting of the fixed values to the right-hand side) but not the choice of
how the remaining free-free block is solved. That choice is a `Backend`,
injected so the two orthogonal axes stay separate: a `SolveStrategy` picks linear
vs. Newton, a `Backend` picks direct vs. iterative, and they compose without a
class per combination.

What a caller touches, versus what is plumbing:

    Use:       DirectBackend() | IterativeBackend()   <- the entire public choice
    Extend:    Backend, LinearSolver                  <- implement these to add one
    Internal:  _CGSolver, _pyamg_csr                  <- never named outside here

`DirectBackend` LU-factors the block (`splu`) and back-substitutes: robust for any
nonsingular system, indefinite ones included, but its fill-in on a 3D mesh grows
super-linearly and caps the reachable resolution. `IterativeBackend` runs
preconditioned conjugate gradients with an algebraic-multigrid preconditioner
(`pyamg`); CG is SPD-only, so it is opt-in (Poisson / small-strain elasticity
stiffness, mass, the time-stepping operators), and on a large 3D system it is O(n)
where the direct factorization is not, the whole point of the exercise.

Both `prepare` an operator into a `LinearSolver`: an object that has
factored/preconditioned one matrix and can solve it against many right-hand sides.
`DiscreteSystem` builds one per operator and reuses it across solves, so a
time-stepper or Newton loop with a constant operator pays the setup once. scipy's
`SuperLU` already is such an object; the iterative path wraps CG in one.

Config (the immutable `Backend`) and the bound solver (`LinearSolver`) are two
objects rather than one because the matrix actually solved (the eliminated
free-free block) is born inside `DiscreteSystem`, so a caller can only hand in
a recipe for building the solver, never the solver itself.

The AMG preconditioner is currently `pyamg`'s smoothed aggregation. A hand-rolled
geometric two-grid V-cycle could replace it behind this same `Backend` seam
without touching a caller (see BACKLOG.md).
"""
from typing import Protocol

import numpy as np
import pyamg
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import cg, splu

from fem.typing import DofVector, FloatArray, Operator, Vertices

__all__ = ['Backend', 'LinearSolver', 'DirectBackend', 'IterativeBackend', 'rigid_body_modes']


# -- the two contracts: implement both to add a backend ------------------------

class LinearSolver(Protocol):
    '''A matrix that has been factored/preconditioned once and solves many b's.'''

    def solve(self, b: DofVector) -> DofVector: ...


class Backend(Protocol):
    '''A strategy for turning an assembled operator into a reusable `LinearSolver`.'''

    def prepare(self, A: Operator) -> LinearSolver: ...


# -- the backends a caller picks from ------------------------------------------

class DirectBackend:
    '''Sparse LU factorization via `splu`. The default: robust for any operator.'''

    def prepare(self, A: Operator) -> LinearSolver:
        # splu wants CSC; the SuperLU it returns already satisfies LinearSolver
        # (its .solve reuses the factorization), so no wrapper is needed.
        return splu(csc_array(A))


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

    def prepare(self, A: Operator) -> LinearSolver:
        A_csr = _pyamg_csr(A)
        ml = pyamg.smoothed_aggregation_solver(A_csr, B=self.near_null_space)
        return _CGSolver(A_csr, ml.aspreconditioner(), self.rtol, self.maxiter)


# -- near-kernel helper for the elastic AMG path -------------------------------

def rigid_body_modes(vertices: Vertices, n_components: int) -> FloatArray:
    '''The rigid-body modes of an elastic body: the AMG near-kernel for elasticity.

    Rigid translations and (infinitesimal) rotations produce no strain, so they lie
    in the kernel of the unconstrained stiffness: the low-energy modes a plain
    smoother cannot damp and the coarse levels must represent. Feeding them to AMG
    keeps CG's iteration count flat under mesh refinement for a lightly constrained
    body; the constant vector pyamg assumes by default does not.

    Returns `(n_dofs, n_modes)` in the interleaved DOF order (component `d` of node
    `v` at `n_components*v + d`): 3 modes in 2D (two translations, one rotation), 6
    in 3D (three of each). Restrict the rows to the free DOFs before use, so the
    block matches the one `IterativeBackend` is handed.
    '''
    n = len(vertices)
    if n_components == 2:
        x, y = vertices[:, 0], vertices[:, 1]
        B = np.zeros((2 * n, 3))
        B[0::2, 0] = 1.0                      # translate x
        B[1::2, 1] = 1.0                      # translate y
        B[0::2, 2], B[1::2, 2] = -y, x        # rotate in-plane: (-y, x)
        return B
    if n_components == 3:
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        B = np.zeros((3 * n, 6))
        B[0::3, 0] = B[1::3, 1] = B[2::3, 2] = 1.0   # three translations
        B[1::3, 3], B[2::3, 3] = -z, y               # rotate about x: (0, -z, y)
        B[0::3, 4], B[2::3, 4] = z, -x               # rotate about y: (z, 0, -x)
        B[0::3, 5], B[1::3, 5] = -y, x               # rotate about z: (-y, x, 0)
        return B
    raise ValueError(f'rigid-body modes are defined for 2D or 3D elasticity, not n_components={n_components}')


# -- internal plumbing: not part of the public surface -------------------------

class _CGSolver:
    '''Preconditioned CG bound to one operator and its AMG preconditioner.

    Holds the AMG hierarchy built by `IterativeBackend.prepare`, so each `solve`
    reuses it rather than re-coarsening. Fails loudly: CG reports non-convergence
    (or an illegal input) through a nonzero `info`, and a silently-wrong vector is
    worse than a raise, so we raise.
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
