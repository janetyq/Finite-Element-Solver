"""Solve strategies over an assembled system.

`LinearSolve` and `NewtonSolve` consume a `Problem` and return a DOF vector;
`EigenSolve` consumes an operator pair plus constraints and returns eigenpairs.
The first two sit on the one algebra atom, `DiscreteSystem` (matrix + Dirichlet
partition + factor-once solve), and know nothing about which PDE produced the
`Problem`. `LinearSolve` assembles once and solves once; `NewtonSolve` iterates.
The two are one engine: a `LinearProblem` has a constant tangent and an affine
residual, so `NewtonSolve` reaches its solution in a single step from any seed;
`LinearSolve` is that step done directly, skipping the residual evaluation.

`EigenSolve` is the eigen-analogue: an eigenproblem has no right-hand side, so it
cannot go through `DiscreteSystem`, but the Dirichlet elimination is the same
free-block reduction. `BucklingSolver` and `ModalSolver` are thin facades over it.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from fem.backends import Backend
from fem.problem import Problem
from fem.system import DiscreteSystem
from fem.typing import DofIndices, DofVector, FloatArray, Operator


class SolveStrategy(Protocol):
    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector: ...


class LinearSolve:
    '''Assemble once, solve once: for a `Problem` with a state-independent tangent.

    `backend` selects the linear algebra for the one solve: direct by default,
    or an `IterativeBackend` for a large SPD system (Poisson, small-strain elasticity).
    '''

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend

    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector:
        system = DiscreteSystem(problem.tangent(None), problem.constraints, self.backend)
        return system.solve(problem.load)


class NewtonSolve:
    '''Newton's method on r(u) = 0, re-factoring the tangent each iteration.

    The increment is pinned to zero at the fixed DOFs (the seed already carries
    their Dirichlet values) and `DiscreteSystem` eliminates them, so the tangent
    needs no special-casing. Convergence is checked before the step is applied, so a
    sub-tolerance increment is never added: on a `LinearProblem` the first step is
    exact and the second is zero, so the exact answer is reached in one applied step.
    '''

    def __init__(self, max_iters: int = 100, tol: float = 1e-6) -> None:
        self.max_iters = max_iters
        self.tol = tol

    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector:
        free, fixed, fixed_values = problem.constraints
        u = np.zeros(problem.space.n_dofs) if u0 is None else np.asarray(u0, dtype=float).copy()
        u[fixed] = fixed_values

        step_constraints = (free, fixed, np.zeros(len(fixed)))
        for _ in range(self.max_iters):
            system = DiscreteSystem(problem.tangent(u), step_constraints)
            step = system.solve(-problem.residual(u))
            if np.linalg.norm(step) < self.tol:
                break
            u = u + step
        return u


@dataclass(frozen=True)
class EigenSolve:
    '''A generalized symmetric eigenproblem `A φ = μ B φ`, reduced to the free block.

    `(A, B)` is the pencil, the matrix pair whose generalized eigenvalues μ and
    eigenvectors φ are sought. The eigen-analogue of `LinearSolve`: it eliminates the
    Dirichlet DOFs, hands the free-free pencil to `scipy.sparse.linalg.eigsh`, and lifts
    each eigenvector back to a full DOF vector (the fixed DOFs are zero in every mode). It
    does not interpret the eigenvalues: `BucklingSolver` reads `μ = 1/λ`, `ModalSolver`
    reads `μ = ω²`. Two modes, by one pair of knobs:

    - **Regular** (`sigma=None`): the largest/smallest `μ` by `which` (buckling uses
      `which='LA'` for the largest).
    - **Shift-invert** (`sigma` set): the `μ` nearest `sigma`, factoring `A - sigma B`
      once (modal uses `sigma=0, which='LM'` for the smallest `ω²`). Needs `A - sigma B`
      non-singular, so a structure with rigid-body modes needs a nonzero shift.

    `B` must be positive definite (the mass side of the pencil). Both operators are
    symmetrised against round-off, which `eigsh` assumes.
    '''
    n_modes: int
    which: str = 'LM'
    sigma: float | None = None

    def solve(
        self, A: Operator, B: Operator, free: DofIndices, n_dofs: int,
    ) -> tuple[FloatArray, FloatArray]:
        '''The `n_modes` eigenpairs of `A φ = μ B φ`, modes lifted to `n_dofs` vectors.

        Returns `(mu, modes)`, shapes `(k,)` and `(k, n_dofs)` with `k <= n_modes` (fewer
        if the free block is small or higher modes stall). No ordering or sign beyond
        `eigsh`'s; the facade owns interpretation.
        '''
        Aff = A[np.ix_(free, free)]
        Bff = B[np.ix_(free, free)]
        n_free = Aff.shape[0]

        # eigsh (Lanczos) needs headroom above the modes requested; cap so a small system
        # asks for fewer rather than failing.
        k = min(self.n_modes, n_free - 2)
        if k < 1:
            raise ValueError(
                f'too few free DOFs ({n_free}) to extract an eigenmode; the system is '
                'over-constrained or the mesh is trivially small'
            )

        A_sym = (0.5 * (Aff + Aff.T)).tocsc()
        B_sym = (0.5 * (Bff + Bff.T)).tocsc()
        try:
            mu, vecs = eigsh(A_sym, k=k, M=B_sym, sigma=self.sigma, which=self.which)
        except ArpackNoConvergence as failure:
            # Keep whatever converged: the lower modes a caller wants resolve first;
            # only nothing-converged is fatal.
            mu, vecs = failure.eigenvalues, failure.eigenvectors
            if mu.size == 0:
                raise ValueError(
                    'the eigensolver did not converge to any mode; the mesh may be too '
                    'coarse for the modes requested, or the system ill-posed'
                ) from failure

        modes = np.zeros((len(mu), n_dofs))
        modes[:, free] = vecs.T
        return mu, modes
