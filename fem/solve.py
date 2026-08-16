"""Solve strategies over an assembled system.

`LinearSolve` and `NewtonSolve` consume a `Problem` and return a DOF vector;
`EigenSolve` consumes an operator *pair* plus constraints and returns eigenpairs.
The first two sit on the one algebra atom, `DiscreteSystem` (matrix + Dirichlet
partition + factor-once solve), and know nothing about which PDE produced the
`Problem`. `LinearSolve` assembles once and solves once; `NewtonSolve` iterates.
The two are one engine: a `LinearProblem` has a constant tangent and an affine
residual, so `NewtonSolve` reaches its solution in a single step from any seed --
`LinearSolve` is that step done directly, skipping the residual evaluation.

`EigenSolve` is the eigen-analogue: an eigenproblem has no right-hand side, so it
cannot go through `DiscreteSystem`, but the Dirichlet elimination is the same
free-block reduction. `BucklingSolver` and `ModalSolver` are thin facades over it.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from fem.backends import Backend
from fem.problem import Problem, SupportsEnergy
from fem.system import DiscreteSystem
from fem.typing import DofIndices, DofVector, FloatArray, Operator


class SolveStrategy(Protocol):
    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector: ...


class LinearSolve:
    '''Assemble once, solve once: for a `Problem` with a state-independent tangent.

    `backend` selects the linear algebra for the one solve -- direct by default,
    or an `IterativeBackend` for a large SPD system (Poisson, small-strain elasticity).
    '''

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend

    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector:
        system = DiscreteSystem(problem.tangent(None), problem.constraints, self.backend)
        return system.solve(problem.load)


@dataclass(frozen=True)
class BacktrackingLineSearch:
    '''Armijo backtracking: scale a Newton step so a merit function decreases.

    Given a search direction and a merit `m` with directional slope `m'(0) = slope`
    at the current `u`, accept the largest `alpha in {1, rho, rho^2, ...}` meeting the
    sufficient-decrease condition `m(u + alpha*step) <= m(u) + c1*alpha*slope`. Starting
    at `alpha = 1` keeps full Newton (and its quadratic convergence) near the solution;
    shrinking it keeps progress from a poor seed. `slope < 0` (a descent direction) is
    the caller's responsibility; this only chooses the length.
    '''
    c1: float = 1e-4
    rho: float = 0.5
    max_backtracks: int = 20

    def search(
        self,
        merit: Callable[[DofVector], float],
        u: DofVector,
        step: DofVector,
        slope: float,
    ) -> DofVector:
        phi0 = merit(u)
        alpha = 1.0
        for _ in range(self.max_backtracks):
            if merit(u + alpha * step) <= phi0 + self.c1 * alpha * slope:
                break
            alpha *= self.rho
        return u + alpha * step


class NewtonSolve:
    '''Newton's method on r(u) = 0, re-factoring the tangent each iteration.

    The increment is pinned to zero at the fixed DOFs -- the seed already carries
    their Dirichlet values -- and `DiscreteSystem` eliminates them, so the tangent
    needs no special-casing. Convergence is checked before the step is applied, so a
    sub-tolerance increment is never added: on a `LinearProblem` the first step is
    exact and the second is zero, so the exact answer is reached in one applied step.

    `line_search=None` takes the full step every iteration (the plain method). Passing
    a `BacktrackingLineSearch` globalizes it: each step is scaled to decrease a merit,
    the problem's energy Π(u) when it has one (`SupportsEnergy`) else ½‖r‖², so a
    non-convex energy (St-Venant–Kirchhoff under compression) converges from a seed a
    full step would send diverging. The line search is a no-op where the full step
    already works, including every `LinearProblem`, whose exact step passes at alpha = 1.
    '''

    def __init__(
        self,
        max_iters: int = 100,
        tol: float = 1e-6,
        line_search: BacktrackingLineSearch | None = None,
    ) -> None:
        self.max_iters = max_iters
        self.tol = tol
        self.line_search = line_search

    def solve(self, problem: Problem, u0: DofVector | None = None) -> DofVector:
        free, fixed, fixed_values = problem.constraints
        u = np.zeros(problem.space.n_dofs) if u0 is None else np.asarray(u0, dtype=float).copy()
        u[fixed] = fixed_values

        step_constraints = (free, fixed, np.zeros(len(fixed)))
        for _ in range(self.max_iters):
            system = DiscreteSystem(problem.tangent(u), step_constraints)
            residual = problem.residual(u)
            step = system.solve(-residual)
            if np.linalg.norm(step) < self.tol:
                break
            u = self._advance(problem, free, u, step, residual)
        return u

    def _advance(
        self, problem: Problem, free: DofIndices, u: DofVector, step: DofVector,
        residual: DofVector,
    ) -> DofVector:
        '''Apply the step, line-searched if a search is configured and the step descends.'''
        if self.line_search is None:
            return u + step
        merit, slope = self._merit(problem, free, step, residual)
        # An indefinite tangent can leave the Newton step non-descent (slope >= 0), where
        # backtracking cannot help; fall back to the full step rather than stalling at a
        # vanishing alpha. Making the tangent SPD there is the open globalization work.
        if slope >= 0:
            return u + step
        return self.line_search.search(merit, u, step, slope)

    @staticmethod
    def _merit(
        problem: Problem, free: DofIndices, step: DofVector, residual: DofVector,
    ) -> tuple[Callable[[DofVector], float], float]:
        '''The merit function and the Newton step's slope on it at the current state.

        Restricted to the free DOFs, which are the system being solved: the fixed DOFs
        carry reaction forces that stay nonzero at equilibrium, so a residual over all
        DOFs would not be minimised at the solution. Energy Π(u) when the problem has one,
        with slope r_free·step; otherwise ½‖r_free‖², whose slope along the Newton step is
        -‖r_free‖² exactly (r^T J step = -r^T r).
        '''
        slope_free = float(residual[free] @ step[free])
        if isinstance(problem, SupportsEnergy):
            return problem.energy, slope_free

        def residual_norm(w: DofVector) -> float:
            r = problem.residual(w)[free]
            return 0.5 * float(r @ r)

        return residual_norm, -float(residual[free] @ residual[free])


@dataclass(frozen=True)
class EigenSolve:
    '''A generalized symmetric eigenproblem `A φ = μ B φ`, reduced to the free block.

    `(A, B)` is the *pencil* -- the matrix pair whose generalized eigenvalues μ and
    eigenvectors φ are sought. The eigen-analogue of `LinearSolve`: it eliminates the
    Dirichlet DOFs, hands the free-free pencil to `scipy.sparse.linalg.eigsh`, and lifts
    each eigenvector back to a full DOF vector (the fixed DOFs are zero in every mode). It does not interpret the
    eigenvalues -- `BucklingSolver` reads `μ = 1/λ`, `ModalSolver` reads `μ = ω²`. Two
    modes, by one pair of knobs:

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
        `eigsh`'s -- the facade owns interpretation.
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
            # Keep whatever converged -- the lower modes a caller wants resolve first;
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
