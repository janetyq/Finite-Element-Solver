"""Solve strategies over an assembled system.

`LinearSolve` and `NewtonSolve` consume a `Problem` and return a DOF vector;
`EigenSolve` consumes an operator pair plus constraints and returns eigenpairs.
The first two sit on `DiscreteSystem` (matrix + Dirichlet partition + factor-once
solve) and know nothing about which PDE produced the `Problem`. `LinearSolve`
assembles once and solves once; `NewtonSolve` iterates, and on a `LinearProblem`
(constant tangent, affine residual) reaches the solution in one step.

`EigenSolve` is the eigen-analogue: no right-hand side, the same free-block Dirichlet
elimination. `BucklingAnalysis` and `ModalAnalysis` interpret its eigenvalues.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from scipy.sparse import eye_array
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from fem.algebra.backends import Backend, IterativeBackend, MinresBackend
from fem.field import NodalField
from fem.problem import Problem
from fem.algebra.system import DiscreteSystem
from fem.typing import Constraints, DofIndices, DofVector, FloatArray, Operator


class SolveStrategy(Protocol):
    '''How a `Problem` is iterated to its solution. Orthogonal to the `Backend`, which is
    how each linear system on the way is solved and is given at the call.'''

    def solve(self, problem: Problem, u0: DofVector | NodalField | None = None, *,
              backend: Backend | None = None) -> DofVector: ...


def default_strategy(problem: Problem) -> 'SolveStrategy':
    '''The strategy a caller gets by naming none: `LinearSolve` for a constant tangent,
    line-searched `NewtonSolve` otherwise (which regularizes its tangent by itself when
    the backend it is handed is iterative).'''
    if problem.is_linear:
        return LinearSolve()
    return NewtonSolve(line_search=BacktrackingLineSearch())


def backend_for(problem: Problem, backend: Backend | None) -> Backend | None:
    '''`backend`, given the problem's AMG near-kernel if it is iterative and has none.

    An elasticity stiffness has the rigid-body modes as its low-energy near-kernel,
    and AMG needs them to keep CG's iteration count flat under refinement. The problem
    supplies them over all DOFs; they are restricted here to the free block the backend
    factors. A near-kernel the caller set is left untouched.
    '''
    if not isinstance(backend, IterativeBackend) or backend.near_null_space is not None:
        return backend
    modes = problem.near_null_space()
    if modes is None:
        return backend
    free = problem.constraints[0]
    return backend.with_near_null_space(modes[free])


@dataclass(frozen=True)
class LinearSolve:
    '''Assemble once, solve once: for a `Problem` with a state-independent tangent.
    Nothing to configure: a named choice.

    `backend` (at the call) selects the linear algebra for the one solve: direct by
    default, or an `IterativeBackend` for a large SPD system (Poisson, small-strain
    elasticity), which is handed the problem's near-kernel (see `backend_for`).
    '''

    def solve(self, problem: Problem, u0: DofVector | NodalField | None = None, *,
              backend: Backend | None = None) -> DofVector:
        if not problem.is_linear:
            raise TypeError(
                f'LinearSolve needs a constant tangent; {type(problem.operator).__name__} '
                'depends on the state. Use NewtonSolve.'
            )
        backend = backend_for(problem, backend)
        system = DiscreteSystem(problem.tangent(), problem.constraints, backend)
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


@dataclass(frozen=True)
class TangentRegularization:
    '''Shift a Newton tangent by tau*I until the step is a descent direction.

    Near a saddle the energy Hessian is indefinite, and a plain Newton step can point
    uphill: no length of it decreases the energy, so a line search stalls at a vanishing
    step. Adding a positive multiple of the identity, `(H + tau I) du = -r`, lifts the
    smallest eigenvalue; for tau large enough the shifted tangent is positive-definite and
    the step approaches steepest descent, which always descends. tau escalates geometrically
    from a small fraction of the tangent's diagonal scale, and the first shift that yields a
    descent step is taken.

    A positive-definite tangent is never shifted (its first, tau=0 step already descends),
    so the usual case, and every `LinearProblem`, keeps plain Newton and its quadratic rate.

    Descent is judged by `r_free . step < 0`, the condition for the energy merit a
    globalized `NewtonSolve` minimises, so this targets energy-minimising Newton (a
    problem with `has_energy`), the nonlinear path here. The escalation also retries when the backend reports a
    breakdown, so it composes with an indefinite-capable iterative backend (`MinresBackend`);
    an SPD-only backend (CG) is not made reliable on an indefinite tangent by it, since CG's
    failure there is not always signalled.
    '''
    max_shifts: int = 20
    base_factor: float = 1e-6
    growth: float = 10.0

    def schedule(self, diagonal_scale: float):
        '''The tau values to try in order: 0 first (plain Newton), then escalating shifts.'''
        yield 0.0
        tau = self.base_factor * diagonal_scale
        for _ in range(self.max_shifts):
            yield tau
            tau *= self.growth


@dataclass(frozen=True)
class NewtonSolve:
    '''Newton's method on r(u) = 0, re-factoring the tangent each iteration.

    The increment is pinned to zero at the fixed DOFs (the seed already carries
    their Dirichlet values) and `DiscreteSystem` eliminates them, so the tangent
    needs no special-casing. Convergence is checked before the step is applied, so a
    sub-tolerance increment is never added: on a `LinearProblem` the first step is
    exact and the second is zero, so the exact answer is reached in one applied step.
    The test is relative, `‖Δu‖ < tol · max(1, ‖u‖)`, so `tol` means the same on a
    metre-scale field as on a millimetre one. Exhausting `max_iters` without meeting
    it raises `RuntimeError`; an unconverged state is never returned as an answer.

    `line_search=None` takes the full step every iteration (the plain method). Passing
    a `BacktrackingLineSearch` globalizes it: each step is scaled to decrease a merit,
    the problem's energy Π(u) when it has one (`has_energy`) else ½‖r‖², so a
    non-convex energy (St-Venant–Kirchhoff under compression) converges from a seed a
    full step would send diverging. The line search is a no-op where the full step
    already works, including every `LinearProblem`, whose exact step passes at alpha = 1.

    The `backend` given at the call selects the linear algebra for each tangent solve
    (direct by default). A nonlinear tangent is indefinite away from a convex minimum, so
    an iterative backend for it must handle that: `MinresBackend`, not the SPD-only CG
    `IterativeBackend`. `regularization` (a `TangentRegularization`) steers each step to a
    descent direction by shifting an indefinite tangent; without it, an indefinite tangent
    falls back to the full step (line-search globalization only). The default `'auto'`
    regularizes exactly when the backend is iterative, where an indefinite tangent would
    otherwise break the solve; `None` never does, an instance always does.
    '''

    max_iters: int = 100
    tol: float = 1e-6
    line_search: BacktrackingLineSearch | None = None
    regularization: TangentRegularization | None | Literal['auto'] = 'auto'

    def regularization_for(self, backend: Backend | None) -> TangentRegularization | None:
        '''The regularization a solve over `backend` applies (see the class docstring).'''
        if isinstance(self.regularization, str):   # 'auto'
            iterative = isinstance(backend, (IterativeBackend, MinresBackend))
            return TangentRegularization() if iterative else None
        return self.regularization

    def solve(self, problem: Problem, u0: DofVector | NodalField | None = None, *,
              backend: Backend | None = None) -> DofVector:
        free, fixed, fixed_values = problem.constraints
        u = np.zeros(problem.space.n_dofs) if u0 is None else np.asarray(u0, dtype=float).copy()
        u[fixed] = fixed_values

        regularization = self.regularization_for(backend)
        step_constraints = (free, fixed, np.zeros(len(fixed)))
        step_norm = np.inf
        for _ in range(self.max_iters):
            residual = problem.residual(u)
            step = self._compute_step(problem, u, residual, free, step_constraints,
                                      backend, regularization)
            step_norm = float(np.linalg.norm(step))
            if step_norm < self.tol * max(1.0, float(np.linalg.norm(u))):
                return u
            u = self._advance(problem, free, u, step, residual)
        raise RuntimeError(
            f'Newton did not converge in {self.max_iters} iterations: the last step had '
            f'norm {step_norm:.3e} against a tolerance of {self.tol:.1e} relative to the '
            f'state. Raise max_iters, add a line search, or start from a closer seed.'
        )

    def _compute_step(
        self, problem: Problem, u: DofVector, residual: DofVector,
        free: DofIndices, step_constraints: Constraints,
        backend: Backend | None, regularization: TangentRegularization | None,
    ) -> DofVector:
        '''The Newton increment, optionally regularized to a descent direction.

        Plain Newton solves `H du = -r` once. With a `regularization`, `H` is shifted by
        `tau*I` and re-solved, escalating tau until the step descends (`r_free . step < 0`),
        so an indefinite tangent still yields a usable direction and an SPD-only backend
        stays safe. A positive-definite tangent is accepted at the first (tau=0) shift, so
        the common case pays no extra solve.
        '''
        H = problem.tangent(u)
        rhs = -residual
        if regularization is None:
            return DiscreteSystem(H, step_constraints, backend).solve(rhs)

        identity = eye_array(H.shape[0], format='csr')
        diagonal_scale = float(np.abs(H.diagonal()).mean()) or 1.0
        step = None
        for tau in regularization.schedule(diagonal_scale):
            operator = H if tau == 0.0 else H + tau * identity
            try:
                step = DiscreteSystem(operator, step_constraints, backend).solve(rhs)
            except RuntimeError:
                # An SPD-only backend (CG) breaks down on a still-indefinite shift; escalate.
                continue
            if float(residual[free] @ step[free]) < 0.0:
                return step
        if step is None:
            raise RuntimeError(
                'Newton could not solve even the most regularized tangent: the backend '
                'rejected every shift. Use DirectBackend, or widen the regularization.'
            )
        return step

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
        # vanishing alpha. A TangentRegularization (when configured) already steers the step
        # to descent upstream, so this fallback is the last resort of the unregularized path.
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
        if problem.has_energy:
            return problem.energy, slope_free

        def residual_norm(w: DofVector) -> float:
            r = problem.residual(w)[free]
            return 0.5 * float(r @ r)

        return residual_norm, -float(residual[free] @ residual[free])


@dataclass(frozen=True)
class EigenSolve:
    '''A generalized symmetric eigenproblem `A φ = μ B φ`, reduced to the free block.

    `(A, B)` is the pencil, the matrix pair whose generalized eigenvalues μ and
    eigenvectors φ are sought. The eigen-analogue of `LinearSolve`: it eliminates the
    Dirichlet DOFs, hands the free-free pencil to `scipy.sparse.linalg.eigsh`, and lifts
    each eigenvector back to a full DOF vector (the fixed DOFs are zero in every mode). It
    does not interpret the eigenvalues: `BucklingAnalysis` reads `μ = 1/λ`, `ModalAnalysis`
    reads `μ = ω²`. Two modes, by one pair of knobs:

    - Regular (`sigma=None`): the largest/smallest `μ` by `which` (buckling uses
      `which='LA'` for the largest).
    - Shift-invert (`sigma` set): the `μ` nearest `sigma`, factoring `A - sigma B`
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
