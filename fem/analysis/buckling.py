"""Linearised buckling: a `Problem` -> critical load factors and modes.

The finite-element analogue of Euler's column formula: how far can a load be scaled
before the structure snaps sideways into a different shape? The result is the load
factor λ and the mode it buckles into.

1. Reference solve. Solve the linear-elastic problem under its reference load,
   recovering the membrane prestress σ₀ in every element.
2. Geometric stiffness. Assemble `K_g(σ₀)` (`GeometricStiffnessForm`), the
   initial-stress matrix that softens the structure under compression.
3. Eigenproblem. Solve `K φ = -λ K_g φ` for the lowest few λ through `EigenSolve`.
   `λ_1` is the critical load factor and `φ_1` the shape it buckles into.
"""
import logging
from dataclasses import dataclass

import numpy as np

from fem.algebra.solve import EigenSolve, LinearSolve
from fem.physics.forms import GeometricStiffnessForm
from fem.post.solution import BucklingSolution, ElasticSolution
from fem.problem import LinearProblem
from fem.typing import FloatArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucklingAnalysis:
    '''Linearised buckling of a `LinearProblem`: load factors and mode shapes.

    The problem's boundary conditions encode the reference load whose buckling
    multiplier is sought (a compressive traction on an end, say). The load factors are
    dimensionless multipliers of that load, so the caller multiplies by the reference
    load's magnitude to get the buckling load in physical units. The operator must be
    a small-strain elastic stiffness: the prestress is read from the problem's
    `ElasticSolution`.

    Bending dominates a buckling mode and the linear (constant-strain) triangle locks in
    bending; a P2 space reaches Euler's loads on a far coarser mesh.
    '''
    n_modes: int = 4

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError(f'n_modes must be at least 1, got {self.n_modes}')

    def solve(self, problem: LinearProblem[ElasticSolution]) -> BucklingSolution:
        '''The buckling factors and modes about the problem's reference solve.'''
        if not problem.is_linear:
            raise TypeError(
                'linearised buckling needs a constant tangent (the small-strain stiffness); '
                f'{type(problem.operator).__name__} depends on the state'
            )
        logger.info('Buckling: reference solve for the prestress state...')
        space = problem.space
        reference = problem.solve(LinearSolve())
        if not isinstance(reference, ElasticSolution):
            raise TypeError('buckling needs the recovered stress; got a bare FieldSolution')

        d = space.spatial_dim
        # The stiffness the reference solve already assembled, including any Robin
        # (elastic support) term, which stiffens the structure in the eigenproblem too.
        K = problem.tangent()
        # The in-plane prestress drives the geometric stiffness; σ_zz (the plane-strain
        # out-of-plane component) has no in-plane displacement gradient to couple to.
        prestress = np.ascontiguousarray(reference.stress[:, :d, :d])

        # Buckling needs compression somewhere: if every principal stress of the prestress
        # is non-negative, K_g is positive-semidefinite and K + λ K_g stays SPD for all
        # λ > 0, so nothing buckles. A rigorous, cheap guard for the clean cases (an
        # unstressed structure or one in pure tension) that also spares `eigsh` a
        # trivial eigenproblem (`K_g = 0` forces every μ to 0, so no finite buckling
        # factor). It does not catch a member in overall tension whose clamped end
        # develops local corner compression: that discretely does have a (huge, spurious)
        # mode, and there is no threshold-free way to rule it out here.
        principal = np.linalg.eigvalsh(prestress)     # (n_el, d), ascending per element
        scale = float(np.abs(prestress).max())
        if scale == 0.0 or float(principal.min()) > -1e-9 * scale:
            raise ValueError(
                'no compressive prestress: the reference load leaves the structure in '
                'tension everywhere (or unstressed), which stiffens rather than buckles, '
                'so there is no buckling mode. Reverse the load.'
            )

        K_g = space.assemble(GeometricStiffnessForm(prestress))

        logger.info('Buckling: solving the eigenproblem K phi = -lambda K_g phi...')
        # -K_g φ = μ K φ with K the PD side; μ = 1/λ, so the largest μ ('LA') are the
        # smallest load factors, reached directly without shift-invert.
        eigensolve = EigenSolve(self.n_modes, which='LA')
        mu, modes = eigensolve.solve(-K_g, K, problem.partition)
        factors, modes = _buckling_factors(mu, modes)
        return BucklingSolution(space, factors, modes, reference=reference)


def _buckling_factors(mu: FloatArray, modes: FloatArray) -> tuple[FloatArray, FloatArray]:
    '''Raw eigenvalues `μ = 1/λ` to ascending, positive-only load factors.

    Only positive μ buckle: a negative μ is a direction the load stiffens, not
    softens. Modes ride along with their factors to stay aligned.
    '''
    tol = 1e-8 * float(np.max(np.abs(mu)))
    positive = mu > tol
    if not positive.any():
        raise ValueError(
            'no positive buckling factor found: the reference load puts the structure '
            'in tension, which stiffens rather than buckles it. Reverse the load.'
        )

    factors = 1.0 / mu[positive]
    modes = modes[positive]
    order = np.argsort(factors)
    return factors[order], modes[order]
