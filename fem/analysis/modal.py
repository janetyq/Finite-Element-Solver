"""Modal analysis: a `Problem` -> natural frequencies and mode shapes.

Free (undamped) vibration: the finite-element analogue of a beam's natural tones.
Where buckling asks how far a load can be scaled before the structure snaps sideways,
modal analysis asks the load-free question: what shapes does the structure oscillate
in when displaced and released, and at what frequencies?

Undamped free vibration is `M u'' + K u = 0`; a standing wave `u(t) = phi cos(omega t)`
turns it into the generalized symmetric eigenproblem

    K phi = omega^2 M phi,

with `K` the stiffness and `M` the consistent mass matrix. The eigenvalues are the
squared natural angular frequencies and the eigenvectors the mode shapes. No applied
load enters, so the result is a property of the structure alone.

The lowest frequencies are the ones that matter (a forcing near them resonates), so the
eigensolve uses shift-invert about `sigma = 0` through `EigenSolve`, factoring `K` on
the free block once. `M` is the problem's mass (the equation's `density` times the
consistent mass matrix); the frequencies go as `sqrt(E / density)`, while the mode shapes
are set by geometry and supports alone.
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from fem.problem import LinearProblem
from fem.post.solution import ModalSolution
from fem.algebra.solve import EigenSolve
from fem.typing import FloatArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalAnalysis:
    '''Free-vibration modes of a `LinearProblem`: the natural frequencies and shapes.

    The problem's Dirichlet supports ground the structure (a cantilever's clamp); its
    load is unused, the modes being load-free. Its `mass` (the equation's density
    times the consistent mass matrix) sets the physical frequency units.

    The supports must remove every rigid-body mode: shift-invert about zero factors `K`
    on the free block, which is singular if the structure can translate or rotate
    freely. A fully unsupported structure needs a different shift and is out of scope.
    Bending dominates the low modes and the constant-strain triangle locks in bending;
    a P2 space reaches the analytic frequencies on a far coarser mesh.
    '''
    n_modes: int = 6

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError(f'n_modes must be at least 1, got {self.n_modes}')

    def solve(self, problem: LinearProblem[Any]) -> ModalSolution:
        '''The modal frequencies and mode shapes of `problem`'s stiffness and supports.'''
        if not problem.is_linear:
            raise TypeError(
                'modal analysis needs a constant tangent (the small-strain stiffness); '
                f'{type(problem.operator).__name__} depends on the state'
            )
        if 2 not in problem.time_orders:
            raise TypeError(
                'modal analysis is the free vibration of a second-order system; this problem '
                f'allows time orders {sorted(problem.time_orders)}'
            )
        space = problem.space
        K = problem.tangent()
        M = problem.mass

        logger.info('Modal: solving the eigenproblem K phi = omega^2 M phi...')
        # Shift-invert about zero returns the smallest omega^2 (the lowest frequencies,
        # the ones a forcing resonates with) factoring K on the free block once.
        eigensolve = EigenSolve(self.n_modes, sigma=0.0, which='LM')
        free = problem.constraints[0]
        omega_squared, modes = eigensolve.solve(K, M, free, space.n_dofs)

        frequencies, modes = _natural_frequencies(omega_squared, modes)
        return ModalSolution(space, frequencies, modes)


def _natural_frequencies(
    omega_squared: FloatArray, modes: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    '''Sort by frequency and take omega = sqrt(omega^2), ascending.

    Round-off can push a well-constrained mode's eigenvalue a hair below zero; clamp
    at zero before the square root so it reads as omega = 0 rather than a NaN. The
    modes ride along with their frequencies so the returned pair stays aligned.
    '''
    order = np.argsort(omega_squared)
    omega = np.sqrt(np.maximum(omega_squared[order], 0.0))
    return omega, modes[order]
