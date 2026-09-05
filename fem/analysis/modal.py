"""Modal analysis: a `Problem` -> natural frequencies and mode shapes.

Free (undamped) vibration: the finite-element analogue of a beam's natural tones. What
shapes does the structure oscillate in when displaced and released, and at what
frequencies?

Undamped free vibration is `M u'' + K u = 0`; a standing wave `u(t) = phi cos(omega t)`
turns it into the generalized symmetric eigenproblem

    K phi = omega^2 M phi,

with `K` the stiffness and `M` the consistent mass matrix. The eigenvalues are the
squared natural angular frequencies and the eigenvectors the mode shapes. `ModalAnalysis`
solves it about the unstressed state, where no applied load enters and the result is a
property of the structure alone.

`PrestressedModalAnalysis` is the same question asked of a loaded structure. A reference
solve recovers the prestress, its geometric stiffness `K_g` joins the pencil,

    (K + K_g(sigma_0)) phi = omega^2 M phi,

and the frequencies shift with the load: tension stiffens and raises them, compression
softens and lowers them, down to zero at the buckling load, where the two analyses meet
(`BucklingAnalysis` is the tool for that load itself).

The lowest frequencies are the ones that matter (a forcing near them resonates), so the
eigensolve uses shift-invert about `sigma = 0` through `EigenSolve`, factoring the
stiffness on the free block once. `M` is the problem's mass (the equation's `density`
times the consistent mass matrix); the frequencies go as `sqrt(E / density)`, while the
unstressed mode shapes are set by geometry and supports alone.
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from fem.algebra.solve import EigenSolve, LinearSolve
from fem.physics.forms import GeometricStiffnessForm
from fem.post.solution import ElasticSolution, ModalSolution
from fem.problem import LinearProblem
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
        _check_vibration_problem(problem, 'modal analysis')
        space = problem.space
        K = problem.tangent()
        M = problem.mass

        logger.info('Modal: solving the eigenproblem K phi = omega^2 M phi...')
        # Shift-invert about zero returns the smallest omega^2 (the lowest frequencies,
        # the ones a forcing resonates with) factoring K on the free block once.
        eigensolve = EigenSolve(self.n_modes, sigma=0.0, which='LM')
        omega_squared, modes = eigensolve.solve(K, M, problem.partition)

        frequencies, modes = _natural_frequencies(omega_squared, modes)
        return ModalSolution(space, frequencies, modes)


@dataclass(frozen=True)
class PrestressedModalAnalysis:
    '''Vibration of a loaded `LinearProblem`: how its frequencies shift under prestress.

    The sibling of `ModalAnalysis` for a structure that is carrying something. A
    reference linear solve recovers the prestress the problem's own load produces, its
    geometric stiffness `K_g` is added to `K`, and the pencil
    `(K + K_g) phi = omega^2 M phi` is solved for the lowest modes. Tension raises the
    frequencies, compression lowers them: the guitar string tuned by its tension, the
    column that goes quiet as it approaches buckling.

    The analysis is posed at the problem's stated load. A caller sweeping the load walks
    `problem.with_load_factor(f)`, exactly as the load factors of `BucklingAnalysis` are
    read. An unloaded problem is allowed: the prestress is then zero, `K_g` vanishes, and
    the result is `ModalAnalysis`'s, the baseline such a sweep starts from.

    Past the buckling load `K + K_g` is indefinite and the lowest `omega^2` goes
    negative: there is no oscillation about a state the structure will not stay in, so
    the analysis raises rather than reporting an imaginary frequency. `BucklingAnalysis`
    is what answers the question at that point.

    As with `ModalAnalysis`, the supports must remove every rigid-body mode, and bending
    dominates the low modes: use a P2 space, since the constant-strain triangle locks.
    '''
    n_modes: int = 6

    def __post_init__(self) -> None:
        if self.n_modes < 1:
            raise ValueError(f'n_modes must be at least 1, got {self.n_modes}')

    def solve(self, problem: LinearProblem[ElasticSolution]) -> ModalSolution:
        '''The modal frequencies and shapes of `problem` under its own load's prestress.'''
        _check_vibration_problem(problem, 'prestressed modal analysis')
        logger.info('Prestressed modal: reference solve for the prestress state...')
        space = problem.space
        reference = problem.solve(LinearSolve())
        if not isinstance(reference, ElasticSolution):
            raise TypeError(
                'prestressed modal analysis needs the recovered stress; got a bare FieldSolution'
            )

        d = space.spatial_dim
        # The in-plane prestress drives the geometric stiffness; sigma_zz (the plane-strain
        # out-of-plane component) has no in-plane displacement gradient to couple to.
        prestress = np.ascontiguousarray(reference.stress[:, :d, :d])
        K = problem.tangent() + space.assemble(GeometricStiffnessForm(prestress))
        M = problem.mass

        logger.info('Prestressed modal: solving (K + K_g) phi = omega^2 M phi...')
        eigensolve = EigenSolve(self.n_modes, sigma=0.0, which='LM')
        omega_squared, modes = eigensolve.solve(K, M, problem.partition)
        _check_stability(omega_squared)

        frequencies, modes = _natural_frequencies(omega_squared, modes)
        return ModalSolution(space, frequencies, modes)


def _check_vibration_problem(problem: LinearProblem[Any], label: str) -> None:
    '''The two things free vibration needs of a problem: a constant tangent to form the
    pencil with, and a second time derivative for the mass to multiply.'''
    if not problem.is_linear:
        raise TypeError(
            f'{label} needs a constant tangent (the small-strain stiffness); '
            f'{type(problem.operator).__name__} depends on the state'
        )
    if 2 not in problem.time_orders:
        raise TypeError(
            f'{label} is the free vibration of a second-order system; this problem '
            f'allows time orders {sorted(problem.time_orders)}'
        )


def _check_stability(omega_squared: FloatArray) -> None:
    '''Refuse a genuinely negative `omega^2`: the prestress is at or past buckling.

    Judged against the eigenvalue scale, so the round-off negative a mode at the
    critical load lands on stays admissible (`_natural_frequencies` clamps it to zero)
    while a softened-through mode, orders of magnitude below zero, does not.
    '''
    scale = float(np.max(np.abs(omega_squared)))
    if float(np.min(omega_squared)) < -1e-8 * scale:
        raise ValueError(
            'the prestress leaves the structure at or past its buckling load: the lowest '
            'omega^2 is negative, so the loaded state does not oscillate but gives way. '
            'Use BucklingAnalysis for the critical load, and stay below it here.'
        )


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
