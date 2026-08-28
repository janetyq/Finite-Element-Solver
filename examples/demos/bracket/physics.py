"""The stress at an L-bracket's inner corner, sharp and filleted, as the mesh refines
into it.

At the re-entrant corner the exact elastic stress is infinite (it grows like r^(-0.46)
into the corner), so no mesh resolves it and the computed peak keeps climbing under
refinement. A fillet removes the singularity and the peak settles. `refine_and_track`
follows one bracket through that refinement; `run` does it for both and returns a
`BracketStudy` of plain results. Nothing here draws: `figures.py` does that from the
`BracketStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.equations import LinearElastic
from fem.estimators import RecoveryEstimator
from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.regions import on_plane
from fem.solution import ElasticSolution

from domains import l_bracket_pslg


def make_bc(arm, traction) -> BoundaryConditions:
    """Clamped at the top of the upright limb, pulled down at the horizontal tip."""
    return BoundaryConditions(
        Dirichlet(on_plane(1, arm), [0, 0]),
        Neumann(on_plane(0, arm), [0, -traction]),
    )


def corner_peak(solution: ElasticSolution, width) -> float:
    """The von Mises peak near the inner corner, clear of the clamp's own concentration
    at the top. Read from the same L2-recovered nodal field the panels draw."""
    corner = np.array([width, width])
    nodal_vm = solution.nodal_von_mises(method='l2')
    near = np.linalg.norm(solution.space.node_coords - corner, axis=1) < 0.8 * width
    return float(nodal_vm[near].max())


@dataclass
class Bracket:
    """One bracket after refinement, and the corner peak at every round."""
    mesh: Mesh
    solution: ElasticSolution
    sizes: np.ndarray       # element count at each round
    peaks: np.ndarray       # corner von Mises at each round

    @property
    def peak(self) -> float:
        return float(self.peaks[-1])


def refine_and_track(fillet, element_type, equation, bc, arm, width, min_angle,
                     max_area_fraction, n_rounds, refine_fraction) -> Bracket:
    """Adaptively refine one bracket, recording the corner peak each round.

    `AdaptiveRefinement`'s loop, unrolled so the corner peak can be read off every
    intermediate mesh. `element_type` is the straight quadratic triangle for the
    sharp corner and the isoparametric one for the fillet, so the arc is a true circle
    rather than a polygon. The recovery estimator drives refinement (it reads the
    curved fillet's flux correctly).
    """
    pslg = l_bracket_pslg(arm, width, fillet_radius=fillet, n_fillet=20)
    pslg.validate()
    mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                             max_area=max_area_fraction * pslg.area()).refine()

    def solve(m):
        problem = equation.problem(m, bc, element_type=element_type)
        return problem, problem.solve()

    refiner = RedGreenRefiner(mesh)
    estimator = RecoveryEstimator()
    problem, solution = solve(mesh)
    sizes, peaks = [], []
    for _ in range(n_rounds):
        sizes.append(len(mesh.elements))
        peaks.append(corner_peak(solution, width))
        residuals = estimator.estimate(problem, solution)
        refine_idxs = np.flatnonzero(residuals >= refine_fraction * residuals.max())
        mesh = refiner.refine([int(i) for i in refine_idxs])
        problem, solution = solve(mesh)
    sizes.append(len(mesh.elements))
    peaks.append(corner_peak(solution, width))
    return Bracket(mesh, solution, np.array(sizes), np.array(peaks))


@dataclass
class BracketStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    fillet_radius: float
    bc: BoundaryConditions
    sharp: Bracket
    rounded: Bracket

    @property
    def reduction(self) -> float:
        """How much the fillet cuts the corner peak, in percent, at the final meshes."""
        return 100 * (1 - self.rounded.peak / self.sharp.peak)


def run(arm=4.0, width=1.2, fillet_radius=0.25, traction=0.4, E=300.0, nu=0.3,
        min_angle=28, max_area_fraction=0.0015, n_rounds=18, refine_fraction=0.9) -> BracketStudy:
    """Refine the sharp and the filleted bracket into their corners."""
    equation = LinearElastic(E, nu)
    bc = make_bc(arm, traction)
    refine = (arm, width, min_angle, max_area_fraction, n_rounds, refine_fraction)
    sharp = refine_and_track(0.0, QuadraticTriangleElement, equation, bc, *refine)
    rounded = refine_and_track(fillet_radius, IsoparametricTriangleElement, equation, bc,
                               *refine)
    return BracketStudy(fillet_radius, bc, sharp, rounded)
