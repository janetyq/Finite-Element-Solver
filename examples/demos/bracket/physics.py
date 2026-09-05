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

from fem.analysis.estimators import RecoveryEstimator
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.mesh.curves import Arc, Line
from fem.mesh.mesh import Mesh
from fem.mesh.outline import Outline
from fem.mesh.refinement import RedGreenRefiner
from fem.physics.equations import LinearElastic
from fem.post.solution import ElasticSolution
from fem.regions import on_plane


def l_bracket_outline(arm: float = 4.0, width: float = 1.2,
                      fillet_radius: float = 0.0) -> Outline:
    """An L-shaped bracket, with an optional fillet at the inner corner.

    Two limbs of thickness `width` and length `arm`: the vertical one up the left edge,
    the horizontal one along the bottom, meeting at a re-entrant (inner) corner at
    `(width, width)`. A sharp corner there is a stress singularity; `fillet_radius > 0`
    rounds it with a concave `Arc`, so an isoparametric solve reads a true circular
    fillet.

    Clamp the top of the vertical limb (`on_plane(1, arm)`) and load the tip of the
    horizontal one (`on_plane(0, arm)`); the concentration then sits at the inner corner.
    """
    corners = [np.array(p) for p in [(0.0, 0.0), (arm, 0.0), (arm, width)]]
    pieces = [Line(corners[0], corners[1]), Line(corners[1], corners[2])]
    if fillet_radius > 0:
        # Round the re-entrant corner: an arc of radius r centred at (width+r, width+r),
        # bulging into the notch to add material. It runs from A = (width+r, width) on the
        # bottom limb's top edge (theta = 3pi/2) to B = (width, width+r) on the vertical
        # limb's right edge (theta = pi), replacing the sharp point between them: the
        # arc reversed, since the outline is traced clockwise through it.
        r = fillet_radius
        fillet = Arc([width + r, width + r], r, np.pi, 1.5 * np.pi).reversed()
        pieces += [Line(corners[2], fillet.start), fillet]
        inner_end = fillet.end
    else:
        pieces.append(Line(corners[2], [width, width]))
        inner_end = np.array([width, width])
    top = [np.array(p) for p in [(width, arm), (0.0, arm)]]
    pieces += [Line(inner_end, top[0]), Line(top[0], top[1]), Line(top[1], corners[0])]
    return Outline([pieces])


def make_bc(arm, traction) -> Conditions:
    """Clamped at the top of the upright limb, pulled down at the horizontal tip."""
    return Conditions(
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
    outline = l_bracket_outline(arm, width, fillet_radius=fillet)
    mesh = outline.mesh(min_angle=min_angle, max_area_fraction=max_area_fraction)

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
    bc: Conditions
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
