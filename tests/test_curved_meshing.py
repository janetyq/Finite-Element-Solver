"""Curved outlines through the meshing pipeline: PSLG -> Ruppert -> refinement.

A PSLG loop that carries a `Circle` produces a mesh whose rim facets carry that circle,
so Ruppert's split points and the red-green midpoints land on the true curve and an
isoparametric solve reads a genuinely round boundary. The Kirsch test is the physical
payoff: at a coarse hole sampling the curved element recovers far more of the true
stress concentration than the straight faceted one.
"""
import numpy as np

from fem.boundary import BCType, BoundaryConditions
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.forms import LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.curves import Circle
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import PSLG
from fem.problem import LinearProblem
from fem.regions import intersect, on_plane
from fem.solution import ElasticSolution
from fem.solve import LinearSolve
from fem.space import FunctionSpace


def _disk_pslg(radius, segments):
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rim = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return PSLG.from_loops([rim], curves=[Circle([0.0, 0.0], radius)])


def _plate_with_hole_pslg(length, height, radius, segments):
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    cx, cy = length / 2, height / 2
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    hole = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
    return PSLG.from_loops([outline, hole], curves=[None, Circle([cx, cy], radius)])


def _disk_mesh(radius=1.0, segments=12, max_area=0.03):
    return RuppertsAlgorithm(_disk_pslg(radius, segments),
                             min_angle=30, max_area=max_area).refine()


def test_ruppert_attaches_curves_and_projects_boundary_nodes():
    radius = 1.0
    mesh = _disk_mesh(radius, segments=12)

    assert mesh.boundary_curves is not None
    assert all(curve is not None for curve in mesh.boundary_curves)

    # Every P1 boundary vertex is on the circle: the sampled corners are, and Ruppert's
    # split points were projected onto it rather than left at chord midpoints.
    rim_r = np.hypot(*mesh.vertices[mesh.boundary_idxs].T)
    assert np.abs(rim_r - radius).max() < 1e-9

    # The isoparametric edge nodes land on the circle too.
    space = FunctionSpace(mesh, IsoparametricTriangleElement, n_components=1)
    node_r = np.hypot(*space.node_coords[np.unique(space.boundary_nodes)].T)
    assert np.abs(node_r - radius).max() < 1e-9


def test_red_green_refinement_keeps_the_boundary_on_the_curve():
    radius = 1.0
    mesh = _disk_mesh(radius, segments=12)
    refined = RedGreenRefiner(mesh).refine(list(range(len(mesh.elements))))

    assert refined.boundary_curves is not None
    # It subdivided the boundary, and the new vertices sit on the circle, not on chords.
    assert len(refined.boundary) > len(mesh.boundary)
    rim_r = np.hypot(*refined.vertices[refined.boundary_idxs].T)
    assert np.abs(rim_r - radius).max() < 1e-9


def _kirsch_peak_sigma_xx(element_type, segments):
    length, height, radius = 20.0, 10.0, 1.0
    cx, cy = length / 2, height / 2
    pslg = _plate_with_hole_pslg(length, height, radius, segments)
    mesh = RuppertsAlgorithm(pslg, min_angle=28, max_area=0.02 * pslg.area()).refine()

    space = FunctionSpace(mesh, element_type, n_components=2)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, None])                       # roller
    bc.add(BCType.DIRICHLET, intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0])
    bc.add(BCType.NEUMANN, on_plane(0, length), [1.0, 0])                       # tension S = 1
    operator = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))
    u = LinearSolve().solve(LinearProblem(space, operator, None, bc))
    solution = ElasticSolution.from_solve(space, u, operator)

    distance = np.hypot(space.node_coords[:, 0] - cx, space.node_coords[:, 1] - cy)
    on_rim = np.abs(distance - radius) < 0.05
    rim_elements = np.flatnonzero(on_rim[space.element_nodes].any(axis=1))
    return float(solution.stress[rim_elements, 0, 0].max())


def test_curved_hole_resolves_the_stress_concentration_at_a_coarse_sampling():
    """Kirsch: a plate with a hole under tension S carries a hoop stress ~3S at the rim.
    At a coarse 16-gon hole the straight facets under-resolve it; the curved element,
    reading a true circle, recovers much more of the concentration on the same mesh."""
    straight = _kirsch_peak_sigma_xx(QuadraticTriangleElement, 16)
    curved = _kirsch_peak_sigma_xx(IsoparametricTriangleElement, 16)

    kt = 3.0
    assert curved > straight + 0.2, f"curved {curved:.3f} not clearly above straight {straight:.3f}"
    assert abs(curved - kt) < abs(straight - kt)
    assert 2.4 < curved < 3.4, f"curved peak {curved:.3f} outside a sane band around Kt=3"
