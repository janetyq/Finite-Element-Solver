"""Curved outlines through the meshing pipeline: PSLG -> Ruppert -> refinement.

A PSLG loop that carries a `Circle` produces a mesh whose rim facets carry that circle,
so Ruppert's split points and the red-green midpoints land on the true curve and an
isoparametric solve reads a round boundary. The Kirsch test is the physical payoff.
"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.forms import LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.curves import Arc, Circle
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


def test_per_segment_curve_tags_only_the_arc_of_a_mixed_outline():
    """A loop that is part straight, part arc (a filleted corner) curves only its arc
    segments: the projection follows the arc but leaves the straight edges alone."""
    radius = 0.2
    cx, cy = 1 - radius, 1 - radius
    theta = np.linspace(0.0, np.pi / 2, 6)   # rounds the top-right corner of a unit square
    arc_points = np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
    points = np.array([[0.0, 0.0], [1.0, 0.0], *arc_points.tolist(), [0.0, 1.0]])

    arc = Arc([cx, cy], radius, 0.0, np.pi / 2)
    point_curves = [None, None, *([arc] * len(arc_points)), None]
    pslg = PSLG.from_loops([points], curves=[point_curves])

    # Only the segments between consecutive arc points carry the curve.
    assert sum(c is not None for c in pslg.segment_curves) == len(arc_points) - 1

    mesh = RuppertsAlgorithm(pslg, min_angle=30, max_area=0.02).refine()
    curved = [facet for facet, curve in zip(mesh.boundary, mesh.boundary_curves)
              if curve is not None]
    assert curved, "no boundary facet inherited the arc"
    for facet in curved:
        r = np.hypot(mesh.vertices[facet, 0] - cx, mesh.vertices[facet, 1] - cy)
        assert np.all(np.abs(r - radius) < 1e-9)
    # A square corner is untouched: straight edges are not projected onto the arc.
    assert np.hypot(*mesh.vertices.T).min() < 1e-12   # (0, 0) is still exactly a vertex


def _kirsch_rim_sigma_xx(element_type, segments):
    """sigma_xx at the top of the hole, read by `nodal_stress` at the rim node itself,
    with the per-element (centroid) stress nearest the rim for comparison."""
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

    xy = space.node_coords
    top = np.flatnonzero(np.isclose(xy[:, 0], cx) & np.isclose(xy[:, 1], cy + radius))
    assert len(top) == 1
    rim = {method: float(solution.nodal_stress(method)[top[0], 0, 0])
           for method in ('average', 'l2')}
    rim_elements = np.flatnonzero((space.element_nodes == top[0]).any(axis=1))
    rim['element'] = float(solution.stress[rim_elements, 0, 0].max())
    return rim


@pytest.mark.parametrize('element_type', [IsoparametricTriangleElement, QuadraticTriangleElement])
def test_hole_stress_concentration_is_read_from_the_rim(element_type):
    """Kirsch: a plate with a hole under tension S carries a hoop stress ~3S at the rim
    (3.14S at this hole/height of 0.2, Howland's value). On a coarse 16-gon mesh the P2
    nodal stress at the rim node gets within 12% of that on the curved and straight
    element alike; the per-element (centroid) stress sits far lower."""
    reference = 3.14
    rim = _kirsch_rim_sigma_xx(element_type, 16)
    for method in ('average', 'l2'):
        assert abs(rim[method] - reference) < 0.12 * reference, (method, rim[method])
    assert abs(rim['average'] - rim['l2']) < 0.05 * reference
    assert rim['element'] < rim['average'] - 0.3
