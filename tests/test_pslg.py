"""`PSLG` construction and meshing, and the boundary tags a PSLG mesh carries."""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.mesh.curves import Arc, Circle
from fem.mesh.mesh import Mesh
from fem.mesh.pslg import PSLG
from fem.mesh.structured import box_mesh
from fem.regions import on_tag
from fem.space import FunctionSpace
from fem.elements import QuadraticTriangleElement

SQUARE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])


def _plate_with_hole() -> PSLG:
    hole = Circle([2.0, 2.0], 0.8)
    return PSLG.from_loops([SQUARE, hole.polygon(12)], curves=[None, hole])


# --- building ---

def test_circle_polygon_samples_the_circle_without_the_closing_repeat():
    points = Circle([1.0, -1.0], 2.0).polygon(8)
    assert points.shape == (8, 2)
    assert np.allclose(np.hypot(points[:, 0] - 1.0, points[:, 1] + 1.0), 2.0)
    assert not np.allclose(points[0], points[-1])


def test_arc_polygon_includes_both_endpoints():
    points = Arc([0.0, 0.0], 1.0, 0.0, np.pi / 2).polygon(5)
    assert np.allclose(points[0], [1.0, 0.0]) and np.allclose(points[-1], [0.0, 1.0])


def test_circle_pslg_is_one_curved_loop():
    pslg = PSLG.circle([0.0, 0.0], 1.0, 16)
    assert len(pslg.segments) == 16
    assert all(isinstance(curve, Circle) for curve in pslg.segment_curves)
    assert pslg.area() == pytest.approx(0.5 * 16 * np.sin(2 * np.pi / 16))


def test_pslg_is_immutable():
    pslg = PSLG.from_loops([SQUARE])
    with pytest.raises(ValueError):
        pslg.vertices[0] = [9.0, 9.0]
    with pytest.raises(AttributeError):
        pslg.vertices = SQUARE  # type: ignore[misc]


def test_with_bounding_box_returns_a_new_graph_with_the_box_as_its_own_loop():
    inner = PSLG.from_loops([SQUARE])
    boxed = inner.with_bounding_box(buffer=0.5)
    assert len(inner.segments) == 4, 'the source is untouched'
    assert len(boxed.segments) == 8
    assert set(np.unique(boxed.loop_ids)) == {0, 1}
    # The box encloses the square, which the even-odd rule then reads as a hole.
    assert boxed.area() == pytest.approx(8.0 * 8.0 - 16.0)


# --- meshing ---

def test_mesh_accepts_an_area_fraction_and_refuses_both_caps():
    pslg = PSLG.from_loops([SQUARE])
    coarse = pslg.mesh(min_angle=25)
    fine = pslg.mesh(min_angle=25, max_area_fraction=0.01)
    assert fine.n_elements > coarse.n_elements
    assert fine.element_measures.max() <= 0.01 * pslg.area() + 1e-12
    with pytest.raises(ValueError):
        pslg.mesh(max_area=1.0, max_area_fraction=0.1)


def test_mesh_validates_first():
    bowtie = PSLG(np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match='cross'):
        bowtie.mesh()


# --- boundary tags ---

def test_tags_survive_red_green_refinement():
    mesh = _plate_with_hole().mesh(min_angle=25, max_area_fraction=0.05)
    refined = mesh.refined()
    assert refined.boundary_tags is not None
    assert len(refined.boundary_tags) == len(refined.boundary) == 2 * len(mesh.boundary)
    rim = refined.boundary[refined.boundary_tags == 1]
    radii = np.hypot(*(refined.vertices[rim].reshape(-1, 2) - [2.0, 2.0]).T)
    assert np.allclose(radii, 0.8), 'the split halves keep the rim tag and stay on the rim'


def test_displaced_keeps_the_tags():
    mesh = _plate_with_hole().mesh(min_angle=25)
    assert mesh.boundary_tags is not None
    moved = mesh.displaced(np.ones((mesh.n_vertices, 2)))
    assert np.array_equal(moved.boundary_tags, mesh.boundary_tags)


def test_tags_must_match_the_facets():
    with pytest.raises(ValueError, match='boundary_tags'):
        Mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], boundary_tags=[0, 0])


# --- on_tag ---

def test_on_tag_selects_the_rim_nodes_and_nothing_else():
    mesh = _plate_with_hole().mesh(min_angle=25, max_area_fraction=0.02)
    idxs = Dirichlet(on_tag(1), 0.0).select(mesh)
    radii = np.hypot(*(mesh.vertices[idxs] - [2.0, 2.0]).T)
    assert len(idxs) > 0 and np.allclose(radii, 0.8)
    outer = Dirichlet(on_tag(0), 0.0).select(mesh)
    assert set(outer) | set(idxs) == set(mesh.boundary_idxs)
    assert not set(outer) & set(idxs)


def test_on_tag_resolves_a_neumann_condition_on_the_rim_facets_only():
    mesh = _plate_with_hole().mesh(min_angle=25, max_area_fraction=0.02)
    resolved = BoundaryConditions(Neumann(on_tag(1), [1.0, 0.0])).resolve(mesh, 2)
    mask = resolved.neumann[0].facet_mask
    assert mask.sum() == int(np.sum(mesh.boundary_tags == 1))


def test_on_tag_reaches_the_edge_nodes_of_a_p2_space():
    mesh = _plate_with_hole().mesh(min_angle=25, max_area_fraction=0.05)
    space = FunctionSpace(mesh, QuadraticTriangleElement)
    idxs = Dirichlet(on_tag(1), 0.0).select(space.nodes)
    assert len(idxs) == 2 * int(np.sum(mesh.boundary_tags == 1)), 'rim vertices and midpoints'


def test_on_tag_refuses_an_untagged_mesh_and_a_geometric_combination():
    plain = box_mesh([[0, 0], [1, 1]], (3, 3))
    with pytest.raises(ValueError, match='boundary_tags'):
        Dirichlet(on_tag(0), 0.0).select(plain)
    with pytest.raises(TypeError):
        on_tag(0)(plain.vertices)
