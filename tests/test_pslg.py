"""`PSLG`, the sampled graph: its checks, its area, the crossing search, and the
boundary tags a mesh of it carries."""
import numpy as np
import pytest

from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.mesh.curves import Circle
from fem.mesh.mesh import Mesh
from fem.mesh.outline import Outline
from fem.mesh.pslg import PSLG, _find_crossing_segments
from fem.mesh.structured import box_mesh
from fem.regions import on_tag
from fem.space import FunctionSpace
from fem.elements import QuadraticTriangleElement

SQUARE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])


def _plate_with_hole() -> PSLG:
    return Outline([Outline.from_polygons([SQUARE]).loops[0],
                    (Circle([2.0, 2.0], 0.8),)]).sample(resolution=0.1)


# --- the graph ---

def test_a_bare_vertex_list_is_one_closed_loop():
    graph = PSLG(SQUARE)
    np.testing.assert_array_equal(graph.segments, [[0, 1], [1, 2], [2, 3], [3, 0]])
    assert graph.loop_ids.tolist() == [0, 0, 0, 0]
    assert all(c is None for c in graph.segment_curves)
    assert repr(graph) == 'PSLG(4 vertices, 4 segments, 1 loops)'


def test_pslg_is_immutable():
    graph = PSLG(SQUARE)
    with pytest.raises(ValueError):
        graph.vertices[0] = [9.0, 9.0]
    with pytest.raises(AttributeError):
        graph.vertices = SQUARE  # type: ignore[misc]


def test_per_segment_data_must_match_the_segments():
    with pytest.raises(ValueError, match='one entry per segment'):
        PSLG(SQUARE, loop_ids=[0, 0])


# --- meshing ---

def test_mesh_accepts_an_area_fraction_and_refuses_both_caps():
    graph = PSLG(SQUARE)
    coarse = graph.mesh(min_angle=25)
    fine = graph.mesh(min_angle=25, max_area_fraction=0.01)
    assert fine.n_elements > coarse.n_elements
    assert fine.element_measures.max() <= 0.01 * graph.area() + 1e-12
    with pytest.raises(ValueError):
        graph.mesh(max_area=1.0, max_area_fraction=0.1)


def test_mesh_validates_first():
    bowtie = PSLG(np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match='cross'):
        bowtie.mesh()


# --- crossing search ---

def _crossing_by_all_pairs(vertices, segments):
    '''The quadratic all-pairs reference the grid must match.'''
    vertices, segments = np.asarray(vertices, float), np.asarray(segments)

    def side(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if set(segments[i]) & set(segments[j]):
                continue
            ai, bi = vertices[segments[i]]
            aj, bj = vertices[segments[j]]
            if ((side(ai, bi, aj) > 0) != (side(ai, bi, bj) > 0)
                    and (side(aj, bj, ai) > 0) != (side(aj, bj, bi) > 0)):
                return (i, j)
    return None


def test_grid_crossing_detection_matches_the_all_pairs_reference():
    """The spatial-grid crossing search returns the pair the all-pairs scan does."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(2, 25))
        vertices = rng.uniform(0, 10, size=(2 * n, 2))
        segments = np.arange(2 * n).reshape(n, 2)
        assert _find_crossing_segments(vertices, segments) == _crossing_by_all_pairs(vertices, segments)


def test_grid_crossing_finds_a_crossing_among_far_apart_clusters():
    """A long spanning segment plus a distant crossing pair: the grid must still find
    the crossing even though the two clusters share no small cell."""
    vertices = np.array([
        [-100.0, 0.0], [100.0, 0.001],   # a long, near-horizontal spanning segment
        [0.0, -1.0], [0.0, 1.0],         # a short vertical segment that it crosses
        [50.0, 50.0], [51.0, 50.0],      # a far-off pair of parallel, non-crossing segments
        [50.0, 51.0], [51.0, 51.0],
    ])
    segments = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    assert _find_crossing_segments(vertices, segments) == (0, 1)


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
    resolved = Conditions(Neumann(on_tag(1), [1.0, 0.0])).resolve(FunctionSpace(mesh, n_components=2))
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
