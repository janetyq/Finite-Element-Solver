"""`Outline`: loops of pieces, joined end to end, sampled into a `PSLG` only when
meshed, and simplified along their straight runs."""
import numpy as np
import pytest

from fem.mesh.curves import Arc, Circle, CubicBezier, Line
from fem.mesh.outline import Outline, douglas_peucker
from fem.mesh.pslg import PSLG

SQUARE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])


def _plate_with_hole(radius=0.8):
    return Outline([Outline.from_polygons([SQUARE]).loops[0], (Circle([2.0, 2.0], radius),)])


# --- building ---

def test_from_polygons_is_a_loop_of_lines_per_polygon():
    outline = Outline.from_polygons([SQUARE, SQUARE + 10.0])
    assert len(outline.loops) == 2 and outline.n_pieces == 8
    assert all(isinstance(p, Line) for loop in outline.loops for p in loop)
    assert np.allclose(outline.loops[0][3].end, SQUARE[0]), 'the last line closes the loop'


def test_loops_must_join_end_to_end():
    with pytest.raises(ValueError, match='piece 0 ends at'):
        Outline([(Line([0, 0], [1, 0]), Line([2, 0], [0, 1]), Line([0, 1], [0, 0]))])
    with pytest.raises(ValueError, match='a loop of its own'):
        Outline([(Circle([0, 0], 1.0), Line([1, 0], [2, 0]))])
    with pytest.raises(ValueError, match='at least one loop'):
        Outline([])


def test_a_circle_is_a_loop_of_one():
    outline = Outline([Circle([0.0, 0.0], 1.0)])
    assert outline.loops == ((outline.loops[0][0],),)
    assert repr(outline) == 'Outline(1 loops, 1 pieces)'


def test_with_bounding_box_adds_the_box_as_its_own_loop():
    inner = Outline.from_polygons([SQUARE])
    boxed = inner.with_bounding_box(buffer=0.5)
    assert len(inner.loops) == 1, 'the source is untouched'
    assert len(boxed.loops) == 2 and len(boxed.loops[1]) == 4
    # The box encloses the square, which the even-odd rule then reads as a hole.
    assert boxed.area() == pytest.approx(8.0 * 8.0 - 16.0)


# --- sampling ---

def test_sampling_straight_loops_reproduces_the_polygon():
    graph = Outline.from_polygons([SQUARE]).sample()
    assert isinstance(graph, PSLG)
    np.testing.assert_array_equal(graph.vertices, SQUARE)
    np.testing.assert_array_equal(graph.segments, [[0, 1], [1, 2], [2, 3], [3, 0]])
    assert all(c is None for c in graph.segment_curves)
    assert graph.loop_ids.tolist() == [0, 0, 0, 0]


def test_curved_pieces_sample_to_chords_that_carry_them():
    outline = _plate_with_hole()
    graph = outline.sample(resolution=0.05)
    hole = outline.loops[1][0]
    curved = [c for c in graph.segment_curves if c is not None]
    assert curved and all(c is hole for c in curved)
    assert set(graph.loop_ids.tolist()) == {0, 1}
    rim = graph.vertices[graph.loop_ids == 1] if False else graph.vertices[4:]
    assert np.allclose(np.hypot(*(rim - [2.0, 2.0]).T), 0.8)


def test_resolution_sets_the_chord_count_of_a_curve():
    circle = Outline([Circle([0.0, 0.0], 1.0)])
    coarse, fine = circle.sample(resolution=0.2), circle.sample(resolution=0.02)
    assert len(fine.segments) > len(coarse.segments) >= 8
    assert len(fine.segments) == pytest.approx(2 * np.pi / (0.02 * 2.0), abs=1)
    line = Outline.from_polygons([SQUARE]).sample(resolution=0.001)
    assert len(line.segments) == 4, 'a line is never subdivided'
    with pytest.raises(ValueError, match='resolution'):
        circle.sample(resolution=0.0)


def test_pieces_share_their_join_vertex_and_a_reversed_arc_joins_clockwise():
    r = 0.5
    fillet = Arc([1.0 + r, 1.0 + r], r, np.pi, 1.5 * np.pi).reversed()
    loop = (Line([0, 0], [3, 0]), Line([3, 0], [3, 1]), Line([3, 1], fillet.start), fillet,
            Line(fillet.end, [1, 3]), Line([1, 3], [0, 3]), Line([0, 3], [0, 0]))
    graph = Outline([loop]).sample(resolution=0.1)
    assert len(np.unique(graph.vertices, axis=0)) == len(graph.vertices), 'no doubled vertex'
    arc_chords = [c for c in graph.segment_curves if isinstance(c, Arc)]
    assert len(arc_chords) >= 2
    graph.validate()


# --- simplifying ---

def test_simplified_drops_vertices_on_straight_runs_but_keeps_curved_pieces():
    t = np.linspace(0.0, 1.0, 40)
    wobble = np.column_stack([4.0 * t, 0.001 * np.sin(30.0 * t)])
    run = [Line(wobble[i], wobble[i + 1]) for i in range(len(wobble) - 1)]
    arc = Arc([4.0, 1.0], 1.0, -np.pi / 2, np.pi / 2)
    outline = Outline([run + [Line(wobble[-1], arc.start), arc,
                              Line(arc.end, [0.0, 2.0]), Line([0.0, 2.0], [0.0, 0.0])]])
    simplified = outline.simplified(0.01)
    assert simplified.n_pieces == 4
    assert any(p is arc for p in simplified.loops[0]), 'the arc is kept as it was'
    assert all(np.allclose(p.end, q.start) for p, q in zip(simplified.loops[0],
                                                            simplified.loops[0][1:], strict=False))


def test_simplifying_an_all_line_loop_matches_douglas_peucker():
    rng = np.random.default_rng(3)
    angles = np.sort(rng.uniform(0, 2 * np.pi, 60))
    ring = np.column_stack([np.cos(angles), np.sin(angles)]) * (1 + 0.01 * rng.standard_normal(60))[:, None]
    simplified = Outline.from_polygons([ring]).simplified(0.02).sample()
    expected = douglas_peucker(ring, 0.02 * 2.0)
    np.testing.assert_allclose(simplified.vertices, expected)


# --- meshing ---

def test_mesh_carries_loop_tags_and_the_hole_circle_onto_the_facets():
    outline = _plate_with_hole()
    mesh = outline.mesh(min_angle=25, max_area_fraction=0.05)
    assert mesh.boundary_tags is not None and set(mesh.boundary_tags.tolist()) == {0, 1}
    assert mesh.boundary_curves is not None
    hole = outline.loops[1][0]
    rim = mesh.boundary_tags == 1
    assert all(curve is hole for curve, on_rim in zip(mesh.boundary_curves, rim, strict=True) if on_rim)
    assert all(curve is None for curve, on_rim in zip(mesh.boundary_curves, rim, strict=True) if not on_rim)
    radii = np.hypot(*(mesh.vertices[mesh.boundary[rim]].reshape(-1, 2) - [2.0, 2.0]).T)
    assert np.allclose(radii, 0.8), 'split points were projected onto the circle'
    assert mesh.min_angle >= 25 - 1e-9


def test_a_bezier_loop_meshes_onto_its_curve():
    top = CubicBezier([0, 0], [0, 4], [4, 4], [4, 0])
    outline = Outline([(top, Line([4, 0], [4, -3]), Line([4, -3], [0, -3]), Line([0, -3], [0, 0]))])
    mesh = outline.mesh(min_angle=25, max_area_fraction=0.05)
    curved = [f for f, c in zip(mesh.boundary, mesh.boundary_curves, strict=True) if c is top]
    assert curved
    for facet in curved:
        ends = mesh.vertices[facet]
        assert np.linalg.norm(top.project(ends) - ends, axis=1).max() < 1e-9
