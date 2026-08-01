"""Ruppert's algorithm: the guarantees it makes about the mesh it returns.

The algorithm terminates only when no triangle is skinny and no segment is
encroached, and then keeps only what the PSLG encloses. Those are what it
promises a caller. The tests below assert them directly rather than pinning a
particular triangulation -- any correct refinement satisfies them, which keeps
the suite useful across changes to insertion order or the underlying Delaunay.
"""
import logging

import numpy as np
import pytest

from fem.geometry import calculate_polygon_area, calculate_triangle_min_angle, point_in_polygon
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import PSLG

L_SHAPE_OUTLINE = np.array([
    [0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
])
SQUARE_OUTLINE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SLAB_OUTLINE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 0.5], [0.0, 0.5]])
# A plate comfortably around the L-shape, sharing no vertex with it.
PLATE_OUTLINE = np.array([[-3.0, -3.0], [5.0, -3.0], [5.0, 5.0], [-3.0, 5.0]])

# The L-shape needs no angle refinement at any bound the algorithm can hold, so
# tests that need refinement to have actually happened drive it with an area cap.
REFINING_AREA = 0.05


def _l_shape() -> PSLG:
    """A non-convex outline. The reflex corner is the point: the convex hull spans
    it, so refinement produces triangles outside the domain that have to go."""
    return PSLG(L_SHAPE_OUTLINE.copy())


def _thin_slab() -> PSLG:
    """A convex outline whose plain triangulation has ~14 degree angles, so the
    angle bound is what drives refinement here rather than the area cap."""
    return PSLG(SLAB_OUTLINE.copy())


def _min_angles(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    return np.array([calculate_triangle_min_angle(vertices[element]) for element in mesh.elements])


@pytest.mark.parametrize('min_angle', [15, 20, 25, 30])
def test_every_triangle_meets_the_angle_bound(min_angle):
    """The headline guarantee: no element is skinnier than what was asked for."""
    algo = RuppertsAlgorithm(_thin_slab(), min_angle=min_angle)
    mesh = algo.refine()

    angles = _min_angles(mesh)
    assert angles.min() >= min_angle, (
        f'{(angles < min_angle).sum()} of {len(angles)} elements are below the '
        f'{min_angle} degree bound, worst {angles.min():.2f}'
    )


def test_no_segment_is_encroached_on_return():
    """The other half of termination, and what makes the mesh conform to the
    outline: every segment's diametral circle is empty, so each survives as an
    edge of the Delaunay triangulation."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=REFINING_AREA)
    algo.refine()

    assert algo.get_encroached_segments() == []


def test_refinement_preserves_the_input_geometry():
    """Refinement only ever appends vertices and splits segments at their
    midpoint, so input vertices keep their indices and the final segments still
    trace the input outline exactly -- no corner is cut or moved."""
    pslg = _l_shape()
    original_vertices = np.array(pslg.vertices)
    original_segments = np.array(pslg.segments)
    algo = RuppertsAlgorithm(pslg, min_angle=20, max_area=REFINING_AREA)
    mesh = algo.refine()

    # Every input corner is still a node of the mesh, at exactly its input position.
    kept = {tuple(vertex) for vertex in np.asarray(mesh.vertices)}
    assert {tuple(vertex) for vertex in original_vertices} <= kept

    for start, end in algo.segments:
        # The segment must be a sub-interval of one of the segments it descends
        # from: collinear with it, and inside it.
        for original in original_segments:
            edge = original_vertices[original[1]] - original_vertices[original[0]]
            offsets = algo.vertices[[start, end]] - original_vertices[original[0]]
            fractions = offsets @ edge / (edge @ edge)
            collinear = np.allclose(edge[0]*offsets[:, 1] - edge[1]*offsets[:, 0], 0, atol=1e-9)
            if collinear and np.all((fractions > -1e-9) & (fractions < 1 + 1e-9)):
                break
        else:
            pytest.fail(f'segment {start}-{end} does not lie on any input segment')


def test_segments_survive_as_edges_of_the_triangulation():
    """Conformity: the mesh resolves the outline instead of cutting across it.

    Compared in coordinates, because the returned mesh is renumbered onto the
    vertices it kept while `algo.segments` still indexes the full working set.
    """
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=REFINING_AREA)
    mesh = algo.refine()

    vertices = np.asarray(mesh.vertices)
    edges = set()
    for element in mesh.elements:
        for i in range(3):
            ends = sorted((tuple(vertices[element[i]]), tuple(vertices[element[(i + 1) % 3]])))
            edges.add(tuple(ends))

    missing = [
        tuple(segment) for segment in algo.segments
        if tuple(sorted((tuple(algo.vertices[segment[0]]), tuple(algo.vertices[segment[1]]))))
        not in edges
    ]
    assert not missing, f'{len(missing)} segments are not edges of the mesh: {missing[:5]}'


def test_max_area_bounds_every_element_without_losing_the_angle_bound():
    """Both criteria are refined through one queue, so satisfying the area cap
    must not leave skinny elements behind."""
    max_area, min_angle = 0.2, 20
    algo = RuppertsAlgorithm(_l_shape(), min_angle=min_angle, max_area=max_area)
    mesh = algo.refine()

    vertices = np.asarray(mesh.vertices)
    areas = np.array([calculate_polygon_area(vertices[element]) for element in mesh.elements])
    assert areas.max() <= max_area
    assert _min_angles(mesh).min() >= min_angle


def test_the_area_cap_does_not_break_conformity():
    """Regression: area refinement used to insert circumcenters without checking
    encroachment, which let a segment stop being an edge of the mesh. The domain
    then leaked into the exterior and the whole triangulation was discarded."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=REFINING_AREA)
    mesh = algo.refine()

    assert algo.get_encroached_segments() == []
    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.elements)].mean(axis=1)
    assert all(point_in_polygon(c, L_SHAPE_OUTLINE) for c in centroids)


def test_a_finer_angle_bound_costs_more_triangles():
    """Sanity check that `min_angle` is actually driving the refinement."""
    coarse = RuppertsAlgorithm(_thin_slab(), min_angle=10).refine()
    fine = RuppertsAlgorithm(_thin_slab(), min_angle=30).refine()

    assert len(fine.elements) > len(coarse.elements)


def test_mesh_covers_the_outline_and_nothing_else():
    """A Delaunay triangulation spans the convex hull of its vertices, so the
    L-shape's notch gets filled with triangles that are not in the domain. They
    must not survive into the mesh."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20)
    mesh = algo.refine()

    vertices = np.asarray(mesh.vertices)
    centroids = vertices[np.asarray(mesh.elements)].mean(axis=1)
    outside = [c for c in centroids if not point_in_polygon(c, L_SHAPE_OUTLINE)]
    assert not outside, f'{len(outside)} of {len(centroids)} elements lie outside the outline'

    # And the notch really was filled, so the discard did some work.
    assert algo.get_exterior_triangles().any()


def test_convex_outline_keeps_everything():
    """The seeding case that is easy to get wrong: when the outline *is* the
    convex hull, every hull edge is a segment, so nothing can be reached from
    outside and the whole triangulation is the domain."""
    algo = RuppertsAlgorithm(PSLG(SQUARE_OUTLINE.copy()), min_angle=20)
    mesh = algo.refine()

    assert not algo.get_exterior_triangles().any()
    assert len(mesh.elements) == len(algo.triangulation.simplices)


def test_returned_mesh_carries_a_usable_boundary():
    """A mesh with no boundary facets cannot take a boundary condition, which is
    what the refinement used to hand back."""
    mesh = RuppertsAlgorithm(_l_shape(), min_angle=20).refine()

    assert len(mesh.boundary) > 0
    # A closed boundary: every boundary edge is used once, and every boundary
    # vertex joins exactly two of them.
    counts = np.bincount(np.asarray(mesh.boundary).ravel())
    assert set(np.unique(counts[counts > 0])) == {2}
    # Interior edges belong to two elements, boundary edges to one.
    single = [edge for edge, elements in mesh.edge_to_elements.items() if len(elements) == 1]
    assert len(single) == len(mesh.boundary)


def test_boundary_vertices_lie_on_the_outline():
    """The boundary the mesh reports is the outline it was given, not the hull."""
    mesh = RuppertsAlgorithm(_l_shape(), min_angle=20).refine()

    vertices = np.asarray(mesh.vertices)
    for vertex in vertices[mesh.boundary_idxs]:
        on_outline = False
        for start, end in zip(L_SHAPE_OUTLINE, np.roll(L_SHAPE_OUTLINE, -1, axis=0)):
            edge, offset = end - start, vertex - start
            fraction = offset @ edge / (edge @ edge)
            if (abs(edge[0]*offset[1] - edge[1]*offset[0]) < 1e-9
                    and -1e-9 < fraction < 1 + 1e-9):
                on_outline = True
                break
        assert on_outline, f'boundary vertex {vertex} is not on the outline'


def test_a_pslg_that_encloses_nothing_is_refused():
    """An open polyline bounds no region; every triangle is reachable from
    outside. Better to say so than to return an empty mesh."""
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
    pslg = PSLG(line, segments=np.array([[0, 1], [1, 2]]))

    with pytest.raises(ValueError, match='encloses no region'):
        RuppertsAlgorithm(pslg, min_angle=20).refine()


def test_a_loop_inside_another_is_a_hole():
    """The even-odd rule at work, and the shape a flow-around-an-obstacle problem
    needs: a box around an outline meshes the material between them and leaves
    the outline itself empty."""
    pslg = _l_shape()
    pslg.add_bounding_box(buffer=0.2)
    mesh = RuppertsAlgorithm(pslg, min_angle=20).refine()

    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.elements)].mean(axis=1)
    inside_the_hole = [c for c in centroids if point_in_polygon(c, L_SHAPE_OUTLINE)]
    assert not inside_the_hole, f'{len(inside_the_hole)} elements fill what should be a hole'
    assert len(mesh.elements) > 0

    # The hole has a rim, so the boundary is the box plus the outline -- two loops,
    # every vertex still joining exactly two boundary edges.
    counts = np.bincount(np.asarray(mesh.boundary).ravel())
    assert set(np.unique(counts[counts > 0])) == {2}


def test_disjoint_loops_are_both_meshed():
    """Two outlines side by side are two pieces of one domain, not a hole."""
    left = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    right = left + np.array([3.0, 0.0])
    mesh = RuppertsAlgorithm(PSLG.from_loops([left, right]), min_angle=20).refine()

    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.elements)].mean(axis=1)
    assert any(point_in_polygon(c, left) for c in centroids)
    assert any(point_in_polygon(c, right) for c in centroids)
    # Nothing in the gap between them.
    assert all(point_in_polygon(c, left) or point_in_polygon(c, right) for c in centroids)


def test_pslg_area_subtracts_holes():
    """Callers scale a max_area fraction against this, so a hole counted as
    material would loosen the cap by however large the hole is."""
    plate = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    hole = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])

    assert PSLG.from_loops([plate]).area() == pytest.approx(12.0)
    assert PSLG.from_loops([plate, hole]).area() == pytest.approx(11.0)
    # Which loop was drawn first is not part of the answer.
    assert PSLG.from_loops([hole, plate]).area() == pytest.approx(11.0)
    # Side by side, neither encloses the other, so both are material.
    assert PSLG.from_loops([plate, plate + [5.0, 0.0]]).area() == pytest.approx(24.0)
    # An island in the hole is enclosed twice over, so it is material again.
    island = np.array([[1.4, 1.4], [1.6, 1.4], [1.6, 1.6], [1.4, 1.6]])
    assert PSLG.from_loops([plate, hole, island]).area() == pytest.approx(11.04)


def test_pslg_area_matches_what_refinement_fills():
    """The even-odd rule decides both what `area` reports and which triangles
    survive refinement, so the two readings of the same loops have to agree."""
    plate = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    hole = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    pslg = PSLG.from_loops([plate, hole])
    expected = pslg.area()

    mesh = RuppertsAlgorithm(pslg, min_angle=20).refine()
    vertices, elements = np.asarray(mesh.vertices), np.asarray(mesh.elements)
    filled = sum(calculate_polygon_area(vertices[element]) for element in elements)
    assert filled == pytest.approx(expected)


def test_boundary_facets_name_the_loop_they_came_from():
    """A plate with a hole: the outer wall and the hole rim are both boundary,
    and a solver has to tell them apart to put different conditions on them.
    That is unrecoverable from the finished mesh, so meshing has to record it."""
    pslg = PSLG.from_loops([PLATE_OUTLINE, L_SHAPE_OUTLINE])
    algo = RuppertsAlgorithm(pslg, min_angle=20)
    mesh = algo.refine()

    assert len(algo.boundary_loops) == len(mesh.boundary)
    assert set(np.unique(algo.boundary_loops)) == {0, 1}, 'both loops should appear'

    # Loop 1 is the hole; every facet attributed to it must sit on that outline.
    vertices = np.asarray(mesh.vertices)
    for facet in np.asarray(mesh.boundary)[algo.boundary_loops == 1]:
        midpoint = vertices[facet].mean(axis=0)
        on_outline = min(
            abs((end - start)[0]*(midpoint - start)[1] - (end - start)[1]*(midpoint - start)[0])
            / np.linalg.norm(end - start)
            for start, end in zip(L_SHAPE_OUTLINE, np.roll(L_SHAPE_OUTLINE, -1, axis=0))
        )
        assert on_outline < 1e-9, f'facet attributed to the hole is not on it: {midpoint}'


def test_split_segments_keep_their_loop():
    """Refinement splits a segment many times over; the halves have to carry the
    attribution or long boundaries lose it."""
    pslg = PSLG.from_loops([PLATE_OUTLINE, L_SHAPE_OUTLINE])
    algo = RuppertsAlgorithm(pslg, min_angle=20, max_area=1.0)
    algo.refine()

    assert len(algo.segments) > len(pslg.segments), 'expected splitting to have happened'
    assert len(algo.segment_loops) == len(algo.segments)
    assert set(np.unique(algo.segment_loops)) == {0, 1}


def test_sharp_input_corners_are_reported(caplog):
    """A sub-60-degree corner is what turns refinement into an apparent hang, so
    it should say so up front rather than leave the caller guessing."""
    spike = np.array([[0.0, 0.0], [10.0, 0.3], [10.0, -0.3]])
    with caplog.at_level(logging.WARNING, logger='fem.mesh.generation'):
        RuppertsAlgorithm(PSLG(spike), min_angle=20)

    assert 'below the 60' in caplog.text


def test_a_square_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger='fem.mesh.generation'):
        RuppertsAlgorithm(PSLG(SQUARE_OUTLINE.copy()), min_angle=20)

    assert caplog.text == ''


def test_crossing_segments_are_refused():
    """A bow-tie is not a planar straight-line graph. Meshing it does not fail,
    it silently meshes the wrong region."""
    bowtie = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match='cross'):
        PSLG(bowtie).validate()


def test_a_valid_outline_passes_validation():
    PSLG(L_SHAPE_OUTLINE.copy()).validate()
    PSLG.from_loops([PLATE_OUTLINE, L_SHAPE_OUTLINE]).validate()


def test_repeated_vertices_are_refused():
    """Two vertices at one place cannot both be triangulated -- qhull keeps one,
    so a segment ending on the other never becomes an edge of the mesh."""
    repeated = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match='more than once'):
        PSLG(repeated).validate()


def test_zero_length_segments_are_refused():
    doubled_back = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match='zero length'):
        PSLG(doubled_back).validate()


def test_min_angle_is_the_same_computed_singly_or_in_a_batch():
    """`calculate_triangle_min_angle` takes a stacked array so the refinement
    loop can test every triangle at once; the two paths must agree."""
    rng = np.random.default_rng(0)
    triangles = rng.random((25, 3, 2))

    batch = calculate_triangle_min_angle(triangles)
    singly = np.array([calculate_triangle_min_angle(t) for t in triangles])

    assert batch.shape == (25,)
    np.testing.assert_allclose(batch, singly)


def test_degenerate_triangle_has_a_zero_angle():
    """Three collinear points are a real possibility in a refinement loop, and
    must read as the worst possible triangle rather than as a NaN that compares
    false against the angle bound."""
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert calculate_triangle_min_angle(collinear) == pytest.approx(0.0)
