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
from scipy.spatial import Delaunay, QhullError

from fem.geometry import (
    calculate_circumcenter,
    calculate_polygon_area,
    calculate_triangle_min_angle,
    point_in_polygon,
)
from fem.mesh.ruppert import ENCROACHMENT_TOLERANCE, RuppertsAlgorithm
from fem.mesh.svg import PSLG

L_SHAPE_OUTLINE = np.array([
    [0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
])
SQUARE_OUTLINE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SLAB_OUTLINE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 0.5], [0.0, 0.5]])
# A plate comfortably around the L-shape, sharing no vertex with it.
PLATE_OUTLINE = np.array([[-3.0, -3.0], [5.0, -3.0], [5.0, 5.0], [-3.0, 5.0]])

# `files/cloud.svg` simplified at tolerance 0.02, from the backlog's qhull-precision
# report. Its tight curvature makes qhull fan a segment split's collinear triple into a
# zero-area sliver that used to crash refinement. Regular shapes do not reproduce it, so
# the exact coordinates are pinned here rather than generated.
CLOUD_OUTLINE = np.array([
    [3.0000, 786.3507], [3.2753, 784.6595], [4.5816, 782.5672], [6.6932, 781.2728],
    [8.4000, 781.0000], [17.9224, 781.2296], [20.1318, 782.8436], [21.0000, 785.5031],
    [20.5073, 787.5956], [18.3000, 789.7500], [17.1148, 792.8627], [15.1901, 794.4193],
    [13.5696, 794.9320], [11.3332, 794.8293], [9.5355, 794.0051], [7.5000, 791.5000],
    [4.2375, 789.6797], [3.0531, 787.0782],
])

# A slender L-bracket (arm 4, limb width 1.2) with a sharp re-entrant corner. Its long
# axis-aligned edges and the circumcenters refinement inserts along the corner are
# nearly cocircular, which used to trip qhull's *incremental* insertion with a "wide
# merge" precision error partway through a run. Distinct from CLOUD_OUTLINE's failure,
# which was a segment-split sliver; this one is in add_points, and the coordinates are
# pinned because a regular shape does not reproduce it.
L_BRACKET_OUTLINE = np.array([
    [0.0, 0.0], [4.0, 0.0], [4.0, 1.2], [1.2, 1.2], [1.2, 4.0], [0.0, 4.0],
])

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


def test_growing_the_triangulation_keeps_it_delaunay():
    """Refinement adds points to the triangulation instead of rebuilding it from
    scratch, so the property everything else rests on has to survive the growing:
    no vertex strictly inside any triangle's circumcircle.

    Not the same triangulation as a rebuild, though, and it cannot be asserted to
    be one. Refinement inserts circumcentres, so cocircular quadrilaterals are
    everywhere, and either diagonal of one is Delaunay.
    """
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=REFINING_AREA)
    algo.refine()

    vertices = np.asarray(algo.vertices)
    simplices = algo.triangulation.simplices
    centres = calculate_circumcenter(vertices[simplices])
    radii = np.linalg.norm(vertices[simplices][:, 0] - centres, axis=-1)
    distances = np.linalg.norm(vertices[None, :, :] - centres[:, None, :], axis=-1)
    # A triangle's own corners sit exactly on its circumcircle.
    np.put_along_axis(distances, simplices, np.inf, axis=1)

    assert (distances >= radii[:, None] * (1 - 1e-9)).all(), (
        f'{(distances < radii[:, None] * (1 - 1e-9)).any(axis=1).sum()} triangles have a '
        'vertex inside their circumcircle'
    )
    assert len(simplices) == len(Delaunay(vertices).simplices)


def test_an_outline_qhull_cannot_start_incrementally_from_still_meshes():
    """Incremental mode needs a non-degenerate initial simplex, and four
    cocircular points do not give it one -- a rectangle is the common case, not
    a corner one. Such a run rebuilds until qhull will take the point set."""
    algo = RuppertsAlgorithm(_thin_slab(), min_angle=25)
    with pytest.raises(QhullError):
        Delaunay(SLAB_OUTLINE, incremental=True)

    mesh = algo.refine()

    assert _min_angles(mesh).min() >= 25
    assert algo._incremental, 'the run never got off the rebuild path'


def test_no_segment_is_encroached_on_return():
    """The other half of termination, and what makes the mesh conform to the
    outline: every segment's diametral circle is empty, so each survives as an
    edge of the Delaunay triangulation."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=REFINING_AREA)
    algo.refine()

    assert len(algo.get_encroached_segments()) == 0


def test_nothing_bad_survives_the_queue():
    """Bad triangles are refined off a queue topped up per insertion rather than
    rescanned each pass, so the mesh could come back with skinny elements the
    queue lost track of. A full scan has to find nothing left."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=25, max_area=REFINING_AREA)
    algo.refine()

    assert len(algo.get_bad_triangles()) == 0


def test_a_segment_between_two_sharp_corners_lands_on_shells_at_both():
    """Segments off a sharp corner split at power-of-two distances from it, which
    is what eventually makes two of them equidistant and stops the cascade that
    would otherwise refine into the corner forever.

    `_split_point` ladders from one end per split, so a segment sharp at *both*
    ends only works because the half left beside the other corner ladders from
    there next time -- which holds because the midpoint is always the newest
    vertex, and so never the lower index. That is implicit and easy to break by
    renumbering, so it is pinned here rather than left to the docstring.
    """
    # A sliver: the base runs between two 3.4 degree corners.
    sliver = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.3]])
    algo = RuppertsAlgorithm(PSLG(sliver.copy()), min_angle=25)
    assert any(int(a) in algo.sharp_vertices and int(b) in algo.sharp_vertices
               for a, b in algo.segments), 'the input no longer has a both-sharp segment'

    algo.refine()

    checked = 0
    for segment in algo.segments:
        for near, far in ((segment[0], segment[1]), (segment[1], segment[0])):
            # Only stubs that a split actually produced; an untouched input
            # segment keeps whatever length it was drawn with.
            if int(near) not in algo.sharp_vertices or int(far) < len(sliver):
                continue
            checked += 1
            distance = np.linalg.norm(algo.vertices[far] - algo.vertices[near])
            exponent = np.log2(distance)
            assert exponent == pytest.approx(round(exponent), abs=1e-9), (
                f'stub at sharp corner {int(near)} is {distance}, off the '
                'power-of-two ladder the termination argument needs'
            )
    assert checked >= 2, 'no sharp corner was split, so nothing was actually tested'


def test_the_queue_does_not_lean_on_the_rescan_that_backs_it():
    """That scan is a correctness net, not the mechanism: it should run once to
    seed the queue and once to confirm the queue is spent, and not again. If the
    incremental tracking misses work, this is where it shows up as cost."""
    algo = RuppertsAlgorithm(_thin_slab(), min_angle=25, max_area=REFINING_AREA)
    scans = []
    full_scan = algo.get_bad_triangles
    algo.get_bad_triangles = lambda: scans.append(None) or full_scan()

    algo.refine()

    assert len(scans) == 2, f'{len(scans)} full rescans, expected the seed and the confirm'


def _scanned_encroachment(algo):
    """Which segments are encroached, straight from the definition: some vertex
    strictly inside the diametral circle."""
    ends = algo.vertices[algo.segments]
    centers = ends.mean(axis=1)
    radii_sq = np.sum((ends[:, 1] - ends[:, 0])**2, axis=-1) / 4
    offsets = algo.vertices[None, :, :] - centers[:, None, :]
    inside = np.sum(offsets**2, axis=-1) < radii_sq[:, None] * (1 - ENCROACHMENT_TOLERANCE)
    return inside.any(axis=1)


def test_encroachment_tracking_does_not_drift_from_a_full_scan():
    """Encroachment is carried in a mask updated as vertices and segments are
    added, rather than rescanned every pass. That is state that can go wrong
    silently and still return a plausible mesh, so hold it against the
    definition through both of the mutations refinement makes."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=REFINING_AREA)
    assert np.array_equal(algo._encroached, _scanned_encroachment(algo))

    for step in range(12):
        encroached = algo.get_encroached_segments()
        if len(encroached):
            algo.split_segment(encroached[-1])
        else:
            # Somewhere inside the L, so the run stays representative.
            algo.add_vertex(np.array([0.35 + 0.04*step, 0.35]))
        assert np.array_equal(algo._encroached, _scanned_encroachment(algo)), (
            f'the mask disagrees with a full scan after step {step}'
        )


def test_refinement_preserves_the_input_geometry():
    """Refinement only ever appends vertices and splits segments in place, so
    input vertices keep their indices and the final segments still trace the
    input outline exactly -- no corner is cut or moved."""
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

    assert len(algo.get_encroached_segments()) == 0
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
    """A sub-60-degree corner is the one place the angle bound does not hold, so
    it should say so up front rather than leave the caller guessing."""
    spike = np.array([[0.0, 0.0], [10.0, 0.3], [10.0, -0.3]])
    with caplog.at_level(logging.WARNING, logger='fem.mesh.generation'):
        algo = RuppertsAlgorithm(PSLG(spike), min_angle=20)

    assert 'below the 60' in caplog.text
    assert algo.sharp_vertices == {0}, 'only the tip is too sharp to mesh'


def test_a_sharp_corner_terminates_and_costs_only_itself():
    """A 15 degree wedge cannot meet a 25 degree bound at its tip, and chasing
    it is what used to make refinement run away. The corner triangle is taken as
    it comes; everything else still meets the bound."""
    wedge = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10 * np.tan(np.radians(15))]])
    algo = RuppertsAlgorithm(PSLG(wedge), min_angle=25, max_area=1.0)
    mesh = algo.refine()

    angles = _min_angles(mesh)
    below = np.flatnonzero(angles < 25)
    assert len(below) == 1, f'{len(below)} elements below the bound, expected just the tip'
    assert angles[below[0]] == pytest.approx(15.0), 'the tip keeps the input angle'
    corner = np.flatnonzero((np.asarray(mesh.vertices) == [0.0, 0.0]).all(axis=1))
    assert corner[0] in mesh.elements[below[0]], 'the exempt element is the one at the tip'


def test_a_sharp_corner_is_still_bound_by_the_area_cap():
    """The exemption covers the angle bound only. A corner cannot be made less
    sharp, but it can be made smaller, and an element left oversized there would
    be the largest in the mesh."""
    wedge = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10 * np.tan(np.radians(15))]])
    max_area = 0.5
    mesh = RuppertsAlgorithm(PSLG(wedge), min_angle=25, max_area=max_area).refine()

    vertices = np.asarray(mesh.vertices)
    corners = vertices[np.asarray(mesh.elements)]
    edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = 0.5 * np.abs(edge_a[:, 0]*edge_b[:, 1] - edge_a[:, 1]*edge_b[:, 0])
    assert areas.max() <= max_area * (1 + 1e-9)


def test_segments_from_a_sharp_corner_split_onto_a_shared_shell():
    """Two segments meeting sharply have to be split at the same distance from
    the corner, or each new vertex lands inside the other segment's diametral
    circle and the splits walk into the corner without ever clearing it. Powers
    of two give them a ladder of radii to agree on however long each one is."""
    corner_angle = np.radians(20)
    arms = np.array([[0.0, 0.0], [10.0, 0.0], [7 * np.cos(corner_angle),
                                               7 * np.sin(corner_angle)]])
    algo = RuppertsAlgorithm(PSLG(arms), min_angle=25)

    long_arm = np.linalg.norm(algo._split_point([0, 1]))
    short_arm = np.linalg.norm(algo._split_point([0, 2]))
    assert long_arm == pytest.approx(4.0), 'a length-10 arm splits on the shell at 4'
    assert short_arm == pytest.approx(4.0), 'so does a length-7 one'


def test_a_segment_away_from_a_sharp_corner_splits_at_its_midpoint():
    """Shell splitting is the exception. Everywhere else the midpoint is what
    makes each half shorter than the parent, which is why refinement converges."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20)

    for segment in algo.segments:
        expected = 0.5 * (algo.vertices[segment[0]] + algo.vertices[segment[1]])
        np.testing.assert_allclose(algo._split_point(segment), expected)


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


def test_a_collinear_sliver_is_never_treated_as_a_bad_triangle():
    """Splitting a segment drops its midpoint exactly on the line through its
    endpoints, and qhull can fan that collinear triple into a zero-area sliver.
    A sliver has no circumcenter to refine towards -- it lands ~1e12 away -- so it
    is excluded from the bad set however far below the bound its zero angle sits,
    while a genuinely skinny triangle is still refined as before."""
    algo = RuppertsAlgorithm(PSLG(SQUARE_OUTLINE.copy()), min_angle=30)
    start, end = np.array([0.0, 0.0]), np.array([2.0, 1.0])
    extra = np.array([
        start, end, 0.5 * (start + end),          # a segment and its split midpoint
        [0.0, 0.0], [2.0, 0.0], [1.0, 0.05],      # a thin but non-degenerate triangle
    ])
    base = len(algo.vertices)
    algo.vertices = np.vstack([algo.vertices, extra])

    collinear = np.array([[base, base + 1, base + 2]])
    assert algo._is_degenerate(collinear)[0]
    assert not algo._fails_a_bound(collinear)[0], 'a zero-area sliver must not be refined'

    skinny = np.array([[base + 3, base + 4, base + 5]])
    assert not algo._is_degenerate(skinny)[0]
    assert algo._fails_a_bound(skinny)[0], 'a real skinny triangle still fails the bound'


def test_a_tight_outline_meshes_without_a_qhull_precision_error():
    """Regression for the backlog's qhull-precision bug: `CLOUD_OUTLINE` under an
    area cap used to raise QhullError partway through refinement. The mesh must come
    back, honour the angle bound (no collinear element dragging an angle to zero),
    and fill exactly what the outline encloses, so discarding the sliver drops no
    real region."""
    pslg = PSLG(CLOUD_OUTLINE.copy())
    algo = RuppertsAlgorithm(pslg, min_angle=30, max_area=0.005 * pslg.area())

    mesh = algo.refine()  # used to raise scipy.spatial.QhullError

    assert _min_angles(mesh).min() >= 30
    vertices = np.asarray(mesh.vertices)
    filled = sum(calculate_polygon_area(vertices[element]) for element in mesh.elements)
    assert filled == pytest.approx(pslg.area())


@pytest.mark.parametrize('max_area_fraction', [0.04, 0.06, 0.08, 0.10])
def test_a_reentrant_corner_meshes_through_an_incremental_precision_error(max_area_fraction):
    """Regression for a qhull precision error in *incremental* insertion: refining the
    sharp re-entrant `L_BRACKET_OUTLINE` under an area cap used to raise QhullError from
    `add_points` partway through, aborting the run. The wide merge is caught and the
    triangulation rebuilt in batch, so the mesh comes back honouring the angle bound and
    filling exactly what the outline encloses."""
    pslg = PSLG(L_BRACKET_OUTLINE.copy())
    algo = RuppertsAlgorithm(pslg, min_angle=25, max_area=max_area_fraction * pslg.area())

    mesh = algo.refine()  # used to raise scipy.spatial.QhullError from add_points

    assert _min_angles(mesh).min() >= 25
    vertices = np.asarray(mesh.vertices)
    filled = sum(calculate_polygon_area(vertices[element]) for element in mesh.elements)
    assert filled == pytest.approx(pslg.area())


def test_refinement_is_reproducible_despite_the_perturbation():
    """Each inserted circumcenter is nudged a hair off its exact position to dodge the
    cocircular precision failure, but from a fixed seed, so a run is deterministic:
    meshing the same outline twice must give the identical triangulation."""
    first = RuppertsAlgorithm(_l_shape(), min_angle=25, max_area=REFINING_AREA).refine()
    second = RuppertsAlgorithm(_l_shape(), min_angle=25, max_area=REFINING_AREA).refine()

    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.elements, second.elements)


def test_a_short_segment_far_from_the_origin_is_not_encroached_by_its_own_endpoints():
    """Regression for a floating-point non-termination. A segment short next to the
    coordinate magnitude (here length ~1e-3 at coordinates ~5) has endpoints that the
    old center-and-radius test computed as strictly inside its own diametral circle,
    from catastrophic cancellation. The segment then read as forever encroached and was
    split without end. The Thales dot-product form is exactly zero at an endpoint, so it
    cannot happen."""
    corner = np.array([4.72, 1.735])
    vertices = np.array([corner, corner + [1.1e-3, 5e-4], [0.0, 0.0], [7.0, 0.0]])
    algo = RuppertsAlgorithm(PSLG(vertices, segments=np.array([[0, 1]])), min_angle=20)

    # Only the segment's own endpoints sit on its circle; nothing is strictly inside.
    assert not algo._is_encroached(np.array([0, 1]))
    assert not algo._encroached.any()   # the KD-tree seed agrees


def test_a_finely_sampled_feature_far_from_the_origin_terminates():
    """End to end: a box with a tiny hole (edges ~1e-3) offset to coordinates ~5, the
    shape of a densely sampled airfoil trailing edge. Its short edges used to
    self-encroach and split forever; the run must now finish, honour the angle bound,
    and cut the hole out rather than fill it."""
    center, half = np.array([4.72, 1.735]), 1.5e-3
    hole = center + half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    box = np.array([[0.0, 0.0], [7.0, 0.0], [7.0, 4.0], [0.0, 4.0]])
    pslg = PSLG.from_loops([box, hole])

    mesh = RuppertsAlgorithm(pslg, min_angle=20, max_area=0.5).refine()

    assert _min_angles(mesh).min() >= 20
    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.elements)].mean(axis=1)
    assert not any(point_in_polygon(c, hole) for c in centroids)   # the hole is empty
