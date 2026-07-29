"""Ruppert's algorithm: the guarantees it makes about the mesh it returns.

The algorithm terminates only when no triangle is skinny and no segment is
encroached, and then keeps only what the PSLG encloses. Those are what it
promises a caller. The tests below assert them directly rather than pinning a
particular triangulation -- any correct refinement satisfies them, which keeps
the suite useful across changes to insertion order or the underlying Delaunay.
"""
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


def test_a_bounding_box_keeps_the_ring_around_the_outline():
    """Documents the enclosure rule at its least obvious: a box around an outline
    leaves the ring between them genuinely enclosed by segments, so it is kept.
    Treating the inner loop as a hole instead needs a fill rule PSLG cannot yet
    express."""
    pslg = _l_shape()
    pslg.add_bounding_box(buffer=0.2)
    algo = RuppertsAlgorithm(pslg, min_angle=20)
    mesh = algo.refine()

    assert not algo.get_exterior_triangles().any()
    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.elements)].mean(axis=1)
    assert not all(point_in_polygon(c, L_SHAPE_OUTLINE) for c in centroids)


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
