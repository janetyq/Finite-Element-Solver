"""Ruppert's algorithm: the guarantees it makes about the mesh it returns.

The algorithm terminates only when no triangle is skinny and no segment is
encroached, so those two properties are what it promises a caller. The tests
below assert them directly rather than pinning a particular triangulation --
any correct refinement satisfies them, which keeps the suite useful across
changes to insertion order or the underlying Delaunay.
"""
import numpy as np
import pytest

from fem.geometry import calculate_polygon_area, calculate_triangle_min_angle
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import PSLG


def _l_shape() -> PSLG:
    """A non-convex outline inside a bounding box.

    The reflex corner is the point: it forces segment splitting, which a convex
    outline of the same size would not exercise.
    """
    outline = np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
    ])
    pslg = PSLG(outline)
    pslg.add_bounding_box(buffer=0.2)
    return pslg


def _min_angles(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    return np.array([calculate_triangle_min_angle(vertices[element]) for element in mesh.elements])


@pytest.mark.parametrize('min_angle', [15, 20, 25])
def test_every_triangle_meets_the_angle_bound(min_angle):
    """The headline guarantee: no element is skinnier than what was asked for."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=min_angle)
    mesh = algo.run_algo()

    angles = _min_angles(mesh)
    assert angles.min() >= min_angle, (
        f'{(angles < min_angle).sum()} of {len(angles)} elements are below the '
        f'{min_angle} degree bound, worst {angles.min():.2f}'
    )


def test_no_segment_is_encroached_on_return():
    """The other half of termination, and what makes the mesh conform to the
    outline: every segment's diametral circle is empty, so each survives as an
    edge of the Delaunay triangulation."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20)
    algo.run_algo()

    assert algo.get_encroached_segments() == []


def test_refinement_preserves_the_input_geometry():
    """Refinement only ever appends vertices and splits segments at their
    midpoint, so input vertices keep their indices and the final segments still
    trace the input outline exactly -- no corner is cut or moved."""
    pslg = _l_shape()
    original_vertices = np.array(pslg.vertices)
    original_segments = np.array(pslg.segments)
    algo = RuppertsAlgorithm(pslg, min_angle=20)
    mesh = algo.run_algo()

    vertices = np.asarray(mesh.vertices)
    np.testing.assert_allclose(vertices[:len(original_vertices)], original_vertices)

    for start, end in algo.segments:
        # The segment must be a sub-interval of one of the segments it descends
        # from: collinear with it, and inside it.
        for original in original_segments:
            edge = original_vertices[original[1]] - original_vertices[original[0]]
            offsets = vertices[[start, end]] - original_vertices[original[0]]
            fractions = offsets @ edge / (edge @ edge)
            collinear = np.allclose(edge[0]*offsets[:, 1] - edge[1]*offsets[:, 0], 0, atol=1e-9)
            if collinear and np.all((fractions > -1e-9) & (fractions < 1 + 1e-9)):
                break
        else:
            pytest.fail(f'segment {start}-{end} does not lie on any input segment')


def test_segments_survive_as_edges_of_the_triangulation():
    """Conformity: the mesh resolves the outline instead of cutting across it."""
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20)
    mesh = algo.run_algo()

    edges = set()
    for element in mesh.elements:
        for i in range(3):
            edges.add(tuple(sorted((int(element[i]), int(element[(i + 1) % 3])))))

    missing = [tuple(s) for s in algo.segments if tuple(sorted(s)) not in edges]
    assert not missing, f'{len(missing)} segments are not edges of the mesh: {missing[:5]}'


def test_max_area_bounds_every_element():
    """The optional area cap refines past the angle bound until no element is
    larger than `max_area`."""
    max_area = 0.2
    algo = RuppertsAlgorithm(_l_shape(), min_angle=20, max_area=max_area)
    mesh = algo.run_algo()

    vertices = np.asarray(mesh.vertices)
    areas = np.array([calculate_polygon_area(vertices[element]) for element in mesh.elements])
    assert areas.max() <= max_area


def test_a_finer_angle_bound_costs_more_triangles():
    """Sanity check that `min_angle` is actually driving the refinement."""
    coarse = RuppertsAlgorithm(_l_shape(), min_angle=15).run_algo()
    fine = RuppertsAlgorithm(_l_shape(), min_angle=25).run_algo()

    assert len(fine.elements) > len(coarse.elements)


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
