"""`IncrementalDelaunay` grows a Delaunay triangulation one point at a time. Each insertion
must leave every circumcircle empty, the hull convex, and the arrays consistent with the
dicts underneath, whether the point lands inside a triangle, on an edge, on the hull, or
past it.
"""
import numpy as np
import pytest
from scipy.spatial import Delaunay

from fem.mesh.delaunay import GHOST, IncrementalDelaunay
from fem.mesh.ruppert import circumcenter


def _twice_areas(corners: np.ndarray) -> np.ndarray:
    u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def _assert_delaunay(tri: IncrementalDelaunay) -> None:
    points, simplices = tri.points, tri.simplices
    centres = circumcenter(points[simplices])
    radii = np.linalg.norm(points[simplices][:, 0] - centres, axis=-1)
    distances = np.linalg.norm(points[None, :, :] - centres[:, None, :], axis=-1)
    np.put_along_axis(distances, simplices, np.inf, axis=1)
    inside = distances < radii[:, None] * (1 - 1e-9)
    assert not inside.any(), f'{inside.any(axis=1).sum()} triangles have a vertex inside'


def _assert_consistent(tri: IncrementalDelaunay) -> None:
    """Every triangle is counter-clockwise, every directed edge has exactly one reverse,
    and the ghosts close the hull: one per hull edge, matching a batch triangulation's
    triangle count for the same points."""
    points, simplices, neighbors = tri.points, tri.simplices, tri.neighbors
    corners = points[simplices]
    assert (_twice_areas(corners) > 0).all()
    for edge, idx in tri._across.items():
        assert tri._across[edge[::-1]] != idx
        assert edge[::-1] in tri._across
    ghosts = [t for t in tri._triangles.values() if GHOST in t]
    assert all(t[2] == GHOST for t in ghosts)
    assert (neighbors == -1).sum() == len(ghosts)
    for row, (a, b, c) in enumerate(simplices):
        for j, (u, v) in enumerate(((b, c), (c, a), (a, b))):
            other = neighbors[row, j]
            if other >= 0:
                assert {u, v} <= set(simplices[other]) and row in neighbors[other]
    assert len(simplices) == len(Delaunay(points).simplices)


def test_random_points_inserted_one_at_a_time_stay_delaunay():
    rng = np.random.default_rng(0)
    points = rng.random((60, 2))
    tri = IncrementalDelaunay(points[:5])
    for p in points[5:]:
        tri.insert(p)
        _assert_delaunay(tri)
    _assert_consistent(tri)
    assert len(tri.points) == 60


def test_an_insertion_reports_the_triangles_it_made():
    tri = IncrementalDelaunay(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    idx, created = tri.insert(np.array([0.3, 0.4]))
    assert idx == 4
    assert all(idx in t for t in created)
    assert all(tri.contains(t) for t in created)
    # The point lay in one of the two initial triangles, and both had it in their
    # circumcircle, so the fan is over the whole square.
    assert len(created) == 4


@pytest.mark.parametrize('point, n_after', [
    ([0.5, 0.0], 3),    # on a hull edge: the edge splits, no sliver against it
    ([0.5, 0.5], 4),    # on the interior diagonal: both triangles re-fan
    ([2.0, 0.5], 3),    # past the hull: the hull grows to take it in
    ([0.5, -1e-13], 3), # a hair outside a hull edge: still a split, not a sliver
])
def test_a_point_on_or_past_the_hull_is_absorbed(point, n_after):
    tri = IncrementalDelaunay(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    tri.insert(np.array(point))
    _assert_consistent(tri)
    _assert_delaunay(tri)
    assert len(tri.simplices) == n_after
    assert np.abs(_twice_areas(tri.points[tri.simplices])).min() > 2e-3


def test_a_point_on_a_collinear_hull_splits_only_its_own_edge():
    """A hull sampled along a straight line has ghosts whose edges are collinear with a
    point on any one of them. Only the edge the point lies on may split; a round-off
    turn must not pull the neighbouring ghosts into the cavity and tear the hull."""
    slope = np.tan(np.radians(15))
    tri = IncrementalDelaunay(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10 * slope]]))
    for x in [4.0, 2.0, 6.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0]:
        tri.insert(np.array([x, x * slope]))
        _assert_consistent(tri)
        _assert_delaunay(tri)
    # Every point lies on the hull, so the hull edge count is the point count.
    assert (tri.neighbors == -1).sum() == len(tri.points)


def test_lookups_name_triangles_by_their_corners():
    tri = IncrementalDelaunay(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    (a, b, c), = tri.simplices[:1]
    assert tri.contains((int(c), int(a), int(b)))
    assert tri.find((int(c), int(a), int(b))) == tri.find((int(a), int(b), int(c)))
    assert not tri.contains((0, 1, 5))
    assert tri.triangle_on(0, 1) is not None
    assert tri.triangle_on(1, 0) == tri.triangle_on(0, 1)
    # The diagonal is one of (0, 2) or (1, 3); the other is not an edge.
    assert (tri.triangle_on(0, 2) is None) != (tri.triangle_on(1, 3) is None)


def test_the_walk_starts_near_and_the_buffer_grows():
    rng = np.random.default_rng(1)
    tri = IncrementalDelaunay(rng.random((4, 2)))
    near = None
    for p in rng.random((100, 2)):
        _, created = tri.insert(p, near=near)
        near = tri.find(created[0])
    _assert_consistent(tri)
    _assert_delaunay(tri)
    assert len(tri.points) == 104
