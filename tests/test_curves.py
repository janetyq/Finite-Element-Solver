"""Projection of points onto the analytic boundary curves."""
import numpy as np
import pytest

from fem.mesh.curves import Arc, Circle, CubicBezier, Line


def test_circle_projects_radially_onto_the_rim():
    circle = Circle([1.0, 2.0], 3.0)
    points = np.array([[1.0, 7.0], [11.0, 2.0], [1.0, 1.9]])
    projected = circle.project(points)

    # Every projection lands on the circle.
    radii = np.hypot(projected[:, 0] - 1.0, projected[:, 1] - 2.0)
    np.testing.assert_allclose(radii, 3.0)
    # Along the ray from the center, so a point due north/east maps to the rim there.
    np.testing.assert_allclose(projected[0], [1.0, 5.0])
    np.testing.assert_allclose(projected[1], [4.0, 2.0])


def test_circle_projection_fixes_points_already_on_the_rim():
    circle = Circle([0.0, 0.0], 2.0)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    on_rim = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    np.testing.assert_allclose(circle.project(on_rim), on_rim, atol=1e-12)


def test_circle_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        Circle([0.0, 0.0], 0.0)


def test_arc_projects_within_its_span_onto_the_circle():
    arc = Arc([0.0, 0.0], 1.0, 0.0, np.pi)   # upper half circle
    np.testing.assert_allclose(arc.project(np.array([0.0, 5.0])), [0.0, 1.0], atol=1e-12)


def test_arc_clamps_outside_its_span_to_the_nearer_endpoint():
    arc = Arc([0.0, 0.0], 1.0, 0.0, np.pi)
    # Just past the angle-0 endpoint snaps to it, not to the far side of the circle.
    np.testing.assert_allclose(arc.project(np.array([1.0, -0.2])), [1.0, 0.0], atol=1e-9)
    # Just past the angle-pi endpoint snaps there.
    np.testing.assert_allclose(arc.project(np.array([-1.0, -0.2])), [-1.0, 0.0], atol=1e-9)


def test_arc_rejects_empty_span():
    with pytest.raises(ValueError):
        Arc([0.0, 0.0], 1.0, np.pi, 0.0)


def _hump():
    """A cubic arching from (0,0) up to an apex and back down to (1,0)."""
    return CubicBezier([0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0])


def test_cubic_projection_fixes_points_already_on_the_curve():
    bezier = _hump()
    on_curve = bezier.sample(8)
    np.testing.assert_allclose(bezier.project(on_curve), on_curve, atol=1e-10)


def test_cubic_projection_is_the_nearest_point():
    """An off-curve point projects to the foot of the perpendicular: the residual is
    orthogonal to the tangent there, and no denser sample is closer."""
    bezier = _hump()
    query = np.array([0.3, 2.0])
    projected = bezier.project(query)

    dense = bezier.sample(4001)
    nearest = dense[np.argmin(np.linalg.norm(dense - query, axis=1))]
    assert np.linalg.norm(projected - query) <= np.linalg.norm(nearest - query) + 1e-9

    t = np.linspace(0, 1, 4001)[np.argmin(np.linalg.norm(dense - query, axis=1))]
    tangent = bezier._eval(t + 1e-5) - bezier._eval(t - 1e-5)
    assert abs(np.dot(projected - query, tangent)) < 1e-5


def test_cubic_projection_clamps_past_the_ends_to_an_endpoint():
    bezier = _hump()
    np.testing.assert_allclose(bezier.project(np.array([-2.0, -3.0])), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(bezier.project(np.array([3.0, -3.0])), [1.0, 0.0], atol=1e-9)


def test_cubic_projection_preserves_batch_shape():
    bezier = _hump()
    points = np.random.RandomState(0).rand(4, 3, 2)
    assert bezier.project(points).shape == (4, 3, 2)


def test_cubic_rejects_non_2d_control_points():
    with pytest.raises(ValueError):
        CubicBezier([0, 0, 0], [1, 1, 1], [2, 0, 0], [3, 3, 3])   # 3D points


# --- pieces ---

def test_circle_sample_omits_the_closing_repeat_and_arc_sample_includes_both_ends():
    ring = Circle([1.0, -1.0], 2.0).sample(8)
    assert ring.shape == (8, 2)
    assert np.allclose(np.hypot(ring[:, 0] - 1.0, ring[:, 1] + 1.0), 2.0)
    assert not np.allclose(ring[0], ring[-1])
    arc = Arc([0.0, 0.0], 1.0, 0.0, np.pi / 2)
    points = arc.sample(4)
    assert points.shape == (5, 2)
    assert np.allclose(points[0], arc.start) and np.allclose(points[-1], arc.end)
    assert np.allclose(arc.start, [1.0, 0.0]) and np.allclose(arc.end, [0.0, 1.0])


def test_a_reversed_arc_runs_the_other_way_and_projects_the_same():
    arc = Arc([0.0, 0.0], 1.0, 0.0, np.pi / 2)
    back = arc.reversed()
    assert np.allclose(back.start, arc.end) and np.allclose(back.end, arc.start)
    np.testing.assert_allclose(back.sample(4), arc.sample(4)[::-1])
    q = np.array([[3.0, 3.0], [-1.0, -1.0]])
    np.testing.assert_allclose(back.project(q), arc.project(q))
    assert back.reversed().start.tolist() == arc.start.tolist()


def test_line_projects_onto_the_segment_and_clamps_to_its_ends():
    line = Line([0.0, 0.0], [2.0, 0.0])
    np.testing.assert_allclose(line.project(np.array([[1.0, 3.0], [5.0, 1.0], [-2.0, -2.0]])),
                               [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    assert line.length() == 2.0
    np.testing.assert_allclose(line.sample(2), [[0, 0], [1, 0], [2, 0]])
    with pytest.raises(ValueError, match='distinct'):
        Line([1.0, 1.0], [1.0, 1.0])


def test_bezier_ends_and_length():
    bezier = CubicBezier([0, 0], [0, 0], [3, 0], [3, 0])   # a straight cubic
    assert np.allclose(bezier.start, [0, 0]) and np.allclose(bezier.end, [3, 0])
    assert bezier.length() == pytest.approx(3.0, rel=1e-6)
