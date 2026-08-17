"""Projection of points onto the analytic boundary curves."""
import numpy as np
import pytest

from fem.mesh.curves import Arc, Circle


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
