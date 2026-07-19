"""Tests for position-based regions and field evaluation."""
import numpy as np
import pytest

from fem.regions import (
    everywhere,
    on_plane,
    in_box,
    intersect,
    union,
    at_indices,
    is_mesh_bound,
    evaluate_field,
)


POINTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [1.0, 0.0],
    [1.0, 1.0],
])


def test_everywhere_selects_all():
    assert everywhere()(POINTS).all()


def test_on_plane_selects_a_face():
    assert list(np.flatnonzero(on_plane(0, 0.0)(POINTS))) == [0, 1]
    assert list(np.flatnonzero(on_plane(1, 1.0)(POINTS))) == [1, 4]


def test_on_plane_tolerates_round_off():
    points = np.array([[1e-12, 0.0]])
    assert on_plane(0, 0.0)(points).all()
    assert not on_plane(0, 0.0, atol=1e-15)(points).all()


def test_in_box_bounds_are_inclusive_and_optional():
    band = in_box([None, 0.4], [None, 0.6])  # unbounded in x
    assert list(np.flatnonzero(band(POINTS))) == [2]

    corner = in_box([0.9, 0.9], [1.1, 1.1])
    assert list(np.flatnonzero(corner(POINTS))) == [4]


def test_intersect_and_union():
    right = on_plane(0, 1.0)
    top = on_plane(1, 1.0)
    assert list(np.flatnonzero(intersect(right, top)(POINTS))) == [4]
    assert list(np.flatnonzero(union(right, top)(POINTS))) == [1, 3, 4]


def test_at_indices_is_mesh_bound_and_plain_regions_are_not():
    assert is_mesh_bound(at_indices([0, 2]))
    assert not is_mesh_bound(everywhere())
    assert list(np.flatnonzero(at_indices([0, 2])(POINTS))) == [0, 2]


def test_mesh_boundness_propagates_through_composition():
    """A composite is only as remeshable as its least remeshable part."""
    assert is_mesh_bound(intersect(everywhere(), at_indices([0])))
    assert is_mesh_bound(union(everywhere(), at_indices([0])))
    assert not is_mesh_bound(intersect(everywhere(), on_plane(0, 0.0)))


# --- fields ---

def test_constant_field_is_broadcast_to_every_point():
    values = evaluate_field([2.0, 3.0], POINTS, dim=2)
    assert values.shape == (5, 2)
    assert np.allclose(values, [2.0, 3.0])


def test_scalar_constant_works_for_dim_one():
    assert np.allclose(evaluate_field(1.5, POINTS, dim=1), 1.5)


def test_callable_field_is_evaluated_per_point():
    values = evaluate_field(lambda p: [p[0] + p[1]], POINTS, dim=1)
    assert np.allclose(values.ravel(), POINTS.sum(axis=1))


def test_none_is_zero():
    assert np.allclose(evaluate_field(None, POINTS, dim=2), 0.0)


def test_wrong_width_raises_rather_than_being_guessed():
    """The old API inferred meaning from a length coincidence; a mismatch must
    simply be an error."""
    with pytest.raises(ValueError):
        evaluate_field([1.0, 2.0, 3.0], POINTS, dim=2)


def test_field_width_is_independent_of_point_count():
    """Regression for the length-coincidence bug: a 2-component value on exactly
    2 points must still mean 'both components at both points'."""
    two_points = POINTS[:2]
    values = evaluate_field([7.0, 9.0], two_points, dim=2)
    assert values.shape == (2, 2)
    assert np.allclose(values, [7.0, 9.0])
