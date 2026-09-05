"""Tests for position-based regions and field evaluation."""
import numpy as np
import pytest

from fem.boundary import _evaluate_dirichlet_value
from fem.regions import (
    at_indices,
    evaluate_field,
    everywhere,
    in_box,
    intersect,
    is_mesh_bound,
    on_plane,
    union,
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
    values = evaluate_field([2.0, 3.0], POINTS, n_components=2)
    assert values.shape == (5, 2)
    assert np.allclose(values, [2.0, 3.0])


def test_scalar_constant_works_for_dim_one():
    assert np.allclose(evaluate_field(1.5, POINTS, n_components=1), 1.5)


def test_callable_field_is_given_every_point_at_once():
    calls = []

    def value(p):
        calls.append(p.shape)
        return [p[:, 0] + p[:, 1]]

    values = evaluate_field(value, POINTS, n_components=1)
    assert np.allclose(values.ravel(), POINTS.sum(axis=1))
    assert calls == [POINTS.shape]


MANY = np.random.default_rng(0).random((40, 2))


@pytest.mark.parametrize('value, n_components, expected', [
    (lambda p: [p[:, 0] + p[:, 1]], 1, lambda q: q.sum(axis=1, keepdims=True)),
    (lambda p: p[:, 0] * p[:, 1], 1, lambda q: (q[:, 0] * q[:, 1])[:, None]),
    (lambda p: [p[:, 1], 0.0], 2, lambda q: np.stack([q[:, 1], np.zeros(len(q))], axis=1)),
    (lambda p: p, 2, lambda q: q),
    (lambda p: np.stack([p[:, 0], -p[:, 0]], axis=-1), 2, lambda q: np.stack([q[:, 0], -q[:, 0]], axis=1)),
    (lambda p: [1.0, 2.0], 2, lambda q: np.tile([1.0, 2.0], (len(q), 1))),
    (lambda p: 3.0, 1, lambda q: np.full((len(q), 1), 3.0)),
    (lambda p: np.where(p[:, 0] > 0.5, 1.0, 0.0), 1, lambda q: (q[:, :1] > 0.5).astype(float)),
], ids=['list', 'bare-array', 'mixed', 'identity', 'stacked', 'constants', 'scalar', 'where'])
def test_the_result_layouts_a_callable_may_use(value, n_components, expected):
    """A sequence of per-component entries (scalar or array), an (N, k) array, a bare
    (N,) array or scalar for one component: all read as the same (N, k) values."""
    assert np.allclose(evaluate_field(value, MANY, n_components), expected(MANY))
    assert np.allclose(evaluate_field(value, MANY[:2], n_components), expected(MANY[:2]))


def test_a_callable_may_leave_a_component_none():
    values = _evaluate_dirichlet_value(lambda p: [p[:, 0], None], MANY, 2)
    assert np.allclose(values[:, 0], MANY[:, 0]) and np.isnan(values[:, 1]).all()


@pytest.mark.parametrize('value, n_components', [
    (lambda p: [p[0], p[1]], 2),       # the first two points, not x and y
    (lambda p: p[0], 1),
    (lambda p: [p[:, 0], p[:, 1], 0.0], 2),
    (lambda p: np.array([p[:, 0], p[:, 1]]), 2),   # (k, N): stack along the last axis
], ids=['first-points', 'first-point', 'too-wide', 'transposed'])
def test_a_result_of_the_wrong_shape_is_an_error_that_names_the_contract(value, n_components):
    with pytest.raises(ValueError, match=r'p\[:, 0\], not p\[0\]'):
        evaluate_field(value, MANY, n_components)


def test_a_body_that_branches_on_a_point_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        evaluate_field(lambda p: [1.0 if p[0] > 0.5 else 0.0], MANY, n_components=1)


def test_none_is_zero():
    assert np.allclose(evaluate_field(None, POINTS, n_components=2), 0.0)


def test_wrong_width_raises_rather_than_being_guessed():
    """A value of the wrong width is an error, not reinterpreted."""
    with pytest.raises(ValueError):
        evaluate_field([1.0, 2.0, 3.0], POINTS, n_components=2)


def test_field_width_is_independent_of_point_count():
    """A 2-component value on exactly 2 points still means 'both components at both points'."""
    two_points = POINTS[:2]
    values = evaluate_field([7.0, 9.0], two_points, n_components=2)
    assert values.shape == (2, 2)
    assert np.allclose(values, [7.0, 9.0])
