"""Per-component Dirichlet conditions: `None` in a value marks a free component
(`[0, None]` pins x, leaves y natural), the mechanism a roller needs. Two conditions on
different components of one vertex merge. A Neumann value may leave a component
`None` too, one the traction does not drive; a Robin `g` may not.
"""
import numpy as np
import pytest

from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions
from fem.physics.equations import LinearElastic
from fem.regions import at_indices, intersect, on_plane
from fem.space import FunctionSpace


def test_partial_pin_leaves_the_other_component_free(make_unit_square):
    """A roller: x pinned along the whole left edge, y pinned only at one corner
    to remove the last rigid-body mode. x must hold at 0 everywhere on that edge;
    y must vary elsewhere on it (a full clamp would hold it at 0 too)."""
    mesh = make_unit_square(10)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, None]),
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0]),
        Neumann(on_plane(0, 1.0), [1.0, 0]),
    )
    solution = LinearElastic(E=200, nu=0.3).problem(mesh, bc).solve()

    u = solution.nodal_values
    left = np.flatnonzero(mesh.vertices[:, 0] == 0.0)
    assert np.allclose(u[left, 0], 0.0, atol=1e-12)
    assert not np.allclose(u[left, 1], 0.0, atol=1e-8), \
        "y should vary along a roller edge, not hold at 0 like a full clamp"


def test_two_conditions_merge_different_components_at_one_vertex(make_unit_square):
    """The corner where a roller edge meets its rigid-body-mode pin: two `add`
    calls, each naming one component, both land in `fixed_idxs`; they do not
    conflict, since they never disagree."""
    mesh = make_unit_square(6)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0.0, None]),
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, -3.0]),
    )

    resolved = bc.resolve(FunctionSpace(mesh, n_components=2))
    origin = np.flatnonzero((mesh.vertices[:, 0] == 0.0) & (mesh.vertices[:, 1] == 0.0))[0]
    assert 2*origin in resolved.fixed_idxs
    assert 2*origin + 1 in resolved.fixed_idxs
    x_value = resolved.fixed_values[np.asarray(resolved.fixed_idxs) == 2*origin][0]
    y_value = resolved.fixed_values[np.asarray(resolved.fixed_idxs) == 2*origin + 1][0]
    assert x_value == 0.0
    assert y_value == -3.0


def test_conflicting_component_still_raises(make_unit_square):
    """Two conditions naming the same component of the same vertex with different values
    is a real conflict."""
    mesh = make_unit_square(6)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0.0, None]),
        Dirichlet(at_indices([0]), [1.0, None]),
    )

    with pytest.raises(ValueError, match='conflicting Dirichlet'):
        bc.resolve(FunctionSpace(mesh, n_components=2))


def test_a_robin_value_rejects_a_free_component():
    """None has no meaning for a Robin g; it is caught when the condition is built."""
    with pytest.raises(ValueError, match='None'):
        Robin(on_plane(0, 1.0), kappa=1.0, g=[1.0, None])


@pytest.mark.parametrize('value', [[1.0, None], lambda p: [1.0, None]], ids=['constant', 'callable'])
def test_a_neumann_free_component_integrates_as_zero(make_unit_square, value):
    """A None component of a traction drives nothing: the load it assembles is the
    load of the same traction with a zero there."""
    mesh = make_unit_square(6)
    space = FunctionSpace(mesh, n_components=2)
    with_none = Conditions(Neumann(on_plane(0, 1.0), value)).resolve(space)
    with_zero = Conditions(Neumann(on_plane(0, 1.0), [1.0, 0.0])).resolve(space)
    np.testing.assert_allclose(with_none.load_at(0.0), with_zero.load_at(0.0))
    assert not with_none.neumann[0].loaded[:, 1].any()
