"""Per-component Dirichlet conditions: `None` in a value marks a free component
(`[0, None]` pins x, leaves y natural), the mechanism a roller needs. Two conditions on
different components of one vertex merge, and `None` is rejected where it is
meaningless (a load has no free component).
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet, Neumann, Robin
from fem.equations import LinearElastic
from fem.regions import at_indices, intersect, on_plane
from fem.solver import Solver


def test_partial_pin_leaves_the_other_component_free(make_unit_square):
    """A roller: x pinned along the whole left edge, y pinned only at one corner
    to remove the last rigid-body mode. x must hold at 0 everywhere on that edge;
    y must vary elsewhere on it (a full clamp would hold it at 0 too)."""
    mesh = make_unit_square(10)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, None]),
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0]),
        Neumann(on_plane(0, 1.0), [1.0, 0]),
    )
    solution = Solver(mesh, LinearElastic(E=200, nu=0.3), bc).solve()

    u = solution.u.reshape(-1, 2)
    left = np.flatnonzero(mesh.vertices[:, 0] == 0.0)
    assert np.allclose(u[left, 0], 0.0, atol=1e-12)
    assert not np.allclose(u[left, 1], 0.0, atol=1e-8), \
        "y should vary along a roller edge, not hold at 0 like a full clamp"


def test_two_conditions_merge_different_components_at_one_vertex(make_unit_square):
    """The corner where a roller edge meets its rigid-body-mode pin: two `add`
    calls, each naming one component, both land in `fixed_idxs`; they do not
    conflict, since they never disagree."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0.0, None]),
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, -3.0]),
    )

    resolved = bc.resolve(mesh, n_components=2)
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
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0.0, None]),
        Dirichlet(at_indices([0]), [1.0, None]),
    )

    with pytest.raises(ValueError, match='conflicting Dirichlet'):
        bc.resolve(mesh, n_components=2)


@pytest.mark.parametrize('condition', [
    lambda region: Neumann(region, [1.0, None]),
    lambda region: Robin(region, kappa=1.0, g=[1.0, None]),
], ids=['neumann', 'robin'])
def test_a_load_rejects_a_free_component(condition):
    """None has no meaning for a load; it is caught when the condition is built."""
    with pytest.raises(ValueError, match='None'):
        condition(on_plane(0, 1.0))


def test_a_callable_load_with_a_free_component_is_caught_at_resolve(make_unit_square):
    """A callable's components are only seen at the nodes, so a None among them is
    caught at resolve()."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions(Neumann(on_plane(0, 1.0), lambda p: [1.0, None]))
    with pytest.raises(ValueError, match='None'):
        bc.resolve(mesh, n_components=2)
