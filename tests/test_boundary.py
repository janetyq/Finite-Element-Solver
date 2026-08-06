"""Per-component Dirichlet conditions: pinning one component of a vector field
while leaving another free, rather than pinning all of them at once.

`None` in a Dirichlet value marks a free component (`[0, None]` pins x, leaves y
natural) -- the mechanism a roller/symmetry support needs and a full clamp does
not. These tests pin down the two things that make it work: two conditions on
different components of the same vertex merge instead of conflicting, and
`None` is rejected outright anywhere it would be meaningless (a load has no
"free" component).
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.equations import LinearElastic
from fem.regions import at_indices, intersect, on_plane
from fem.solver import Solver


def test_partial_pin_leaves_the_other_component_free(make_unit_square):
    """A roller: x pinned along the whole left edge, y pinned only at one corner
    to remove the last rigid-body mode. x must hold at 0 everywhere on that edge;
    y must vary elsewhere on it (a full clamp would hold it at 0 too)."""
    mesh = make_unit_square(10)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, None])
    bc.add(BCType.DIRICHLET, intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])
    solution = Solver(mesh, LinearElastic(E=200, nu=0.3), bc).solve()

    u = solution.u.reshape(-1, 2)
    left = np.flatnonzero(mesh.vertices[:, 0] == 0.0)
    assert np.allclose(u[left, 0], 0.0, atol=1e-12)
    assert not np.allclose(u[left, 1], 0.0, atol=1e-8), \
        "y should vary along a roller edge, not hold at 0 like a full clamp"


def test_two_conditions_merge_different_components_at_one_vertex(make_unit_square):
    """The corner where a roller edge meets its rigid-body-mode pin: two `add`
    calls, each naming one component, must both land in `fixed_idxs` rather than
    the second silently overwriting the first (or the two being flagged a
    conflict, since they never actually disagree)."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, None])
    bc.add(BCType.DIRICHLET, intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, -3.0])

    resolved = bc.resolve(mesh, n_components=2)
    origin = np.flatnonzero((mesh.vertices[:, 0] == 0.0) & (mesh.vertices[:, 1] == 0.0))[0]
    assert 2*origin in resolved.fixed_idxs
    assert 2*origin + 1 in resolved.fixed_idxs
    x_value = resolved.fixed_values[np.asarray(resolved.fixed_idxs) == 2*origin][0]
    y_value = resolved.fixed_values[np.asarray(resolved.fixed_idxs) == 2*origin + 1][0]
    assert x_value == 0.0
    assert y_value == -3.0


def test_conflicting_component_still_raises(make_unit_square):
    """Two conditions naming the *same* component of the same vertex with
    different values is a real conflict -- merging must not paper over it."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, None])
    bc.add(BCType.DIRICHLET, at_indices([0]), [1.0, None])  # vertex 0 is at (0, 0)

    with pytest.raises(ValueError, match='conflicting Dirichlet'):
        bc.resolve(mesh, n_components=2)


def test_neumann_rejects_a_free_component(make_unit_square):
    """None has no meaning for a load -- catch it at resolve() rather than let
    it become a silent NaN in the assembled traction."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, None])

    with pytest.raises(ValueError, match='None'):
        bc.resolve(mesh, n_components=2)


def test_robin_rejects_a_free_component(make_unit_square):
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add_robin(on_plane(0, 1.0), kappa=1.0, g=[1.0, None])

    with pytest.raises(ValueError, match='None'):
        bc.resolve(mesh, n_components=2)
