"""`Conditions`: one object for everything applied to a domain, and its resolution."""
import numpy as np
import pytest

from fem.algebra.integrators import NewmarkMethod
from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, ResolvedConditions
from fem.loads import BoundaryLoad, PointLoad, Source
from fem.physics.equations import LinearElastic, Poisson
from fem.physics.forms import ScaledForm
from fem.regions import TimeDependent, at_indices, everywhere, on_plane
from fem.space import FunctionSpace


def _plate(make_unit_square):
    return make_unit_square(5)


def test_items_are_kept_in_order_and_viewed_by_kind():
    pinned = Dirichlet(on_plane(0, 0.0), [0, 0])
    pulled = Neumann(on_plane(0, 1.0), [1, 0])
    spring = Robin(on_plane(1, 1.0), 2.0, [0, 0])
    gravity = Source([0, -1])
    tip = PointLoad(at_indices([3]), [0, -1])
    conditions = Conditions(pinned, pulled, spring, gravity, tip)
    assert conditions.items == (pinned, pulled, spring, gravity, tip)
    assert conditions.boundary == (pinned, pulled, spring)
    assert conditions.dirichlet == (pinned,) and conditions.neumann == (pulled,)
    assert conditions.robin == (spring,) and conditions.source is gravity
    assert conditions.point_loads == (tip,)
    assert len(conditions) == 5 and list(conditions) == list(conditions.items)


def test_adding_appends_and_leaves_the_original():
    base = Conditions(Dirichlet(everywhere(), 0.0))
    more = base + Source(1.0)
    assert len(base) == 1 and len(more) == 2 and more.source is not None
    merged = more + Conditions(PointLoad(at_indices([0]), 1.0))
    assert [type(i).__name__ for i in merged] == ['Dirichlet', 'Source', 'PointLoad']


def test_one_volume_source_at_most_and_only_known_items():
    with pytest.raises(ValueError, match='one volume source'):
        Conditions(Source(1.0), Source(2.0))
    with pytest.raises(ValueError, match='one volume source'):
        Conditions(Source(1.0)) + Source(2.0)
    with pytest.raises(TypeError, match='Dirichlet, Neumann, or Robin'):
        Conditions(('dirichlet', everywhere(), 0))  # type: ignore[arg-type]


def test_time_dependence_tells_a_moving_support_from_a_moving_load():
    moving_load = Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]),
                             Neumann(on_plane(0, 1.0), TimeDependent(lambda p, t: [t, 0])))
    moving_support = Conditions(Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: [t, 0])))
    assert moving_load.is_time_dependent and not moving_load.has_time_dependent_dirichlet
    assert moving_support.has_time_dependent_dirichlet
    assert not Conditions(Source(1.0)).is_time_dependent


def test_resolution_carries_constraints_operator_terms_and_loads(make_unit_square):
    mesh = _plate(make_unit_square)
    space = FunctionSpace(mesh, n_components=2)
    conditions = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]), Neumann(on_plane(0, 1.0), [1, 0]),
        Robin(on_plane(1, 1.0), 2.0, [0, 0]), Source([0, -1]), PointLoad(at_indices([0]), [0, -1]))
    resolved = conditions.resolve(space)
    assert isinstance(resolved, ResolvedConditions)
    free, fixed, values = resolved.constraints
    assert len(fixed) == 2 * len(Dirichlet(on_plane(0, 0.0), [0, 0]).select(mesh))
    assert len(resolved.operator_terms) == 1 and isinstance(resolved.operator_terms[0], ScaledForm)
    assert [type(t).__name__ for t in resolved.loads] == ['Source', 'BoundaryLoad', 'BoundaryLoad', 'PointLoad']
    assert resolved.source is conditions.source and isinstance(resolved.loads[1], BoundaryLoad)
    assert resolved.load_at(0.0).shape == (space.n_dofs,)


def test_a_snapshot_at_a_time_is_no_longer_time_dependent(make_unit_square):
    space = FunctionSpace(_plate(make_unit_square), n_components=1)
    ramp = TimeDependent(lambda p, t: t)
    resolved = Conditions(Dirichlet(on_plane(0, 0.0), ramp), Source(ramp)).resolve(space)
    assert resolved.is_time_dependent and resolved.has_time_dependent_dirichlet
    np.testing.assert_array_equal(resolved.fixed_values, 0.0)
    np.testing.assert_array_equal(resolved.constraints_at(2.0)[2], 2.0)
    snapshot = resolved.at(2.0)
    assert not snapshot.is_time_dependent
    np.testing.assert_array_equal(snapshot.fixed_values, 2.0)
    np.testing.assert_allclose(snapshot.load_at(0.0), resolved.load_at(2.0))


def test_a_problem_is_its_conditions_resolved(make_unit_square):
    """The problem's constraints, operator terms, and loads are exactly the resolution's,
    and a problem stated with the same conditions two ways solves the same."""
    mesh = _plate(make_unit_square)
    conditions = Conditions(Dirichlet(everywhere(), 0.0), Source(1.0))
    problem = Poisson().problem(mesh, conditions)
    assert problem.resolved.loads == problem.loads
    assert problem.source is problem.resolved.source
    np.testing.assert_array_equal(problem.constraints[1], problem.resolved.fixed_idxs)
    split = Poisson().problem(mesh, Conditions(Dirichlet(everywhere(), 0.0)) + Source(1.0))
    np.testing.assert_allclose(split.solve().dofs, problem.solve().dofs)


def test_newmark_accepts_a_time_dependent_traction_but_not_a_moving_support(make_unit_square):
    mesh = _plate(make_unit_square)
    equation = LinearElastic(E=1.0, nu=0.3, density=1.0)
    ramp = TimeDependent(lambda p, t: [t, 0.0])
    loaded = equation.problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]),
                                                Neumann(on_plane(0, 1.0), ramp)))
    rest = np.zeros(loaded.space.n_dofs)
    series = NewmarkMethod(dt=0.05, steps=4).solve(loaded, rest, rest)
    assert np.abs(series.dofs[-1]).max() > 0
    moving = equation.problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), ramp)))
    with pytest.raises(NotImplementedError, match='Dirichlet'):
        NewmarkMethod(dt=0.05, steps=4).solve(moving, rest, rest)


def test_a_traction_on_a_pinned_component_conflicts_whatever_its_value(make_unit_square):
    """The conflict is read off the specification: a TimeDependent traction that is
    zero at t = 0 on a pinned component is still a traction on that component."""
    space = FunctionSpace(_plate(make_unit_square), n_components=2)
    ramp = TimeDependent(lambda p, t: [t, 0.0])
    clashing = Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]), Neumann(on_plane(0, 0.0), ramp))
    with pytest.raises(ValueError, match='Dirichlet and a Neumann'):
        clashing.resolve(space)
    # A constant that writes zero on the pinned component loads only the other.
    roller = Conditions(Dirichlet(on_plane(0, 0.0), [0, None]), Neumann(on_plane(0, 0.0), [0, -1.0]))
    assert len(roller.resolve(space).neumann) == 1
    # A callable of position is read at the nodes, where its zeros are exact.
    sheared = Conditions(Dirichlet(on_plane(0, 0.0), [0, None]),
                         Neumann(on_plane(0, 0.0), lambda p: [0.0, p[1]]))
    assert len(sheared.resolve(space).neumann) == 1
    # A TimeDependent value is sampled over time, so a ramp that only ever drives the
    # free x component of a y-pinned roller resolves, on the roller's own nodes.
    rolling = Conditions(Dirichlet(on_plane(0, 0.0), [None, 0]), Neumann(on_plane(0, 0.0), ramp))
    assert len(rolling.resolve(space).neumann) == 1
