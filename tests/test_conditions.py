"""`Conditions`: one object for everything applied to a domain, and its resolution."""
import numpy as np
import pytest
from helpers import pinned

from fem.algebra.integrators import NewmarkMethod, ThetaMethod
from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial, ResolvedConditions
from fem.loads import BoundaryLoad, PointLoad, Source
from fem.physics.equations import Heat, LinearElastic, Poisson, Wave
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
    base = pinned()
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
    fixed = resolved.partition.fixed
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
    np.testing.assert_array_equal(resolved.fixed_values_at(2.0), 2.0)
    snapshot = resolved.at(2.0)
    assert not snapshot.is_time_dependent
    np.testing.assert_array_equal(snapshot.fixed_values, 2.0)
    np.testing.assert_allclose(snapshot.load_at(0.0), resolved.load_at(2.0))


def test_a_problem_is_its_conditions_resolved(make_unit_square):
    """The problem's partition, operator terms, and loads are exactly the resolution's,
    and a problem stated with the same conditions two ways solves the same."""
    mesh = _plate(make_unit_square)
    conditions = Conditions(Dirichlet(everywhere(), 0.0), Source(1.0))
    problem = Poisson().problem(mesh, conditions)
    assert problem.resolved.loads == problem.loads
    assert problem.source is problem.resolved.source
    np.testing.assert_array_equal(problem.partition.fixed, problem.resolved.fixed_idxs)
    split = Poisson().problem(mesh, pinned() + Source(1.0))
    np.testing.assert_allclose(split.solve().dofs, problem.solve().dofs)


def test_newmark_accepts_a_time_dependent_traction_but_not_a_moving_support(make_unit_square):
    mesh = _plate(make_unit_square)
    equation = LinearElastic(E=1.0, nu=0.3, density=1.0)
    ramp = TimeDependent(lambda p, t: [t, 0.0])
    loaded = equation.problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]),
                                                Neumann(on_plane(0, 1.0), ramp)))
    series = NewmarkMethod(dt=0.05, steps=4).solve(loaded)
    assert np.abs(series.dofs[-1]).max() > 0
    moving = equation.problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), ramp)))
    with pytest.raises(NotImplementedError, match='Dirichlet'):
        NewmarkMethod(dt=0.05, steps=4).solve(moving)


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
                         Neumann(on_plane(0, 0.0), lambda p: [0.0, p[:, 1]]))
    assert len(sheared.resolve(space).neumann) == 1
    # A TimeDependent value cannot be known to vanish, so on a roller's own nodes it
    # conflicts unless it leaves the pinned component None.
    rolling = Conditions(Dirichlet(on_plane(0, 0.0), [None, 0]), Neumann(on_plane(0, 0.0), ramp))
    with pytest.raises(ValueError, match='None'):
        rolling.resolve(space)
    free_y = TimeDependent(lambda p, t: [t, None])
    rolling = Conditions(Dirichlet(on_plane(0, 0.0), [None, 0]), Neumann(on_plane(0, 0.0), free_y))
    resolved = rolling.resolve(space)
    assert len(resolved.neumann) == 1
    np.testing.assert_allclose(resolved.at(1.0).neumann[0].nodal_values[:, 1], 0.0)


# -- the initial state -------------------------------------------------------------------


def test_initial_is_one_item_at_most_and_never_time_dependent():
    with pytest.raises(ValueError, match='one initial state'):
        Conditions(Initial(0.0), Initial(1.0))
    with pytest.raises(TypeError, match='TimeDependent'):
        Initial(TimeDependent(lambda p, t: t))
    with pytest.raises(TypeError, match='TimeDependent'):
        Initial(0.0, v0=TimeDependent(lambda p, t: t))
    conditions = Conditions(Dirichlet(everywhere(), 0.0), Initial(1.0))
    assert conditions.initial == Initial(1.0) and not conditions.is_time_dependent


def test_no_initial_resolves_to_the_dirichlet_lift_at_rest(make_unit_square):
    space = FunctionSpace(make_unit_square(4), n_components=1)
    resolved = Conditions(Dirichlet(on_plane(0, 0.0), 3.0)).resolve(space)
    lift = resolved.u0.dofs
    assert np.allclose(lift[resolved.fixed_idxs], 3.0) and np.allclose(lift[resolved.free_idxs], 0.0)
    assert np.allclose(resolved.v0.dofs, 0.0)


def test_initial_is_interpolated_at_the_nodes_and_checked_against_dirichlet(make_unit_square):
    space = FunctionSpace(make_unit_square(4), n_components=1)
    profile = Initial(lambda p: 1.0 + p[:, 0], v0=lambda p: p[:, 0])
    resolved = Conditions(Dirichlet(on_plane(0, 0.0), 1.0), profile).resolve(space)
    np.testing.assert_allclose(resolved.u0.dofs, 1.0 + space.node_coords[:, 0])
    np.testing.assert_allclose(resolved.v0.dofs, space.interpolate(profile.v0).dofs)
    with pytest.raises(ValueError, match='u0 disagrees with the Dirichlet'):
        Conditions(Dirichlet(on_plane(0, 0.0), 0.0), profile).resolve(space)
    with pytest.raises(ValueError, match='v0 must be zero'):
        Conditions(Dirichlet(on_plane(0, 1.0), 2.0), profile).resolve(space)


def test_a_field_is_taken_as_is_on_its_space_and_evaluated_on_another(make_unit_square):
    coarse = FunctionSpace(make_unit_square(3), n_components=1)
    fine = FunctionSpace(make_unit_square(6), n_components=1)
    ramp = coarse.interpolate(lambda p: p[:, 0] + 2 * p[:, 1])
    on_coarse, _ = Conditions().resolve(coarse).resolve_initial(Initial(ramp))
    assert on_coarse is ramp
    on_fine, _ = Conditions().resolve(fine).resolve_initial(Initial(ramp))
    np.testing.assert_allclose(on_fine.dofs, fine.interpolate(lambda p: p[:, 0] + 2 * p[:, 1]).dofs, atol=1e-12)


def test_integrators_step_from_the_initial_state_unless_overridden(make_unit_square):
    mesh = make_unit_square(5)
    hot = Conditions(Dirichlet(on_plane(0, 0.0), 1.0), Initial(lambda p: 1.0 - p[:, 0]))
    heat = Heat().problem(mesh, hot)
    stepped = ThetaMethod(dt=0.05, steps=4).solve(heat)
    by_hand = ThetaMethod(dt=0.05, steps=4).solve(heat, initial=Initial(lambda p: 1.0 - p[:, 0]))
    np.testing.assert_allclose(stepped.dofs, by_hand.dofs)
    continued = ThetaMethod(dt=0.05, steps=2).solve(heat, initial=Initial(stepped[2]))
    np.testing.assert_allclose(continued.dofs[-1], stepped.dofs[-1])
    with pytest.raises(ValueError, match='u0 disagrees with the Dirichlet'):
        ThetaMethod(dt=0.05, steps=1).solve(heat, initial=Initial(0.0))

    def bump(p):
        return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    wave = Wave().problem(mesh, Conditions(Dirichlet(everywhere(), 0.0), Initial(0.0, v0=bump)))
    rest = Wave().problem(mesh, pinned())
    rung = NewmarkMethod(dt=0.02, steps=5).solve(wave)
    np.testing.assert_allclose(rung.dofs[0], 0.0)
    assert np.abs(rung.dofs[-1]).max() > 1e-3
    np.testing.assert_allclose(NewmarkMethod(dt=0.02, steps=5).solve(rest).dofs, 0.0)


def test_newton_iterates_from_the_initial_state(make_unit_square):
    """A linear problem is solved in one Newton step from any seed, so the answer is the
    same from the conditions' state, from an `initial=`, and from the default lift; a
    seed is a guess, so unlike an integrator's start it is not held to the Dirichlet data."""
    from fem.algebra.solve import NewtonSolve
    mesh = make_unit_square(4)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), 0.0), Source(1.0))
    cold = Poisson().problem(mesh, bc)
    warm = Poisson().problem(mesh, bc + Initial(lambda p: p[:, 0]))
    reference = cold.solve().dofs
    np.testing.assert_allclose(NewtonSolve().solve(warm), reference, atol=1e-10)
    np.testing.assert_allclose(NewtonSolve().solve(cold, initial=Initial(warm.u0)), reference, atol=1e-10)
    np.testing.assert_allclose(NewtonSolve().solve(cold, initial=Initial(5.0)), reference, atol=1e-10)


def test_a_time_dependent_support_re_evaluates_values_without_re_partitioning(make_unit_square, monkeypatch):
    """The DOFs a condition fixes are set by its region, resolved once; only the values
    move in time. `fixed_values_at(t)` therefore reads the partition it already holds and
    never partitions again, and agrees with a fresh resolution of the same value fixed
    at `t`."""
    import fem.conditions as conditions_module
    mesh = make_unit_square(6)
    space = FunctionSpace(mesh, n_components=2)
    ramp = TimeDependent(lambda p, t: [t * p[:, 1], None])
    moving = Conditions(Dirichlet(on_plane(0, 0.0), ramp), Dirichlet(on_plane(1, 0.0), [0.0, 0.0]))
    resolved = moving.resolve(space)

    def refuse(*args, **kwargs):
        raise AssertionError('fixed_values_at must not partition the DOFs again')

    monkeypatch.setattr(conditions_module, '_partition', refuse)
    values = resolved.fixed_values_at(0.7)
    fixed = resolved.partition.fixed
    assert fixed is resolved.fixed_idxs
    monkeypatch.undo()

    frozen = Conditions(Dirichlet(on_plane(0, 0.0), lambda p: [0.7 * p[:, 1], None]),
                        Dirichlet(on_plane(1, 0.0), [0.0, 0.0])).resolve(space)
    np.testing.assert_array_equal(fixed, frozen.fixed_idxs)
    np.testing.assert_allclose(values, frozen.fixed_values)
    # The values at t = 0 are the resolution's own.
    np.testing.assert_allclose(resolved.fixed_values_at(0.0), resolved.fixed_values)

    # A value that fixes different components at different times is refused.
    flicker = Conditions(Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: [0.0, None if t > 0 else 0.0])))
    with pytest.raises(ValueError, match='different set of components'):
        flicker.resolve(space).fixed_values_at(1.0)


def test_the_partition_merges_overlapping_conditions_and_free_components_by_hand():
    """On a 2x2 vertex square with two components: the left edge pins x and leaves y free,
    the bottom edge pins both; the corner they share takes x from either (they agree) and
    y from the bottom edge. Fixed DOFs come out node-major, values in the same order, and
    the free set is the complement."""
    from fem.mesh.mesh import Mesh
    mesh = Mesh([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 3], [0, 3, 2]])
    space = FunctionSpace(mesh, n_components=2)
    resolved = Conditions(Dirichlet(on_plane(0, 0.0), [1.0, None]),
                          Dirichlet(on_plane(1, 0.0), [1.0, 2.0])).resolve(space)
    # vertex 0 (corner): x=1 from both, y=2 from the bottom; vertex 1 (bottom): x=1, y=2;
    # vertex 2 (left): x=1 only; vertex 3: free.
    np.testing.assert_array_equal(resolved.fixed_idxs, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(resolved.fixed_values, [1.0, 2.0, 1.0, 2.0, 1.0])
    np.testing.assert_array_equal(resolved.free_idxs, [5, 6, 7])
    with pytest.raises(ValueError, match='conflicting Dirichlet values at vertex 0'):
        Conditions(Dirichlet(on_plane(0, 0.0), [1.0, None]),
                   Dirichlet(on_plane(1, 0.0), [3.0, 2.0])).resolve(space)
