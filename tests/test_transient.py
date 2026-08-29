"""Time-dependent loads and boundary values, and the packaging of a transient step.

The semi-discrete system `M u' + K u = b(t)` with natural boundaries conserves nothing
but obeys `1ᵀ M u' = 1ᵀ b(t)` exactly (K annihilates constants), so the mean of `u` is
the time integral of the mean source: an exact reference for the integrators that
isolates how they treat a time-dependent load.
"""
import numpy as np

from fem.field import NodalField
import pytest

from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial
from fem.physics.equations import Heat, LinearElastic, Poisson, Wave
from fem.algebra.integrators import NewmarkMethod, ThetaMethod
from fem.regions import TimeDependent, everywhere, evaluate_field, field_at, on_plane
from fem.post.solution import ElasticSolution, FieldSolution, DiffusionSolution, TransientSolution, WaveSolution
from fem.space import FunctionSpace
from fem.loads import Source


def test_time_dependent_field_is_evaluated_at_a_time():
    field = TimeDependent(lambda p, t: p[0] * t)
    points = np.array([[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(evaluate_field(field_at(field, 3.0), points, 1)[:, 0], [3.0, 6.0])
    assert field_at(2.5, 3.0) == 2.5
    with pytest.raises(TypeError, match='TimeDependent'):
        evaluate_field(field, points, 1)


def test_a_time_independent_problem_has_one_load(make_unit_square):
    problem = Poisson().problem(make_unit_square(4), Conditions(Source(lambda p: 1.0)))
    assert not problem.is_time_dependent
    assert problem.load_at(5.0) is problem.load
    assert problem.constraints_at(5.0) == problem.constraints


def test_constant_time_dependent_source_matches_the_steady_load(make_unit_square):
    mesh = make_unit_square(5)
    steady = Poisson().problem(mesh, Conditions(Source(lambda p: 1.0)))
    transient = Poisson().problem(mesh, Conditions(Source(TimeDependent(lambda p, t: 1.0 + 0.0 * t))))
    assert transient.is_time_dependent
    np.testing.assert_allclose(transient.load_at(7.0), steady.load, atol=1e-14)


def test_time_dependent_traction_and_robin_values_are_taken_at_the_time(make_unit_square):
    """The boundary integrals are held; only the values are re-evaluated per time."""
    mesh = make_unit_square(5)
    at_two = Conditions(
        Neumann(on_plane(0, 1.0), 2.0),
        Robin(on_plane(1, 1.0), kappa=0.5, g=4.0),
    )
    varying = Conditions(
        Neumann(on_plane(0, 1.0), TimeDependent(lambda p, t: t)),
        Robin(on_plane(1, 1.0), kappa=0.5, g=TimeDependent(lambda p, t: 2 * t)),
    )
    reference = Poisson().problem(mesh, at_two)
    problem = Poisson().problem(mesh, varying)
    np.testing.assert_allclose(problem.load_at(2.0), reference.load, atol=1e-14)
    np.testing.assert_allclose(problem.load_at(0.0), 0.0, atol=1e-14)


def _mean(problem, u):
    return NodalField(problem.space, u).mean()


def test_theta_method_integrates_a_time_dependent_source_to_second_order(make_unit_square):
    """With natural boundaries and a uniform source sin(t), the mean of u is 1 - cos(t)
    exactly; Crank-Nicolson's trapezoid on the source converges at second order."""
    mesh = make_unit_square(6)
    problem = Heat().problem(mesh, Conditions(Source(TimeDependent(lambda p, t: np.sin(t)))))
    T = 1.0
    errors = []
    for steps in (5, 10, 20):
        solution = ThetaMethod(dt=T / steps, steps=steps).solve(problem)
        errors.append(abs(_mean(problem, solution.dofs[-1]) - (1 - np.cos(T))))
    orders = np.log(np.array(errors[:-1]) / errors[1:]) / np.log(2)
    assert np.all(orders > 1.9), orders


def test_theta_method_follows_time_dependent_dirichlet_data(make_unit_square):
    """Boundary held at g(t) = 1 + t with source g'(t) = 1: u(x, t) = 1 + t is the exact
    discrete solution for every theta, since K annihilates constants."""
    mesh = make_unit_square(5)
    bc = Conditions(Dirichlet(everywhere(), TimeDependent(lambda p, t: 1.0 + t)))
    problem = Heat().problem(mesh, bc + Source(TimeDependent(lambda p, t: 1.0)))
    assert problem.is_time_dependent
    np.testing.assert_allclose(problem.constraints_at(2.0)[2], 3.0)
    for theta in (0.5, 1.0):
        solution = ThetaMethod(dt=0.1, steps=10, theta=theta).solve(problem, initial=Initial(1.0))
        np.testing.assert_allclose(solution.dofs[-1], 2.0, atol=1e-10)
        np.testing.assert_allclose(solution.dofs[5], 1.5, atol=1e-10)


def test_newmark_integrates_a_time_dependent_source(make_unit_square):
    """A uniform constant source f under natural boundaries gives mean u = f t^2 / 2
    exactly; average acceleration is exact for a constant acceleration. A source
    sin(t) gives mean u = t - sin(t), converging at second order."""
    mesh = make_unit_square(6)

    constant = Wave(stiffness=1.0).problem(mesh, Conditions(Source(TimeDependent(lambda p, t: 2.0 + 0.0 * t))))
    solution = NewmarkMethod(dt=0.05, steps=20).solve(constant)
    assert _mean(constant, solution.dofs[-1]) == pytest.approx(1.0, abs=1e-10)

    forced = Wave(stiffness=1.0).problem(mesh, Conditions(Source(TimeDependent(lambda p, t: np.sin(t)))))
    T = 1.0
    errors = []
    for steps in (5, 10, 20):
        solution = NewmarkMethod(dt=T / steps, steps=steps).solve(forced)
        errors.append(abs(_mean(forced, solution.dofs[-1]) - (T - np.sin(T))))
    orders = np.log(np.array(errors[:-1]) / errors[1:]) / np.log(2)
    assert np.all(orders > 1.9), orders


def test_newmark_refuses_time_dependent_dirichlet_data(make_unit_square):
    mesh = make_unit_square(4)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: t)))
    problem = Wave(stiffness=1.0).problem(mesh, bc)
    with pytest.raises(NotImplementedError, match='Dirichlet'):
        NewmarkMethod(dt=0.01, steps=1).solve(problem)


def test_transient_solution_packages_a_step_as_the_typed_steady_solution(make_unit_square, tmp_path):
    """A heat step carries the gradient, an elastic step the stress; a loaded series,
    which has no operator, packages a bare field."""
    mesh = make_unit_square(4)
    heat = Heat().problem(mesh, Conditions(Source(1.0)))
    history = ThetaMethod(dt=0.1, steps=3).solve(heat)
    assert isinstance(history, TransientSolution)
    assert len(history) == 4 and history.dofs.shape == (4, heat.space.n_dofs)
    step = history[2]
    assert isinstance(step, DiffusionSolution)
    np.testing.assert_array_equal(step.dofs, history.dofs[2])
    np.testing.assert_allclose(step.gradient, heat.space.gradient(history.dofs[2]))
    assert isinstance(history[-1], DiffusionSolution)
    np.testing.assert_array_equal(history[-1].dofs, history.dofs[-1])
    assert [type(s) for s in history] == [DiffusionSolution] * 4
    with pytest.raises(TypeError, match='indexed by step'):
        history[1:2]  # type: ignore[index]

    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]))
    elastic = LinearElastic(E=10.0, nu=0.3).problem(mesh, bc + Source([0.0, -1.0]))
    waves = NewmarkMethod(dt=0.01, steps=2).solve(elastic)
    assert isinstance(waves[-1], ElasticSolution)
    assert waves[-1].stress.shape == (len(mesh.elements), 3, 3)
    np.testing.assert_array_equal(waves.velocity(1).dofs, waves.dudt[1])

    path = tmp_path / 'heat.npz'
    history.save(str(path))
    loaded = TransientSolution.load(str(path))
    assert isinstance(loaded, TransientSolution) and loaded.operator is None
    assert type(loaded[1]) is FieldSolution
    np.testing.assert_array_equal(loaded[1].dofs, history.dofs[1])


def test_transient_solution_checks_its_shape(make_unit_square):
    """One row per step on the space, and one time per row."""
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='n_steps'):
        TransientSolution(space, np.array([0.0, 1.0]), np.zeros((2, space.n_dofs + 1)))
    with pytest.raises(ValueError, match='n_steps'):
        TransientSolution(space, np.array([0.0]), np.zeros((2, space.n_dofs)))
    with pytest.raises(ValueError, match='dudt'):
        WaveSolution(space, np.array([0.0]), np.zeros((1, space.n_dofs)), np.zeros((2, space.n_dofs)))


def test_snapshot_at_a_time_is_a_steady_problem(make_unit_square):
    """`problem.at(t)` fixes every time-dependent value; a steady solve needs a `t`, and
    an estimator takes the snapshot."""
    from fem.analysis.estimators import RecoveryEstimator
    mesh = make_unit_square(5)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: t)))
    problem = Poisson().problem(mesh, bc + Source(TimeDependent(lambda p, t: 2.0 * t)))

    reference_bc = Conditions(Dirichlet(on_plane(0, 0.0), 3.0))
    reference = Poisson().problem(mesh, reference_bc + Source(lambda p: 6.0))

    snapshot = problem.at(3.0)
    assert not snapshot.is_time_dependent and problem.is_time_dependent
    np.testing.assert_allclose(snapshot.load, reference.load, atol=1e-14)
    np.testing.assert_allclose(snapshot.constraints[2], reference.constraints[2])
    assert snapshot.load_at(0.0) is snapshot.load

    with pytest.raises(ValueError, match='pass t='):
        problem.solve()
    np.testing.assert_allclose(problem.solve(t=3.0).dofs, reference.solve().dofs, atol=1e-12)

    solution = snapshot.solve()
    with pytest.raises(ValueError, match='problem.at'):
        RecoveryEstimator().estimate(problem, solution)
    assert RecoveryEstimator().estimate(snapshot, solution).shape == (len(mesh.elements),)


def test_bc_plot_labels_a_time_dependent_value(make_unit_square):
    from fem.plot.bc import _classify
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: 1.0 + t)),
        Neumann(on_plane(0, 1.0), 2.0),
    )
    marks, _ = _classify(bc, make_unit_square(4))
    assert 'varies in time' in marks[0].label and 'varies' not in marks[1].label
