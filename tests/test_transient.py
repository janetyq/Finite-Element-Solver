"""Time-dependent loads and boundary values, and the packaging of a transient step.

The semi-discrete system `M u' + K u = b(t)` with natural boundaries conserves nothing
but obeys `1ᵀ M u' = 1ᵀ b(t)` exactly (K annihilates constants), so the mean of `u` is
the time integral of the mean source: an exact reference for the integrators that
isolates how they treat a time-dependent load.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet, Neumann, Robin
from fem.equations import LinearElastic, Poisson, Wave
from fem.integrators import NewmarkMethod, ThetaMethod
from fem.regions import TimeDependent, everywhere, evaluate_field, field_at, on_plane
from fem.solution import ElasticSolution, FieldSolution, ScalarFieldSolution, TransientSolution


def test_time_dependent_field_is_evaluated_at_a_time():
    field = TimeDependent(lambda p, t: p[0] * t)
    points = np.array([[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(evaluate_field(field_at(field, 3.0), points, 1)[:, 0], [3.0, 6.0])
    assert field_at(2.5, 3.0) == 2.5
    with pytest.raises(TypeError, match='TimeDependent'):
        evaluate_field(field, points, 1)


def test_a_time_independent_problem_has_one_load(make_unit_square):
    problem = Poisson(source=lambda p: 1.0).problem(make_unit_square(4))
    assert not problem.is_time_dependent
    assert problem.load_at(5.0) is problem.load
    assert problem.constraints_at(5.0) == problem.constraints


def test_constant_time_dependent_source_matches_the_steady_load(make_unit_square):
    mesh = make_unit_square(5)
    steady = Poisson(source=lambda p: 1.0).problem(mesh)
    transient = Poisson(source=TimeDependent(lambda p, t: 1.0 + 0.0 * t)).problem(mesh)
    assert transient.is_time_dependent
    np.testing.assert_allclose(transient.load_at(7.0), steady.load, atol=1e-14)


def test_time_dependent_traction_and_robin_values_are_taken_at_the_time(make_unit_square):
    """The boundary integrals are held; only the values are re-evaluated per time."""
    mesh = make_unit_square(5)
    at_two = BoundaryConditions()
    at_two = at_two + Neumann(on_plane(0, 1.0), 2.0)
    at_two = at_two + Robin(on_plane(1, 1.0), kappa=0.5, g=4.0)
    varying = BoundaryConditions()
    varying = varying + Neumann(on_plane(0, 1.0), TimeDependent(lambda p, t: t))
    varying = varying + Robin(on_plane(1, 1.0), kappa=0.5, g=TimeDependent(lambda p, t: 2 * t))
    reference = Poisson().problem(mesh, at_two)
    problem = Poisson().problem(mesh, varying)
    np.testing.assert_allclose(problem.load_at(2.0), reference.load, atol=1e-14)
    np.testing.assert_allclose(problem.load_at(0.0), 0.0, atol=1e-14)


def _mean(problem, u):
    return problem.space.mean_value(u)


def test_theta_method_integrates_a_time_dependent_source_to_second_order(make_unit_square):
    """With natural boundaries and a uniform source sin(t), the mean of u is 1 - cos(t)
    exactly; Crank-Nicolson's trapezoid on the source converges at second order."""
    mesh = make_unit_square(6)
    problem = Poisson(source=TimeDependent(lambda p, t: np.sin(t))).problem(mesh)
    T = 1.0
    errors = []
    for steps in (5, 10, 20):
        solution = ThetaMethod(dt=T / steps, steps=steps).solve(problem, np.zeros(problem.space.n_dofs))
        errors.append(abs(_mean(problem, solution.u[-1]) - (1 - np.cos(T))))
    orders = np.log(np.array(errors[:-1]) / errors[1:]) / np.log(2)
    assert np.all(orders > 1.9), orders


def test_theta_method_follows_time_dependent_dirichlet_data(make_unit_square):
    """Boundary held at g(t) = 1 + t with source g'(t) = 1: u(x, t) = 1 + t is the exact
    discrete solution for every theta, since K annihilates constants."""
    mesh = make_unit_square(5)
    bc = BoundaryConditions()
    bc = bc + Dirichlet(everywhere(), TimeDependent(lambda p, t: 1.0 + t))
    problem = Poisson(source=TimeDependent(lambda p, t: 1.0)).problem(mesh, bc)
    assert problem.is_time_dependent
    np.testing.assert_allclose(problem.constraints_at(2.0)[2], 3.0)
    u0 = problem.space.interpolate(1.0)
    for theta in (0.5, 1.0):
        solution = ThetaMethod(dt=0.1, steps=10, theta=theta).solve(problem, u0)
        np.testing.assert_allclose(solution.u[-1], 2.0, atol=1e-10)
        np.testing.assert_allclose(solution.u[5], 1.5, atol=1e-10)


def test_newmark_integrates_a_time_dependent_source(make_unit_square):
    """A uniform constant source f under natural boundaries gives mean u = f t^2 / 2
    exactly; average acceleration is exact for a constant acceleration. A source
    sin(t) gives mean u = t - sin(t), converging at second order."""
    mesh = make_unit_square(6)
    n = Wave(c=1.0).space(mesh).n_dofs
    zero = np.zeros(n)

    constant = Wave(c=1.0, source=TimeDependent(lambda p, t: 2.0 + 0.0 * t)).problem(mesh)
    solution = NewmarkMethod(dt=0.05, steps=20).solve(constant, zero, zero)
    assert _mean(constant, solution.u[-1]) == pytest.approx(1.0, abs=1e-10)

    forced = Wave(c=1.0, source=TimeDependent(lambda p, t: np.sin(t))).problem(mesh)
    T = 1.0
    errors = []
    for steps in (5, 10, 20):
        solution = NewmarkMethod(dt=T / steps, steps=steps).solve(forced, zero, zero)
        errors.append(abs(_mean(forced, solution.u[-1]) - (T - np.sin(T))))
    orders = np.log(np.array(errors[:-1]) / errors[1:]) / np.log(2)
    assert np.all(orders > 1.9), orders


def test_newmark_refuses_time_dependent_dirichlet_data(make_unit_square):
    mesh = make_unit_square(4)
    bc = BoundaryConditions()
    bc = bc + Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: t))
    problem = Wave(c=1.0).problem(mesh, bc)
    n = problem.space.n_dofs
    with pytest.raises(NotImplementedError, match='Dirichlet'):
        NewmarkMethod(dt=0.01, steps=1).solve(problem, np.zeros(n), np.zeros(n))


def test_transient_solution_packages_a_step_as_the_typed_steady_solution(make_unit_square, tmp_path):
    """A heat step carries the flux, an elastic step the stress; a loaded series, which
    has no problem, packages a bare field."""
    mesh = make_unit_square(4)
    heat = Poisson(source=1.0).problem(mesh)
    u0 = heat.space.interpolate(0.0)
    history = ThetaMethod(dt=0.1, steps=3).solve(heat, u0)
    assert isinstance(history, TransientSolution)
    step = history.at(2)
    assert isinstance(step, ScalarFieldSolution)
    np.testing.assert_array_equal(step.u, history.u[2])
    np.testing.assert_allclose(step.flux, heat.space.gradient(history.u[2]))
    assert isinstance(history.final, ScalarFieldSolution)
    np.testing.assert_array_equal(history.final.u, history.u[-1])

    bc = BoundaryConditions()
    bc = bc + Dirichlet(on_plane(0, 0.0), [0.0, 0.0])
    elastic = LinearElastic(E=10.0, nu=0.3, source=[0.0, -1.0]).problem(mesh, bc)
    zero = np.zeros(elastic.space.n_dofs)
    waves = NewmarkMethod(dt=0.01, steps=2).solve(elastic, zero, zero)
    assert isinstance(waves.final, ElasticSolution)
    assert waves.final.stress.shape == (len(mesh.elements), 3, 3)

    path = tmp_path / 'heat.npz'
    history.save(str(path))
    loaded = TransientSolution.load(str(path))
    assert isinstance(loaded, TransientSolution) and loaded.problem is None
    assert type(loaded.at(1)) is FieldSolution
    np.testing.assert_array_equal(loaded.at(1).u, history.u[1])


def test_snapshot_at_a_time_is_a_steady_problem(make_unit_square):
    """`problem.at(t)` fixes every time-dependent value; a steady solve needs a `t`, and
    an estimator takes the snapshot."""
    from fem.estimators import RecoveryEstimator
    mesh = make_unit_square(5)
    bc = BoundaryConditions()
    bc = bc + Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: t))
    problem = Poisson(source=TimeDependent(lambda p, t: 2.0 * t)).problem(mesh, bc)

    reference_bc = BoundaryConditions()
    reference_bc = reference_bc + Dirichlet(on_plane(0, 0.0), 3.0)
    reference = Poisson(source=lambda p: 6.0).problem(mesh, reference_bc)

    snapshot = problem.at(3.0)
    assert not snapshot.is_time_dependent and problem.is_time_dependent
    np.testing.assert_allclose(snapshot.load, reference.load, atol=1e-14)
    np.testing.assert_allclose(snapshot.constraints[2], reference.constraints[2])
    assert snapshot.load_at(0.0) is snapshot.load

    with pytest.raises(ValueError, match='pass t='):
        problem.solve()
    np.testing.assert_allclose(problem.solve(t=3.0).u, reference.solve().u, atol=1e-12)

    solution = snapshot.solve()
    with pytest.raises(ValueError, match='problem.at'):
        RecoveryEstimator().estimate(problem, solution)
    assert RecoveryEstimator().estimate(snapshot, solution).shape == (len(mesh.elements),)


def test_bc_plot_labels_a_time_dependent_value(make_unit_square):
    from fem.plot.bc import _classify
    bc = BoundaryConditions()
    bc = bc + Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: 1.0 + t))
    bc = bc + Neumann(on_plane(0, 1.0), 2.0)
    marks, _ = _classify(bc, make_unit_square(4))
    assert 'varies in time' in marks[0].label and 'varies' not in marks[1].label
