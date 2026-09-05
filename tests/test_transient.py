"""Time-dependent loads and boundary values, and the packaging of a transient step.

The semi-discrete system `M u' + K u = b(t)` with natural boundaries conserves nothing
but obeys `1ᵀ M u' = 1ᵀ b(t)` exactly (K annihilates constants), so the mean of `u` is
the time integral of the mean source: an exact reference for the integrators that
isolates how they treat a time-dependent load.

The nonlinear path (a state-dependent tangent, solved by Newton per step) is checked
against the linear one where the two must agree, and against the invariants only
finite-strain kinematics has: bounded energy drift and a stress-free rigid rotation.
"""
import numpy as np
import pytest

from fem.algebra.integrators import NewmarkMethod, ThetaMethod, wave_energy
from fem.algebra.solve import NewtonDivergence
from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial
from fem.field import NodalField
from fem.loads import Source
from fem.physics.energies import SmallStrain
from fem.physics.equations import FiniteStrainElastic, Heat, LinearElastic, Poisson, Wave
from fem.physics.forms import EnergyForm, LinearElasticForm
from fem.physics.materials import LinearElasticMaterial
from fem.post.solution import DiffusionSolution, ElasticSolution, FieldSolution, TransientSolution, WaveSolution
from fem.problem import LinearProblem, Problem, RayleighDamping
from fem.regions import TimeDependent, evaluate_field, everywhere, field_at, on_plane
from fem.space import FunctionSpace


def test_time_dependent_field_is_evaluated_at_a_time():
    field = TimeDependent(lambda p, t: p[:, 0] * t)
    points = np.array([[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(evaluate_field(field_at(field, 3.0), points, 1)[:, 0], [3.0, 6.0])
    assert field_at(2.5, 3.0) == 2.5
    with pytest.raises(TypeError, match='TimeDependent'):
        evaluate_field(field, points, 1)


def test_a_time_independent_problem_has_one_load(make_unit_square):
    problem = Poisson().problem(make_unit_square(4), Conditions(Source(lambda p: 1.0)))
    assert not problem.is_time_dependent
    assert problem.load_at(5.0) is problem.load
    assert problem.fixed_values_at(5.0) is problem.fixed_values


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
    np.testing.assert_allclose(problem.fixed_values_at(2.0), 3.0)
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


def test_newmark_gamma_sets_the_numerical_damping(make_unit_square):
    """gamma = 1/2 (average acceleration) conserves the energy of a free vibration; gamma
    above 1/2 introduces numerical dissipation, so the energy decays monotonically. A
    seeded standing wave with no source or damping isolates the scheme's own dissipation.
    """
    mesh = make_unit_square(8)
    problem = Wave(stiffness=1.0).problem(mesh, Conditions(Dirichlet(everywhere(), 0.0)))
    x, y = mesh.vertices[:, 0], mesh.vertices[:, 1]
    u0 = NodalField(problem.space, np.sin(np.pi * x) * np.sin(np.pi * y))

    def energies(gamma, beta):
        run = NewmarkMethod(dt=0.02, steps=80, gamma=gamma, beta=beta).solve(
            problem, initial=Initial(u0))
        return np.array([wave_energy(problem, run.dofs[i], run.dudt[i]) for i in range(len(run))])

    conserving = energies(0.5, 0.25)
    assert (conserving.max() - conserving.min()) < 1e-9 * conserving[0], 'energy not conserved'

    # gamma > 1/2 with the matching beta = (gamma + 1/2)^2 / 4 stays stable and dissipative.
    damping = energies(0.6, (0.6 + 0.5) ** 2 / 4)
    assert np.all(np.diff(damping) <= 1e-12), f'energy did not decay monotonically: {damping}'
    assert damping[-1] < 0.98 * damping[0], f'no numerical dissipation at gamma=0.6: {damping}'


def test_newmark_refuses_time_dependent_dirichlet_data(make_unit_square):
    mesh = make_unit_square(4)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: t)))
    problem = Wave(stiffness=1.0).problem(mesh, bc)
    with pytest.raises(NotImplementedError, match='Dirichlet'):
        NewmarkMethod(dt=0.01, steps=1).solve(problem)


E, NU = 200.0, 0.3
CLAMPED = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]))


def _rigid_rotation(mesh, angle):
    """The displacement field of a rigid rotation by `angle` about the mesh centroid."""
    center = mesh.vertices.mean(axis=0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return ((mesh.vertices - center) @ rotation.T + center - mesh.vertices).ravel()


def test_newmark_on_a_small_strain_energy_matches_the_linear_operator(make_unit_square):
    """`SmallStrain` is the same physics `LinearElastic` assembles directly, stated as an
    energy density: its tangent is state-dependent as far as the integrator is concerned,
    so the step is solved by Newton, but the physics is exactly linear. The two runs must
    therefore agree to round-off, displacement and velocity alike."""
    mesh = make_unit_square(4)
    conditions = CLAMPED + Source([0.0, -2.0])
    linear = LinearElastic(E, NU).problem(mesh, conditions)
    energy = FiniteStrainElastic(E, NU, law=SmallStrain).problem(mesh, conditions)
    assert linear.is_linear and not energy.is_linear

    run = NewmarkMethod(dt=0.01, steps=6)
    reference, through_newton = run.solve(linear), run.solve(energy)
    np.testing.assert_allclose(through_newton.dofs, reference.dofs, atol=1e-13)
    np.testing.assert_allclose(through_newton.dudt, reference.dudt, atol=1e-12)


def test_theta_method_on_a_small_strain_energy_matches_the_linear_operator(make_unit_square):
    """The same parity for the first-order scheme, on the gradient flow
    `M u' + r_int(u) = b`: an elastic operator under a first-order time derivative, which
    is the shape `ThetaMethod` integrates, stated once each way."""
    mesh = make_unit_square(4)
    conditions = CLAMPED + Source([0.0, -2.0])
    space = LinearElastic(E, NU).space(mesh)
    first_order = frozenset({1})
    linear = LinearProblem(space, LinearElasticForm(LinearElasticMaterial(E, NU)), conditions,
                           time_orders=first_order)
    energy = Problem(space, EnergyForm(SmallStrain(E, NU)), conditions, time_orders=first_order)

    run = ThetaMethod(dt=0.02, steps=5)
    np.testing.assert_allclose(run.solve(energy).dofs, run.solve(linear).dofs, atol=1e-13)


def test_newmark_finite_strain_matches_linear_elasticity_under_a_small_load(make_unit_square):
    """Green-Lagrange strain differs from the infinitesimal one at O(‖∇u‖²), so a load
    small enough to keep the motion in the small-strain regime makes the finite-strain
    and linear runs agree; here to 0.1% of the peak displacement over the whole series."""
    mesh = make_unit_square(4)
    conditions = CLAMPED + Source([0.0, -0.2])
    run = NewmarkMethod(dt=0.01, steps=10)
    linear = run.solve(LinearElastic(E, NU).problem(mesh, conditions))
    finite = run.solve(FiniteStrainElastic(E, NU).problem(mesh, conditions))

    peak = np.abs(linear.dofs).max()
    assert np.abs(finite.dofs - linear.dofs).max() < 1e-3 * peak


def test_newmark_conserves_the_energy_of_a_finite_strain_vibration(make_unit_square):
    """Average acceleration is energy-conserving for a linear system and bounded-drift
    for a nonlinear one: a St-Venant-Kirchhoff body released from a sheared state, with
    no load and no damping, exchanges stored and kinetic energy while the total stays
    within a fraction of a percent of its initial value."""
    mesh = make_unit_square(4)
    problem = FiniteStrainElastic(E, NU).problem(mesh, CLAMPED)
    sheared = np.zeros((len(mesh.vertices), 2))
    sheared[:, 1] = 0.02 * mesh.vertices[:, 0]
    released = Initial(NodalField(problem.space, sheared.ravel()))

    run = NewmarkMethod(dt=0.005, steps=60).solve(problem, initial=released)
    energies = np.array([wave_energy(problem, run.dofs[i], run.dudt[i]) for i in range(len(run))])
    assert energies.min() > 0.0
    drift = (energies.max() - energies.min()) / energies[0]
    assert drift < 1e-3, f'energy drifted by {drift:.2e} over the run: {energies}'


def test_newmark_leaves_a_rigidly_rotated_finite_strain_body_at_rest(make_unit_square):
    """A rigid rotation is strain-free in Green-Lagrange kinematics, so an unconstrained
    body started in a rotated configuration has zero internal force and stays there: the
    integrator's Newton solve returns the predictor at every step and the stress stays at
    round-off. Infinitesimal strain reads the same rotation as a strain of O(θ²/2), so the
    linear run develops stress and starts moving. Free-free: with no Dirichlet condition
    the effective operator is still SPD, the mass term carrying it."""
    mesh = make_unit_square(4)
    rotated = _rigid_rotation(mesh, 0.3)
    scale = np.abs(rotated).max()
    run = NewmarkMethod(dt=0.01, steps=5)

    problem = FiniteStrainElastic(E, NU).problem(mesh, Conditions())
    assert problem.partition.fixed.size == 0
    seed = Initial(NodalField(problem.space, rotated))
    finite = run.solve(problem, initial=seed)
    np.testing.assert_allclose(finite.dofs[-1], rotated, atol=1e-12)
    assert np.abs(finite[-1].stress).max() < 1e-9 * E

    linear = run.solve(LinearElastic(E, NU).problem(mesh, Conditions()), initial=seed)
    assert np.abs(linear.dofs[-1] - rotated).max() > 0.1 * scale
    assert np.abs(linear[-1].stress).max() > 0.1 * E


def test_newmark_refuses_to_damp_a_state_dependent_tangent(make_unit_square):
    """`RayleighDamping` is C = alpha M + beta K over one constant stiffness, which a
    finite-strain operator has not got; the refusal is explicit rather than a damping
    matrix built from an arbitrary tangent."""
    mesh = make_unit_square(3)
    problem = FiniteStrainElastic(E, NU, damping=RayleighDamping(alpha=0.1)).problem(mesh, CLAMPED)
    with pytest.raises(TypeError, match='RayleighDamping'):
        NewmarkMethod(dt=0.01, steps=1).solve(problem)


def test_a_step_that_will_not_converge_reports_the_step_and_advises_a_smaller_dt(make_unit_square):
    """One Newton iteration cannot resolve a loaded step, so the step raises rather than
    recording an unconverged state; the exception carries the last iterate."""
    mesh = make_unit_square(3)
    problem = FiniteStrainElastic(E, NU).problem(mesh, CLAMPED + Source([0.0, -2.0]))
    with pytest.raises(NewtonDivergence, match='Reduce dt') as raised:
        NewmarkMethod(dt=0.05, steps=2, newton_max_iters=1).solve(problem)
    assert raised.value.iterations == 1 and raised.value.u.shape == (problem.space.n_dofs,)

    with pytest.raises(NewtonDivergence, match='Reduce dt'):
        space = problem.space
        first_order = Problem(space, EnergyForm(SmallStrain(E, NU)),
                              CLAMPED + Source([0.0, -2.0]), time_orders=frozenset({1}))
        ThetaMethod(dt=0.05, steps=2, newton_max_iters=1).solve(first_order)


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
    np.testing.assert_allclose(snapshot.fixed_values, reference.fixed_values)
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
