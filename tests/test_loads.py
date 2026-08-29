"""Composable operator and load terms: a sum of forms assembles as the sum of its
terms over their own domains, the load is a sum of `Load` terms, a point load is a
nodal force, and Rayleigh damping decays a free vibration at the analytic rate.
"""
import numpy as np
import pytest

from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions
from fem.physics.energies import StVenantKirchhoff
from fem.physics.equations import LinearElastic, Poisson
from fem.physics.forms import EnergyForm, DiffusionForm, LinearElasticForm, BoundaryMassForm, MassForm, SumForm
from fem.algebra.integrators import NewmarkMethod
from fem.loads import BoundaryLoad, PointLoad, Source
from fem.physics.materials import LinearElasticMaterial
from fem.analysis.modal import ModalAnalysis
from fem.problem import LinearProblem, RayleighDamping
from fem.regions import TimeDependent, at_indices, everywhere, on_plane
from fem.space import FunctionSpace


# -- forms compose --------------------------------------------------------------


def test_a_sum_of_forms_assembles_to_the_sum_of_its_terms(make_unit_square):
    space = FunctionSpace(make_unit_square(6))
    mask = np.ones(len(space.boundary_nodes), dtype=bool)
    operator = DiffusionForm() + 3.0 * BoundaryMassForm(1, mask)

    assert isinstance(operator, SumForm) and len(operator.terms) == 2
    expected = space.assemble(DiffusionForm()) + 3.0 * space.assemble(MassForm(), boundary=True)
    np.testing.assert_allclose(space.assemble(operator).toarray(), expected.toarray(), atol=1e-12)


def test_a_sum_answers_the_hooks_from_its_terms(make_unit_square):
    space = FunctionSpace(make_unit_square(4), n_components=2)
    mask = np.ones(len(space.boundary_nodes), dtype=bool)
    elastic = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))
    spring = 5.0 * BoundaryMassForm(2, mask)

    linear = elastic + spring
    assert linear.constant_tangent and linear.has_energy
    assert linear.derived_field() is not None
    assert linear.near_null_space(space).shape == (space.n_dofs, 3)

    nonlinear = EnergyForm(StVenantKirchhoff(200.0, 0.3)) + spring
    assert not nonlinear.constant_tangent and nonlinear.has_energy
    with pytest.raises(TypeError, match='element blocks'):
        nonlinear.element_tangents(space.geometry, np.zeros((1, 3, 2)))
    with pytest.raises(ValueError, match='more than one'):
        (elastic + elastic).derived_field()


def test_a_state_dependent_sum_has_consistent_energy_residual_and_tangent(make_unit_square):
    """A boundary spring on a finite-strain operator: the residual is the gradient of the
    energy and the tangent the gradient of the residual, both terms included."""
    from fem.numerics import central_difference_order
    space = FunctionSpace(make_unit_square(4), n_components=2)
    mask = np.ones(len(space.boundary_nodes), dtype=bool)
    operator = EnergyForm(StVenantKirchhoff(200.0, 0.3)) + 20.0 * BoundaryMassForm(2, mask)
    rng = np.random.default_rng(0)
    u = 0.05 * rng.standard_normal(space.n_dofs)

    def energy(w):
        return space.total_energy(operator, w)

    def residual(w):
        return space.assemble_residual(operator, w)

    tangent = space.assemble_tangent(operator, u)
    assert 1.9 < central_difference_order(energy, lambda d: residual(u) @ d, u) < 2.1
    assert 1.9 < central_difference_order(residual, lambda d: tangent @ d, u) < 2.1


def test_the_robin_term_is_a_term_of_the_operator(make_unit_square):
    """A problem with a Robin condition has one boundary term in its operator, which the
    physics form, the packaging, and `with_operator` all see."""
    from fem.post.solution import ScalarFieldSolution
    mesh = make_unit_square(6)
    bc = Conditions(Robin(everywhere(), kappa=2.0, g=1.0))
    problem = Poisson().problem(mesh, bc + Source(1.0))

    assert isinstance(problem.operator, SumForm) and len(problem.operator.terms) == 2
    assert isinstance(problem.physics, DiffusionForm)
    assert isinstance(problem.solve(), ScalarFieldSolution)
    derived = problem.with_operator(2.0 * DiffusionForm())
    assert len(derived.operator.terms) == 2
    expected = 2.0 * problem.space.assemble(DiffusionForm()) + problem.space.assemble(
        problem.operator.terms[1])
    np.testing.assert_allclose(derived.tangent().toarray(), expected.toarray(), atol=1e-12)


# -- loads compose --------------------------------------------------------------


def test_the_load_is_the_sum_of_its_terms(make_unit_square):
    mesh = make_unit_square(6)
    bc = Conditions(
        Neumann(on_plane(0, 1.0), 2.0),
        Robin(on_plane(1, 1.0), kappa=1.0, g=3.0),
    )
    problem = Poisson().problem(mesh, bc + Source(1.0))
    kinds = [type(term) for term in problem.loads]
    assert kinds == [Source, BoundaryLoad, BoundaryLoad]
    total = sum(term.vector(problem.space) for term in problem.loads)
    np.testing.assert_allclose(problem.load, total, atol=1e-14)


def test_a_point_load_is_a_nodal_force(make_unit_square):
    mesh = make_unit_square(8)
    tip = int(np.argmin(np.linalg.norm(mesh.vertices - [1.0, 1.0], axis=1)))
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]))
    equation = LinearElastic(E=100.0, nu=0.3)
    problem = equation.problem(mesh, bc + PointLoad(at_indices([tip]), [0.0, -1.0]))

    load = problem.load.reshape(-1, 2)
    assert load[tip, 1] == -1.0 and np.count_nonzero(load) == 1
    assert problem.solve().u.reshape(-1, 2)[tip, 1] < 0

    with pytest.raises(ValueError, match='selects no node'):
        PointLoad(at_indices([]), [0.0, -1.0]).vector(problem.space)


def test_a_neumann_condition_on_a_lone_node_is_refused(make_unit_square):
    mesh = make_unit_square(6)
    bc = Conditions(Neumann(at_indices([0]), 1.0))
    with pytest.raises(ValueError, match='PointLoad'):
        Poisson().problem(mesh, bc)


def test_time_dependent_terms_are_evaluated_per_time(make_unit_square):
    mesh = make_unit_square(5)
    tip = 0
    ramp = PointLoad(at_indices([tip]), TimeDependent(lambda p, t: [0.0, -t]))
    problem = LinearElastic(E=1.0, nu=0.3).problem(mesh, Conditions(ramp))
    assert problem.is_time_dependent
    assert problem.load_at(3.0).reshape(-1, 2)[tip, 1] == -3.0
    snapshot = problem.at(2.0)
    assert not snapshot.is_time_dependent
    assert snapshot.load.reshape(-1, 2)[tip, 1] == -2.0

    time_form = Source(TimeDependent(lambda p, t: t))
    assert time_form.is_time_dependent
    space = FunctionSpace(mesh)
    np.testing.assert_allclose(time_form.vector(space, 2.0), 2.0 * Source(1.0).vector(space))
    with pytest.raises(TypeError, match='at\\(t\\)'):
        time_form.element_vectors(space.geometry, 1)


def test_a_traction_holds_its_boundary_mass_across_times(make_unit_square):
    mesh = make_unit_square(5)
    bc = Conditions(Neumann(on_plane(0, 1.0), TimeDependent(lambda p, t: t)))
    problem = Poisson().problem(mesh, bc)
    traction = problem.loads[0]
    assert isinstance(traction, BoundaryLoad)
    np.testing.assert_allclose(problem.load_at(2.0).sum(), 2.0, atol=1e-12)
    np.testing.assert_allclose(problem.load_at(0.5), 0.25 * problem.load_at(2.0), atol=1e-12)


# -- damping ----------------------------------------------------------------------


def test_rayleigh_damping_decays_a_mode_at_the_analytic_rate(make_unit_square):
    """Under mass-proportional damping C = alpha M, a mode of frequency omega decays as
    exp(-alpha t / 2) cos(omega_d t), omega_d = omega sqrt(1 - zeta^2), zeta = alpha / (2 omega).
    Newmark is second order, so a fine step reproduces the modal coordinate closely."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]))
    alpha = 0.8
    equation = LinearElastic(E=100.0, nu=0.3, density=1.0, damping=RayleighDamping(alpha=alpha))
    problem = equation.problem(mesh, bc)
    modal = ModalAnalysis(n_modes=1).solve(problem)
    omega, phi = float(modal.angular_frequencies[0]), modal.modes[0]
    M = problem.mass

    zeta = alpha / (2 * omega)
    omega_d = omega * np.sqrt(1 - zeta**2)
    period = 2 * np.pi / omega
    steps = 400
    dt = 2 * period / steps
    solution = NewmarkMethod(dt=dt, steps=steps).solve(problem, phi.copy(), np.zeros_like(phi))

    q = np.array([phi @ (M @ u) / (phi @ (M @ phi)) for u in solution.u])
    t = solution.t
    exact = np.exp(-alpha * t / 2) * (np.cos(omega_d * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t))
    assert np.abs(q - exact).max() < 5e-3
    assert q[-1] ** 2 < 0.5 * q[0] ** 2  # it did decay


def test_undamped_is_the_no_damping_path(make_unit_square):
    mesh = make_unit_square(4)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]))
    plain = LinearElastic(E=10.0, nu=0.3).problem(mesh, bc)
    zero = LinearElastic(E=10.0, nu=0.3, damping=RayleighDamping()).problem(mesh, bc)
    assert plain.damping_matrix is None
    u0 = plain.space.interpolate(lambda p: [0.0, 0.01 * p[0]])
    v0 = np.zeros_like(u0)
    a = NewmarkMethod(dt=0.01, steps=5).solve(plain, u0, v0)
    b = NewmarkMethod(dt=0.01, steps=5).solve(zero, u0, v0)
    np.testing.assert_allclose(a.u[-1], b.u[-1], atol=1e-12)
    with pytest.raises(ValueError, match='non-negative'):
        RayleighDamping(alpha=-1.0)


def test_with_operator_resets_the_damping_matrix(make_unit_square):
    mesh = make_unit_square(4)
    problem = LinearElastic(E=10.0, nu=0.3, damping=RayleighDamping(beta=0.1)).problem(mesh)
    C = problem.damping_matrix
    derived = problem.with_operator(2.0 * problem.physics)
    np.testing.assert_allclose(derived.damping_matrix.toarray(), 2.0 * C.toarray(), atol=1e-12)
    assert isinstance(derived, LinearProblem)
