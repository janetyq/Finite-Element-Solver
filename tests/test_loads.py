"""Composable operator and load terms: a sum of forms assembles as the sum of its
terms over their own domains, the load is a sum of `Load` terms, a point load is a
nodal force, and Rayleigh damping decays a free vibration at the analytic rate.
"""
import numpy as np
import pytest

from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial
from fem.elements import QuadraticTriangleElement
from fem.physics.energies import StVenantKirchhoff
from fem.physics.equations import LinearElastic, Poisson
from fem.physics.forms import EnergyForm, DiffusionForm, LinearElasticForm, BoundaryMassForm, MassForm, SumForm
from fem.algebra.integrators import NewmarkMethod
from fem.loads import BoundaryLoad, PointLoad, Source
from fem.physics.materials import LinearElasticMaterial
from fem.analysis.modal import ModalAnalysis
from fem.mesh.structured import box_mesh
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
    assert linear.flux() is not None
    assert linear.near_null_space(space).shape == (space.n_dofs, 3)

    nonlinear = EnergyForm(StVenantKirchhoff(200.0, 0.3)) + spring
    assert not nonlinear.constant_tangent and nonlinear.has_energy
    with pytest.raises(TypeError, match='element blocks'):
        nonlinear.element_tangents(space.geometry, np.zeros((1, 3, 2)))
    with pytest.raises(ValueError, match='more than one'):
        (elastic + elastic).flux()


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
    from fem.post.solution import DiffusionSolution
    mesh = make_unit_square(6)
    bc = Conditions(Robin(everywhere(), kappa=2.0, g=1.0))
    problem = Poisson().problem(mesh, bc + Source(1.0))

    assert isinstance(problem.operator, SumForm) and len(problem.operator.terms) == 2
    assert isinstance(problem.physics, DiffusionForm)
    assert isinstance(problem.solve(), DiffusionSolution)
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
    assert problem.solve().component(1)[tip] < 0

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
    solution = NewmarkMethod(dt=dt, steps=steps).solve(problem, initial=Initial(modal.mode(0)))

    q = np.array([phi @ (M @ u) / (phi @ (M @ phi)) for u in solution.dofs])
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
    start = Initial(lambda p: [0.0, 0.01 * p[:, 0]])
    a = NewmarkMethod(dt=0.01, steps=5).solve(plain, initial=start)
    b = NewmarkMethod(dt=0.01, steps=5).solve(zero, initial=start)
    np.testing.assert_allclose(a.dofs[-1], b.dofs[-1], atol=1e-12)
    with pytest.raises(ValueError, match='non-negative'):
        RayleighDamping(alpha=-1.0)


def test_with_operator_resets_the_damping_matrix(make_unit_square):
    mesh = make_unit_square(4)
    problem = LinearElastic(E=10.0, nu=0.3, damping=RayleighDamping(beta=0.1)).problem(mesh)
    C = problem.damping_matrix
    derived = problem.with_operator(2.0 * problem.physics)
    np.testing.assert_allclose(derived.damping_matrix.toarray(), 2.0 * C.toarray(), atol=1e-12)
    assert isinstance(derived, LinearProblem)


# -- resultants -----------------------------------------------------------------


@pytest.mark.parametrize('dim', [2, 3])
def test_a_uniform_source_sums_to_the_source_times_the_volume(dim):
    """The load is int f phi_i and the shape functions sum to one, so its total is f
    times the measure, per component, in any dimension. The unit box has unit measure."""
    mesh = box_mesh(corners=[[0] * dim, [1] * dim], resolution=(4,) * dim)
    scalar = FunctionSpace(mesh).assemble_load(Source(3.0))
    np.testing.assert_allclose(scalar.sum(), 3.0, atol=1e-12)

    f = [1.0, -2.0, 0.5][:dim]
    vector = FunctionSpace(mesh, n_components=dim).assemble_load(Source(f))
    np.testing.assert_allclose(vector.reshape(-1, dim).sum(axis=0), f, atol=1e-12)


@pytest.mark.parametrize('space_for', [
    lambda: FunctionSpace(box_mesh([[0, 0], [1, 1]], (5, 5))),
    lambda: FunctionSpace(box_mesh([[0, 0], [1, 1]], (5, 5)), QuadraticTriangleElement),
    lambda: FunctionSpace(box_mesh([[0, 0, 0], [1, 1, 1]], (3, 3, 3)), n_components=3),
], ids=['P1', 'P2', '3D vector'])
def test_a_constant_source_integrates_element_wise_without_the_mass_matrix(space_for):
    """A constant load is the mass matrix times the constant's nodal vector, but it is
    integrated element by element so the load never assembles the mass matrix, which
    is a full block scatter the steady solve otherwise has no use for."""
    space = space_for()
    value = 2.5 if space.n_components == 1 else [1.0, -2.0, 0.5]
    load = Source(value).vector(space)
    assert 'mass_matrix' not in space.__dict__
    expected = space.mass_matrix @ space.interpolate(value).dofs
    np.testing.assert_allclose(load, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('dim', [2, 3])
def test_a_uniform_traction_sums_to_the_traction_times_the_loaded_measure(dim):
    """The face x = 1 of the unit box has unit measure (an edge in 2D, a square in 3D),
    so the traction's resultant is the traction itself, per component, and nothing
    lands on a node off that face."""
    mesh = box_mesh(corners=[[0] * dim, [1] * dim], resolution=(4,) * dim)
    traction = [0.0, -1.0, 0.5][:dim]
    space = FunctionSpace(mesh, n_components=dim)
    bc = Conditions(Neumann(on_plane(0, 1.0), traction))
    operator = LinearElasticForm(LinearElasticMaterial(1.0, 0.3))
    load = LinearProblem(space, operator, bc).load.reshape(-1, dim)

    np.testing.assert_allclose(load.sum(axis=0), traction, atol=1e-12)
    off_face = mesh.vertices[:, 0] < 1.0 - 1e-9
    np.testing.assert_allclose(load[off_face], 0.0, atol=1e-12)
