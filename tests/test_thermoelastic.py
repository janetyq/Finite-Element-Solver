"""Thermoelasticity: an eigenstrain in the elastic law, and its plane-strain reduction.

The law is `sigma = C : (eps - alpha dT I)`. The stiffness is unchanged; the thermal
term is a load the form carries and a correction to the recovered stress. The trap the
tests guard is the 2D reduction: a plane-strain solve blocks the out-of-plane
expansion, and that blocked expansion loads the in-plane stress, so the thermal
coefficient is the 3D `beta = (3 lambda + 2 mu) alpha` in the plane too. The closed
forms (a clamped body, free expansion, a bar between walls) are exact for P1, so they
are asserted to round-off; the rate is in `test_convergence.py`.
"""
import numpy as np
import pytest

from fem.algebra.integrators import ThetaMethod
from fem.analysis.design import SIMPModel
from fem.analysis.sensitivity import (
    Compliance, MeanStress, ModulusParameterization, SensitivityAnalysis, SoftMaxStress,
)
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.equations import FiniteStrainElastic, Heat, LinearElastic, Poisson
from fem.physics.forms import (
    LinearElasticForm, ThermalStrain, tensor_to_voigt, voigt_to_tensor,
)
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.problem import LinearProblem
from fem.regions import TimeDependent, on_plane
from fem.space import FunctionSpace
from helpers import close, pinned, rollers, solved

E, NU = 200.0, 0.3
ALPHA, DT = 1e-3, 50.0
MU, LAMB = Enu_to_Lame(E, NU)
BETA = (3 * LAMB + 2 * MU) * ALPHA


def thermal(temperature=DT, alpha=ALPHA):
    return ThermalStrain(alpha, temperature)


def heated(mesh, bc, temperature=DT, alpha=ALPHA, **kwargs):
    """The thermoelastic problem on `mesh` under `bc` and its solution."""
    return solved(LinearElastic(E, NU, thermal=thermal(temperature, alpha)), mesh, bc, **kwargs)


def unit_box(dim, n):
    return box_mesh(corners=[[0.0] * dim, [1.0] * dim], resolution=(n,) * dim)


# -- the material's reduction -----------------------------------------------------


def test_constrained_stress_of_an_isotropic_strain_is_beta_on_the_diagonal():
    """`C_3D : (alpha dT I)` is `beta dT I` with the 3D beta, whatever the mesh."""
    eigenstrain = ALPHA * DT * np.broadcast_to(np.eye(3), (4, 2, 3, 3))
    sigma = LinearElasticMaterial(E, NU).constrained_stress(eigenstrain)
    assert sigma.shape == (4, 2, 3, 3)
    close(sigma, BETA * DT * np.eye(3), rtol=1e-12)


def test_constrained_stress_of_a_deviatoric_strain_has_no_lambda_part():
    """A traceless eigenstrain (a plastic strain) gives `2 mu eps*`: the lambda term
    multiplies the trace, so it needs all three diagonal entries to vanish."""
    eps = np.array([[[[0.01, 0.002, 0.0], [0.002, -0.004, 0.0], [0.0, 0.0, -0.006]]]])
    sigma = LinearElasticMaterial(E, NU).constrained_stress(eps)
    close(sigma, 2 * MU * eps, rtol=1e-12)


def test_constrained_stress_broadcasts_a_per_element_modulus():
    moduli = np.array([100.0, 200.0, 400.0])
    eigenstrain = ALPHA * DT * np.broadcast_to(np.eye(3), (3, 2, 3, 3))
    sigma = LinearElasticMaterial(moduli, NU).constrained_stress(eigenstrain)
    mu, lamb = Enu_to_Lame(moduli, NU)
    expected = ((3 * lamb + 2 * mu) * ALPHA * DT)[:, None, None, None] * np.eye(3)
    close(sigma, expected, rtol=1e-12)


def test_constrained_stress_takes_a_full_3x3_tensor_only():
    with pytest.raises(ValueError, match='3, 3'):
        LinearElasticMaterial(E, NU).constrained_stress(np.zeros((2, 1, 2, 2)))
    with pytest.raises(ValueError, match='per-element modulus'):
        LinearElasticMaterial(np.ones(3), NU).constrained_stress(np.zeros((2, 1, 3, 3)))


def test_tensor_to_voigt_inverts_voigt_to_tensor():
    rng = np.random.default_rng(1)
    for n_strains in (3, 6):
        voigt = rng.normal(size=(5, n_strains))
        tensor = voigt_to_tensor(voigt)                      # (5, d, d) stress packing
        d = tensor.shape[-1]
        full = np.zeros((5, 3, 3))
        full[:, :d, :d] = tensor
        close(tensor_to_voigt(full, d), voigt)


# -- the closed forms, exact for P1 ------------------------------------------------


def test_a_clamped_square_under_uniform_heating_carries_minus_beta_dT():
    """Every strain zero, so `sigma = -beta dT I` in all three normal components; the
    in-plane ones with the 3D beta, which is the plane-strain trap's whole content."""
    _, solution = heated(unit_box(2, 6), pinned(2))
    close(solution.dofs, 0.0, atol=1e-12)
    close(solution.strain, 0.0, atol=1e-13)
    close(solution.stress, -BETA * DT * np.eye(3), rtol=1e-11)


def test_a_clamped_cube_agrees_with_the_clamped_square():
    """A fully clamped state has no dimension in it: the 3D solve must give the same
    stress as the 2D one, including the out-of-plane component."""
    _, square = heated(unit_box(2, 4), pinned(2))
    _, cube = heated(unit_box(3, 3), pinned(3))
    close(cube.dofs, 0.0, atol=1e-12)
    close(cube.stress, -BETA * DT * np.eye(3), rtol=1e-11)
    close(cube.stress.mean(axis=0), square.stress.mean(axis=0), rtol=1e-11)


def test_free_expansion_in_plane_strain():
    """Free in the plane, held in z: in-plane stress zero, the in-plane strain
    `(1 + nu) alpha dT` (larger than the 3D `alpha dT`, the blocked z growth pushed
    sideways by Poisson's effect), and `sigma_zz = -E alpha dT` holding the plate flat."""
    _, solution = heated(unit_box(2, 6), rollers(2))
    scale = BETA * DT
    close(solution.stress[:, :2, :2], 0.0, atol=1e-11 * scale)
    close(solution.strain[:, :2, :2],
                               (1 + NU) * ALPHA * DT * np.eye(2), rtol=1e-10)
    close(solution.stress[:, 2, 2], -E * ALPHA * DT, rtol=1e-10)


def test_free_expansion_in_3d_is_stress_free():
    _, solution = heated(unit_box(3, 3), rollers(3))
    close(solution.stress, 0.0, atol=1e-11 * BETA * DT)
    close(solution.strain, ALPHA * DT * np.eye(3), rtol=1e-10)


def test_a_bar_between_walls():
    """Held in x (the walls) and z (plane strain), free to grow in y: the textbook
    `sigma_xx = -E alpha dT / (1 - nu)`, `sigma_yy = 0`."""
    walls = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, None]),
                       Dirichlet(on_plane(0, 1.0), [0.0, None]),
                       Dirichlet(on_plane(1, 0.0), [None, 0.0]))
    _, solution = heated(unit_box(2, 6), walls)
    close(solution.stress[:, 0, 0], -E * ALPHA * DT / (1 - NU), rtol=1e-10)
    close(solution.stress[:, 1, 1], 0.0, atol=1e-11 * BETA * DT)


def test_the_closed_forms_hold_on_p2_too():
    _, solution = heated(unit_box(2, 4), pinned(2), element_type=QuadraticTriangleElement)
    close(solution.stress, -BETA * DT * np.eye(3), rtol=1e-11)


def test_compliance_is_twice_the_elastic_energy():
    """Clamped: the mechanical strain is `-alpha dT I` in all three directions against
    `sigma = -beta dT I`, so `sigma : eps_el = 3 beta alpha dT^2` per unit area. The z
    term is a third of it, which `sigma : eps_total` (zero here) would miss entirely."""
    _, solution = heated(unit_box(2, 5), pinned(2))
    close(solution.compliance.sum(), 3 * BETA * ALPHA * DT**2, rtol=1e-10)


def test_per_element_alpha_is_read_per_element():
    """A two-material body (a bimetallic strip's layers): each element carries its own
    thermal stress modulus."""
    # Two triangles and no interior node, so the clamped state stays at rest under a
    # load that is not self-balanced across the diagonal.
    mesh = unit_box(2, 2)
    alpha = np.where(mesh.centroids[:, 1] < 0.5, ALPHA, 3 * ALPHA)
    _, solution = heated(mesh, pinned(2), alpha=alpha)
    beta = (3 * LAMB + 2 * MU) * alpha
    close(solution.stress[:, 0, 0], -beta * DT, rtol=1e-11)


# -- degeneracy and the temperature's forms ---------------------------------------


@pytest.mark.parametrize('kind', ['alpha', 'dT'])
def test_no_thermal_strain_reproduces_linear_elasticity(make_unit_square, kind):
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]), Source([0.0, -1.0]))
    plain = LinearElastic(E, NU).problem(mesh, bc)
    zero = (heated(mesh, bc, alpha=0.0) if kind == 'alpha' else heated(mesh, bc, temperature=0.0))
    problem, solution = zero
    close(problem.load, plain.load, atol=1e-14)
    reference = plain.solve()
    close(solution.dofs, reference.dofs, atol=1e-14)
    close(solution.stress, reference.stress, atol=1e-14)
    close(solution.compliance, reference.compliance, atol=1e-14)


def test_a_heat_solution_drives_the_same_load_as_its_closed_form(make_unit_square):
    """The coupling path: a Poisson solve for T on the same mesh, handed over as its
    `NodalField`. The solve is exact for a linear T, so the two loads agree to round-off."""
    mesh = make_unit_square(7)
    hot, cold = 100.0, 0.0
    field = Poisson().problem(mesh, Conditions(
        Dirichlet(on_plane(0, 0.0), hot), Dirichlet(on_plane(0, 1.0), cold))).solve()

    coupled, _ = heated(mesh, pinned(2), temperature=field)
    closed, _ = heated(mesh, pinned(2), temperature=lambda p: hot + (cold - hot) * p[:, 0])
    assert coupled.operator_load is not None
    close(coupled.operator_load, closed.operator_load, atol=1e-10)


def test_a_bare_nodal_array_is_refused(make_unit_square):
    """Nodal values without their space cannot be read at a quadrature point."""
    mesh = make_unit_square(5)
    with pytest.raises(TypeError, match='NodalField'):
        heated(mesh, pinned(2), temperature=1.0 + mesh.vertices[:, 0])


def test_a_nodal_temperature_must_share_the_elastic_discretization(make_unit_square):
    mesh = make_unit_square(4)
    p1_temperature = FunctionSpace(mesh, n_components=1).interpolate(1.0)
    with pytest.raises(ValueError, match='same mesh and element type'):
        heated(mesh, pinned(2), temperature=p1_temperature, element_type=QuadraticTriangleElement)
    vector = FunctionSpace(mesh, n_components=2).interpolate([1.0, 0.0])
    with pytest.raises(ValueError, match='scalar'):
        heated(mesh, pinned(2), temperature=vector)


def test_a_time_dependent_temperature_is_refused(make_unit_square):
    with pytest.raises(TypeError, match='one instant'):
        ThermalStrain(ALPHA, TimeDependent(lambda p, t: t))
    mesh = make_unit_square(3)
    history = ThetaMethod(dt=0.1, steps=2).solve(Heat().problem(mesh, pinned() + Source(1.0)))
    with pytest.raises(TypeError, match=r'solution\[i\]'):
        ThermalStrain(ALPHA, history)
    heated(mesh, pinned(2), temperature=history[1])


def test_a_temperature_of_another_kind_is_refused():
    with pytest.raises(TypeError, match='NodalField'):
        ThermalStrain(ALPHA, [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match='got str'):
        ThermalStrain(ALPHA, 'hot')


def test_per_element_alpha_length_is_checked(make_unit_square):
    with pytest.raises(ValueError, match='per-element alpha'):
        heated(make_unit_square(4), pinned(2), alpha=np.ones(3))


# -- the problem's bookkeeping -----------------------------------------------------


def test_residual_is_the_gradient_of_the_energy(make_unit_square):
    """The thermal load enters the energy as `−f_thᵀ u`, so the residual must still be
    the gradient of `½ uᵀ K u − (f + f_th)ᵀ u`."""
    mesh = make_unit_square(5)
    problem, _ = heated(mesh, pinned(2), temperature=lambda p: p[:, 0] * p[:, 1])
    rng = np.random.default_rng(0)
    u = rng.normal(size=problem.space.n_dofs)
    residual = problem.residual(u)
    h = 1e-6
    for i in rng.choice(problem.space.n_dofs, 8, replace=False):
        e = np.zeros_like(u)
        e[i] = h
        numeric = (problem.energy(u + e) - problem.energy(u - e)) / (2 * h)
        close(residual[i], numeric, rtol=1e-6, atol=1e-8)


def test_with_operator_carries_the_new_operators_load(make_unit_square):
    """The operator's own load follows the operator: a stiffer restatement of the same
    thermal strain doubles it, and a restatement without one drops it."""
    mesh = make_unit_square(5)
    space = FunctionSpace(mesh, n_components=2)
    strain = thermal()
    problem = LinearProblem(space, LinearElasticForm(LinearElasticMaterial(E, NU), strain),
                            pinned(2) + Source([0.0, -1.0]))
    assert problem.operator_load is not None

    stiffer = problem.with_operator(LinearElasticForm(LinearElasticMaterial(2 * E, NU), strain))
    close(stiffer.operator_load, 2 * problem.operator_load, rtol=1e-12)
    close(stiffer.load - stiffer.operator_load,
                               problem.load - problem.operator_load, atol=1e-14)

    plain = problem.with_operator(LinearElasticForm(LinearElasticMaterial(E, NU)))
    assert plain.operator_load is None
    close(plain.load, problem.load - problem.operator_load, atol=1e-14)


def test_with_operator_keeps_the_conditions_load(make_unit_square):
    """Without an operator load the restated problem's load is the very vector the
    original assembled: nothing is reassembled per restatement."""
    problem = LinearElastic(E, NU).problem(
        make_unit_square(4), pinned(2) + Source([0.0, -1.0]))
    stiffer = problem.with_operator(LinearElastic(2 * E, NU).operator(problem.space))
    assert stiffer.load is problem.load


def test_a_scaled_form_scales_its_load(make_unit_square):
    space = FunctionSpace(make_unit_square(4), n_components=2)
    form = LinearElasticForm(LinearElasticMaterial(E, NU), thermal())
    close(space.assemble_loads(3.0 * form), 3.0 * space.assemble_loads(form))
    assert space.assemble_loads(LinearElasticForm(LinearElasticMaterial(E, NU))) is None


def test_simp_refuses_a_thermal_template(make_unit_square):
    problem, _ = heated(make_unit_square(4), pinned(2))
    with pytest.raises(NotImplementedError, match='eigenstrain'):
        SIMPModel(problem)


def test_energy_density_refuses_a_thermal_strain():
    with pytest.raises(NotImplementedError, match='thermal'):
        LinearElastic(E, NU, thermal=thermal()).energy_density()
    with pytest.raises(NotImplementedError, match='LinearElastic'):
        FiniteStrainElastic(E, NU, thermal=thermal())


# -- the estimator's interior term -------------------------------------------------


def test_stress_divergence_carries_the_constrained_stress_gradient(make_unit_square):
    """At rest under a linear T, `div sigma = -beta grad(dT)`: a nonzero interior
    residual on P1, where the Navier part vanishes."""
    mesh = make_unit_square(5)
    problem, _ = heated(mesh, pinned(2), temperature=lambda p: 2.0 * p[:, 0] - 3.0 * p[:, 1])
    at_rest = problem.solution(np.zeros(problem.space.n_dofs))
    flux = problem.operator.flux()
    assert flux is not None
    divergence = flux.divergence(at_rest)
    close(divergence, -BETA * np.array([2.0, -3.0]), rtol=1e-10)


def test_per_element_modulus_on_p2_reports_each_elements_stress():
    """A P2 body with two moduli held at rest: the sample runs at every point of a
    three-point rule with each element's own `beta`, and so does the plain one."""
    mesh = unit_box(2, 2)
    moduli = np.where(mesh.centroids[:, 1] < 0.5, E, 2 * E)
    problem = LinearElastic(moduli, NU, thermal=thermal()).problem(
        mesh, pinned(2), element_type=QuadraticTriangleElement)
    at_rest = problem.solution(np.zeros(problem.space.n_dofs))
    mu, lamb = Enu_to_Lame(moduli, NU)
    close(at_rest.stress[:, 0, 0], -(3 * lamb + 2 * mu) * ALPHA * DT, rtol=1e-11)
    plain = LinearElastic(moduli, NU).problem(
        mesh, pinned(2) + Source([0.0, -1.0]), element_type=QuadraticTriangleElement).solve()
    close(plain.stress[:, 2, 2], NU * (plain.stress[:, 0, 0] + plain.stress[:, 1, 1]), rtol=1e-11)


# -- the load's rule ---------------------------------------------------------------


def test_the_load_has_its_own_rule(make_unit_square):
    """The load pairs a shape gradient with the temperature: degree 0 for a constant on
    P1, `2p - 1` for a solution on degree-`p` elements, and the stiffness rule is
    untouched either way."""
    mesh = make_unit_square(3)
    constant = LinearElasticForm(LinearElasticMaterial(E, NU), thermal())
    assert constant.load_quadrature_degree(1) == 0
    assert constant.quadrature_degree(1) == 0
    assert LinearElasticForm(LinearElasticMaterial(E, NU)).load_quadrature_degree(2) == 0
    p2 = FunctionSpace(mesh, element_type=QuadraticTriangleElement).interpolate(lambda p: p[:, 0])
    on_p2 = LinearElasticForm(LinearElasticMaterial(E, NU), thermal(p2))
    assert on_p2.load_quadrature_degree(2) == 3
    assert on_p2.quadrature_degree(2) == 0
    function = LinearElasticForm(LinearElasticMaterial(E, NU), thermal(lambda p: p[:, 0]))
    assert function.load_quadrature_degree(1) == 2


def test_a_p2_temperature_is_integrated_exactly(make_unit_square):
    """On P2 the load is a cubic, which the stiffness's degree-2 rule under-integrates;
    the load's own rule matches a rule of higher degree to round-off."""
    mesh = make_unit_square(3)
    space = FunctionSpace(mesh, n_components=2, element_type=QuadraticTriangleElement)
    temperature = Poisson().problem(mesh, pinned() + Source(1.0),
                                    element_type=QuadraticTriangleElement).solve()
    form = LinearElasticForm(LinearElasticMaterial(E, NU), thermal(temperature))
    exact = space._term_vector_scatter(False).scatter(form.element_loads(space.geometry_at(4)))
    under = space._term_vector_scatter(False).scatter(form.element_loads(space.geometry_at(2)))
    load = space.assemble_loads(form)
    assert load is not None
    close(load, exact, rtol=1e-12)
    assert not np.allclose(under, exact, rtol=1e-3)


# -- the analyses that take no eigenstrain yet -------------------------------------


def test_the_stress_quantities_of_interest_refuse_a_thermal_problem(make_unit_square):
    """They measure the stress of the displacement alone, which is not the thermal
    stress; under plane strain the two happen to share a von Mises, but the measure
    is refused rather than relied on."""
    problem, solution = heated(make_unit_square(4), pinned(2))
    for qoi in (MeanStress(problem.space, problem.physics.material),
                SoftMaxStress(problem.space, problem.physics.material)):
        with pytest.raises(NotImplementedError, match='eigenstrain'):
            qoi.value(problem, solution.dofs)
        with pytest.raises(NotImplementedError, match='eigenstrain'):
            qoi.dJ_du(problem, solution.dofs)


def test_the_adjoint_gradient_refuses_a_thermal_problem(make_unit_square):
    """The thermal load scales with the modulus, which the parameterizations take as
    fixed: their gradient would be wrong, so it is refused."""
    mesh = make_unit_square(3)
    space = FunctionSpace(mesh, n_components=2)
    moduli = np.linspace(E, 2 * E, len(space.element_nodes))
    problem = LinearElastic(moduli, NU, thermal=thermal()).problem(
        mesh, Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]), Source([0.0, -1.0])))
    analysis = SensitivityAnalysis(problem)
    u = analysis.solve_forward()
    with pytest.raises(NotImplementedError, match='eigenstrain'):
        analysis.gradient(Compliance(), ModulusParameterization.create(space, LinearElasticMaterial(moduli, NU)), u)
