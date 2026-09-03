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

from fem.analysis.design import SIMPModel
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.equations import FiniteStrainElastic, LinearElastic, Poisson
from fem.physics.forms import (
    LinearElasticForm, ThermalStrain, tensor_to_voigt, voigt_to_tensor,
)
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.problem import LinearProblem
from fem.regions import TimeDependent, everywhere, on_plane
from fem.space import FunctionSpace

E, NU = 200.0, 0.3
ALPHA, DT = 1e-3, 50.0
MU, LAMB = Enu_to_Lame(E, NU)
BETA = (3 * LAMB + 2 * MU) * ALPHA


def close(actual, expected, **tolerances):
    """`assert_allclose` with `expected` broadcast to `actual`'s shape: a closed form is
    one tensor, the solve reports one per element."""
    actual, expected = np.asarray(actual), np.asarray(expected, dtype=float)
    # Round-off against a closed form: an entry that should be zero comes out at
    # machine precision times the largest entry, which a relative tolerance rejects.
    tolerances.setdefault('atol', 1e-12 * max(1.0, float(np.abs(expected).max())))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), **tolerances)


def thermal(temperature=DT, alpha=ALPHA):
    return ThermalStrain(alpha, temperature)


def heated(mesh, bc, temperature=DT, alpha=ALPHA, **kwargs):
    """The thermoelastic problem on `mesh` under `bc` and its solution."""
    problem = LinearElastic(E, NU, thermal=thermal(temperature, alpha)).problem(mesh, bc, **kwargs)
    return problem, problem.solve()


def clamped(dim):
    return Conditions(Dirichlet(everywhere(), [0.0] * dim))


def rollers(dim):
    """Each coordinate plane through the origin a roller: rigid modes removed, every
    face free to move away from it."""
    return Conditions(*[
        Dirichlet(on_plane(axis, 0.0), [0.0 if c == axis else None for c in range(dim)])
        for axis in range(dim)
    ])


def unit_box(dim, n):
    return box_mesh(corners=[[0.0] * dim, [1.0] * dim], resolution=(n,) * dim)


# -- the material's reduction -----------------------------------------------------


def test_eigenstress_of_an_isotropic_strain_is_beta_on_the_diagonal():
    """`C_3D : (alpha dT I)` is `beta dT I` with the 3D beta, whatever the mesh."""
    eigenstrain = ALPHA * DT * np.broadcast_to(np.eye(3), (4, 2, 3, 3))
    sigma = LinearElasticMaterial(E, NU).eigenstress(eigenstrain)
    assert sigma.shape == (4, 2, 3, 3)
    close(sigma, BETA * DT * np.eye(3), rtol=1e-12)


def test_eigenstress_of_a_deviatoric_strain_has_no_lambda_part():
    """A traceless eigenstrain (a plastic strain) gives `2 mu eps*`: the lambda term
    multiplies the trace, so it needs all three diagonal entries to vanish."""
    eps = np.array([[[[0.01, 0.002, 0.0], [0.002, -0.004, 0.0], [0.0, 0.0, -0.006]]]])
    sigma = LinearElasticMaterial(E, NU).eigenstress(eps)
    close(sigma, 2 * MU * eps, rtol=1e-12)


def test_eigenstress_broadcasts_a_per_element_modulus():
    moduli = np.array([100.0, 200.0, 400.0])
    eigenstrain = ALPHA * DT * np.broadcast_to(np.eye(3), (3, 2, 3, 3))
    sigma = LinearElasticMaterial(moduli, NU).eigenstress(eigenstrain)
    mu, lamb = Enu_to_Lame(moduli, NU)
    expected = ((3 * lamb + 2 * mu) * ALPHA * DT)[:, None, None, None] * np.eye(3)
    close(sigma, expected, rtol=1e-12)


def test_eigenstress_takes_a_full_3x3_tensor_only():
    with pytest.raises(ValueError, match='3, 3'):
        LinearElasticMaterial(E, NU).eigenstress(np.zeros((2, 1, 2, 2)))
    with pytest.raises(ValueError, match='per-element modulus'):
        LinearElasticMaterial(np.ones(3), NU).eigenstress(np.zeros((2, 1, 3, 3)))


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
    _, solution = heated(unit_box(2, 6), clamped(2))
    close(solution.dofs, 0.0, atol=1e-12)
    close(solution.strain, 0.0, atol=1e-13)
    close(solution.stress, -BETA * DT * np.eye(3), rtol=1e-11)


def test_a_clamped_cube_agrees_with_the_clamped_square():
    """A fully clamped state has no dimension in it: the 3D solve must give the same
    stress as the 2D one, including the out-of-plane component."""
    _, square = heated(unit_box(2, 4), clamped(2))
    _, cube = heated(unit_box(3, 3), clamped(3))
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
    _, solution = heated(unit_box(2, 4), clamped(2), element_type=QuadraticTriangleElement)
    close(solution.stress, -BETA * DT * np.eye(3), rtol=1e-11)


def test_compliance_is_twice_the_elastic_energy():
    """Clamped: the mechanical strain is `-alpha dT I` in all three directions against
    `sigma = -beta dT I`, so `sigma : eps_el = 3 beta alpha dT^2` per unit area. The z
    term is a third of it, which `sigma : eps_total` (zero here) would miss entirely."""
    _, solution = heated(unit_box(2, 5), clamped(2))
    close(solution.compliance.sum(), 3 * BETA * ALPHA * DT**2, rtol=1e-10)


def test_per_element_alpha_is_read_per_element():
    """A two-material body (a bimetallic strip's layers): each element carries its own
    thermal stress modulus."""
    # Two triangles and no interior node, so the clamped state stays at rest under a
    # load that is not self-balanced across the diagonal.
    mesh = unit_box(2, 2)
    alpha = np.where(mesh.centroids[:, 1] < 0.5, ALPHA, 3 * ALPHA)
    _, solution = heated(mesh, clamped(2), alpha=alpha)
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

    coupled, _ = heated(mesh, clamped(2), temperature=field)
    closed, _ = heated(mesh, clamped(2), temperature=lambda p: hot + (cold - hot) * p[:, 0])
    assert coupled.operator_load is not None
    close(coupled.operator_load, closed.operator_load, atol=1e-10)


def test_a_bare_nodal_array_is_refused(make_unit_square):
    """Nodal values without their space cannot be read at a quadrature point."""
    mesh = make_unit_square(5)
    with pytest.raises(TypeError, match='NodalField'):
        heated(mesh, clamped(2), temperature=1.0 + mesh.vertices[:, 0])


def test_a_nodal_temperature_must_share_the_elastic_discretization(make_unit_square):
    mesh = make_unit_square(4)
    p1_temperature = FunctionSpace(mesh, n_components=1).interpolate(1.0)
    with pytest.raises(ValueError, match='same mesh and element type'):
        heated(mesh, clamped(2), temperature=p1_temperature, element_type=QuadraticTriangleElement)
    vector = FunctionSpace(mesh, n_components=2).interpolate([1.0, 0.0])
    with pytest.raises(ValueError, match='scalar'):
        heated(mesh, clamped(2), temperature=vector)


def test_a_time_dependent_temperature_is_refused():
    with pytest.raises(TypeError, match='one instant'):
        ThermalStrain(ALPHA, TimeDependent(lambda p, t: t))


def test_per_element_alpha_length_is_checked(make_unit_square):
    with pytest.raises(ValueError, match='per-element alpha'):
        heated(make_unit_square(4), clamped(2), alpha=np.ones(3))


# -- the problem's bookkeeping -----------------------------------------------------


def test_residual_is_the_gradient_of_the_energy(make_unit_square):
    """The thermal load enters the energy as `−f_thᵀ u`, so the residual must still be
    the gradient of `½ uᵀ K u − (f + f_th)ᵀ u`."""
    mesh = make_unit_square(5)
    problem, _ = heated(mesh, clamped(2), temperature=lambda p: p[:, 0] * p[:, 1])
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
                            clamped(2) + Source([0.0, -1.0]))
    assert problem.operator_load is not None

    stiffer = problem.with_operator(LinearElasticForm(LinearElasticMaterial(2 * E, NU), strain))
    close(stiffer.operator_load, 2 * problem.operator_load, rtol=1e-12)
    close(stiffer.load - stiffer.operator_load,
                               problem.load - problem.operator_load, atol=1e-14)

    plain = problem.with_operator(LinearElasticForm(LinearElasticMaterial(E, NU)))
    assert plain.operator_load is None
    close(plain.load, problem.load - problem.operator_load, atol=1e-14)


def test_a_scaled_form_scales_its_load(make_unit_square):
    space = FunctionSpace(make_unit_square(4), n_components=2)
    form = LinearElasticForm(LinearElasticMaterial(E, NU), thermal())
    close(space.assemble_loads(3.0 * form), 3.0 * space.assemble_loads(form))
    assert space.assemble_loads(LinearElasticForm(LinearElasticMaterial(E, NU))) is None


def test_simp_refuses_a_thermal_template(make_unit_square):
    problem, _ = heated(make_unit_square(4), clamped(2))
    with pytest.raises(NotImplementedError, match='eigenstrain'):
        SIMPModel(problem)


def test_energy_density_refuses_a_thermal_strain():
    with pytest.raises(NotImplementedError, match='thermal'):
        LinearElastic(E, NU, thermal=thermal()).energy_density()
    with pytest.raises(NotImplementedError, match='LinearElastic'):
        FiniteStrainElastic(E, NU, thermal=thermal())


# -- the estimator's interior term -------------------------------------------------


def test_stress_divergence_carries_the_eigenstress_gradient(make_unit_square):
    """At rest under a linear T, `div sigma = -beta grad(dT)`: a nonzero interior
    residual on P1, where the Navier part vanishes."""
    mesh = make_unit_square(5)
    problem, _ = heated(mesh, clamped(2), temperature=lambda p: 2.0 * p[:, 0] - 3.0 * p[:, 1])
    at_rest = problem.solution(np.zeros(problem.space.n_dofs))
    flux = problem.operator.flux()
    assert flux is not None
    divergence = flux.divergence(at_rest)
    close(divergence, -BETA * np.array([2.0, -3.0]), rtol=1e-10)
