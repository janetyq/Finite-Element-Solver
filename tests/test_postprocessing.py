"""Derived fields: the quantities recovered from a solved displacement.

The anchor is the compliance identity: element compliance is the volume integral of
sigma:epsilon and K = int B^T D B, so the per-element compliances sum to u^T K u
exactly, tying the recovered fields to the operator without a magic number.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.elements import LinearTriangleElement
from fem.energies import SmallStrain, StVenantKirchhoff
from fem.equations import LinearElastic
from fem.forms import EnergyForm, LinearElasticForm
from fem.materials import Enu_to_Lame, LinearElasticMaterial
from fem.regions import on_plane
from fem.solver import Solver

E, NU = 210.0, 0.3
REFERENCE_TRIANGLE = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])


def _geometry_and_nodal(A):
    """A single triangle carrying the exact linear field u(x) = A x, whose analytic strain
    and stress are an independent reference."""
    geometry = LinearTriangleElement.geometry(REFERENCE_TRIANGLE)
    nodal = (A @ REFERENCE_TRIANGLE[0].T).T
    return geometry, nodal[None]


def _cantilever_bc() -> BoundaryConditions:
    """Left edge pinned, right edge pulled down."""
    bc = BoundaryConditions()
    bc = bc + Dirichlet(on_plane(0, 0.0), [0.0, 0.0])
    bc = bc + Neumann(on_plane(0, 1.0), [0.0, -1.0])
    return bc


def _solved(mesh):
    solver = Solver(mesh, LinearElastic(E=1.0, nu=0.3), _cantilever_bc())
    return solver, solver.solve()


def test_element_compliance_sums_to_the_strain_energy_form(make_unit_square):
    """sum_e int sigma:epsilon == u^T K u. A dropped factor of two on the shear terms or a
    mismatched Voigt ordering breaks the identity."""
    mesh = make_unit_square(8)
    solver, solution = _solved(mesh)

    K = solver.problem().tangent(None)
    np.testing.assert_allclose(
        solution.compliance.sum(), solution.u @ (K @ solution.u), rtol=1e-10
    )


def test_compliance_is_positive_and_finite(make_unit_square):
    """Strain energy density is non-negative for a positive-definite D, so no
    element may report negative compliance."""
    _, solution = _solved(make_unit_square(8))
    assert np.all(np.isfinite(solution.compliance))
    assert np.all(solution.compliance >= 0.0)


def test_solver_and_design_model_recover_the_same_fields(make_unit_square):
    """`Solver` and `SIMPModel` recover the same fields: at a uniform unit density the
    diluted problem is the plain elastic solve."""
    from fem.design import SIMPModel
    from fem.solve import LinearSolve

    mesh = make_unit_square(6)
    _, solution = _solved(mesh)

    equation = LinearElastic(E=1.0, nu=0.3)
    model = SIMPModel(equation.problem(mesh, _cantilever_bc()), penalty=1.0)
    rho = np.ones(len(mesh.elements))
    design_solution = model.solution(rho, LinearSolve().solve(model.problem(rho)))

    np.testing.assert_allclose(design_solution.u, solution.u, rtol=1e-10)
    np.testing.assert_allclose(design_solution.compliance, solution.compliance, rtol=1e-10)


# -- what the recovered tensors actually are ---------------------------------


def test_recovered_strain_is_the_analytic_symmetric_gradient():
    """For u(x) = A x the strain is the symmetric part of A, which ties B, the Voigt
    unpacking, and the engineering-shear factor together."""
    A = np.array([[0.03, 0.08], [-0.01, -0.02]])
    geometry, u_elements = _geometry_and_nodal(A)

    fields = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
        geometry, u_elements
    )

    np.testing.assert_allclose(fields.strain[0][:2, :2], 0.5 * (A + A.T), atol=1e-12)
    # Plane strain: the restrained direction carries no strain, by definition.
    np.testing.assert_allclose(fields.strain[0][2, 2], 0.0, atol=1e-12)


def test_recovered_stress_satisfies_the_isotropic_law():
    """sigma = 2*mu*eps + lambda*tr(eps)*I on the full 3x3 tensor, out-of-plane included."""
    A = np.array([[0.02, 0.05], [0.01, -0.03]])
    geometry, u_elements = _geometry_and_nodal(A)
    mu, lamb = Enu_to_Lame(E, NU)

    fields = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
        geometry, u_elements
    )
    strain, stress = fields.strain[0], fields.stress[0]

    expected = 2 * mu * strain + lamb * np.trace(strain) * np.eye(3)
    np.testing.assert_allclose(stress, expected, atol=1e-10)


def test_energy_and_linear_paths_report_the_same_stress_at_small_strain():
    """The energy path's Cauchy stress and the linear path's stress agree to O(||grad u||),
    the pushforward J^-1 P F^T being what separates them. Asserted as a rate (halving the
    displacement halves the gap), which also pins the transposes."""
    shape = np.array([[1.0, 3.0], [0.5, -2.0]])

    def discrepancy(amplitude):
        geometry, u_elements = _geometry_and_nodal(amplitude * shape)
        linear = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
            geometry, u_elements
        )
        energy = EnergyForm(SmallStrain(E, NU)).derived_fields(geometry, u_elements)
        # Strain is the same measure in both, so it must match to machine precision;
        # only the stress carries the pushforward difference.
        np.testing.assert_allclose(energy.strain, linear.strain, atol=1e-14)
        return np.abs(energy.stress - linear.stress).max() / np.abs(linear.stress).max()

    # Asserted as a *rate*, not a tolerance: halving the displacement must halve
    # the relative gap. A transposed pushforward would leave an O(1) discrepancy
    # that no amount of shrinking removes, so this pins the orientation rather
    # than just passing at one amplitude.
    coarse, fine = discrepancy(1e-5), discrepancy(5e-6)
    assert coarse < 1e-3
    assert fine == pytest.approx(coarse / 2, rel=0.05)


def test_out_of_plane_stress_agrees_across_both_elastic_paths():
    """The reconstructed sigma_zz is the same number from the material (nu*(sxx + syy)) and
    from the density (lambda*tr(S))."""
    A = 1e-6 * np.array([[2.0, 1.0], [-1.0, 4.0]])
    geometry, u_elements = _geometry_and_nodal(A)

    linear = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
        geometry, u_elements
    )
    energy = EnergyForm(SmallStrain(E, NU)).derived_fields(geometry, u_elements)

    assert linear.stress[0][2, 2] != pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(energy.stress[0][2, 2], linear.stress[0][2, 2], rtol=1e-5)


def test_energy_path_compliance_is_the_work_conjugate_pairing():
    """Compliance pairs work-conjugate measures (second Piola-Kirchhoff with Green-Lagrange).
    Contracting the reported Cauchy stress with Green-Lagrange strain is wrong by tens
    of percent at finite strain, so this checks the finite regime against S:E."""
    A = np.array([[0.15, 0.30], [0.05, -0.10]])   # finite strain
    geometry, u_elements = _geometry_and_nodal(A)

    fields = EnergyForm(StVenantKirchhoff(E, NU)).derived_fields(geometry, u_elements)

    # Built from scratch here rather than from the density's derivative chain, so
    # the reference is independent of the code under test: Green-Lagrange strain
    # from the deformation gradient, the St-VK stress from the isotropic law, and
    # their contraction over the reference volume.
    mu, lamb = Enu_to_Lame(E, NU)
    F = np.eye(2) + A.T                       # A is exact for a P1 linear field
    green_lagrange = 0.5 * (F.T @ F - np.eye(2))
    pk2 = lamb * np.trace(green_lagrange) * np.eye(2) + 2 * mu * green_lagrange
    conjugate = (pk2 * green_lagrange).sum() * geometry.volumes[0]

    np.testing.assert_allclose(fields.compliance[0], conjugate, rtol=1e-12)

    # And the mismatched pairing really differs here, so the test has teeth.
    mixed = np.einsum('eij,eij,e->e', fields.stress, fields.strain, geometry.volumes)
    assert abs(mixed[0] - conjugate) / abs(conjugate) > 0.1


def test_compliance_agrees_across_both_paths_at_small_strain():
    """Where the models coincide, so must the energy each reports."""
    A = 1e-6 * np.array([[1.0, 3.0], [0.5, -2.0]])
    geometry, u_elements = _geometry_and_nodal(A)

    linear = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
        geometry, u_elements
    )
    energy = EnergyForm(SmallStrain(E, NU)).derived_fields(geometry, u_elements)

    np.testing.assert_allclose(energy.compliance, linear.compliance, rtol=1e-5)


def test_green_lagrange_strain_vanishes_under_rigid_rotation():
    """A rigid rotation stores no energy, so the Green-Lagrange strain is zero where
    small strain reports a spurious compression."""
    theta = 0.4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = R - np.eye(2)  # u(x) = (R - I)x rotates the element rigidly
    geometry, u_elements = _geometry_and_nodal(A)

    exact = EnergyForm(StVenantKirchhoff(E, NU)).derived_fields(geometry, u_elements)
    np.testing.assert_allclose(exact.strain, 0.0, atol=1e-12)

    linearised = EnergyForm(SmallStrain(E, NU)).derived_fields(geometry, u_elements)
    assert np.abs(linearised.strain).max() > 1e-3


@pytest.mark.parametrize('n', [4, 6])
def test_compliance_is_mesh_convergent(n, make_unit_square):
    """Total compliance is a physical quantity, so refining the mesh must not move it wildly."""
    _, coarse = _solved(make_unit_square(n))
    _, fine = _solved(make_unit_square(n + 4))
    assert coarse.compliance.sum() == pytest.approx(fine.compliance.sum(), rel=0.5)
