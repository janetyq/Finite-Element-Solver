"""Derived fields: the quantities recovered from a solved displacement.

Recovery is shared -- `Solver` and `TopologyOptimizer` both turn a solved `u`
into strain, stress, and compliance -- so these pin the contract both go through
rather than either one's implementation of it.

The compliance identity below is the anchor. Element compliance is the volume
integral of the double contraction sigma:epsilon, and the assembled stiffness is
K = int B^T D B, so summing the per-element compliance must reproduce u^T K u
exactly. That ties the recovered fields to the operator that produced them
without a single hard-coded magic number, which is what makes it survive a
change of representation.
"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.elements import LinearTriangleElement
from fem.energies import SmallStrain, StVenantKirchhoff
from fem.equations import LinearElastic
from fem.forms import EnergyForm, LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.regions import on_plane
from fem.solver import Solver
from fem.topology import TopologyOptimizer

E, NU = 210.0, 0.3
REFERENCE_TRIANGLE = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])


def _geometry_and_nodal(A):
    """A single triangle carrying the exact linear field u(x) = A x.

    P1 elements reproduce a linear field exactly, so the recovered gradient is A
    to machine precision -- which makes the analytic strain and stress available
    as an independent reference rather than a second implementation.
    """
    geometry = LinearTriangleElement.geometry(REFERENCE_TRIANGLE)
    nodal = (A @ REFERENCE_TRIANGLE[0].T).T
    return geometry, nodal[None]


def _cantilever_bc() -> BoundaryConditions:
    """Left edge pinned, right edge pulled down -- a loaded 2D cantilever."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])
    return bc


def _solved(mesh):
    solver = Solver(mesh, LinearElastic(E=1.0, nu=0.3), _cantilever_bc())
    return solver, solver.solve()


def test_element_compliance_sums_to_the_strain_energy_form(make_unit_square):
    """sum_e int sigma:epsilon == u^T K u.

    The per-element compliance is a contraction of the recovered stress against
    the recovered strain; K is assembled from the same B and D. If the two ever
    disagree -- a dropped factor of two on the shear terms, a mismatched Voigt
    ordering -- the identity breaks, which is exactly the class of error a
    representation change can introduce.
    """
    mesh = make_unit_square(8)
    solver, solution = _solved(mesh)

    K = solver._steady_problem().tangent(None)
    np.testing.assert_allclose(
        solution.compliance.sum(), solution.u @ (K @ solution.u), rtol=1e-10
    )


def test_compliance_is_positive_and_finite(make_unit_square):
    """Strain energy density is non-negative for a positive-definite D, so no
    element may report negative compliance."""
    _, solution = _solved(make_unit_square(8))
    assert np.all(np.isfinite(solution.compliance))
    assert np.all(solution.compliance >= 0.0)


def test_solver_and_optimizer_recover_the_same_fields(make_unit_square):
    """The two call sites that recover derived fields must agree.

    `Solver` and `TopologyOptimizer` run the same numerical path -- LinearProblem
    -> LinearSolve -> derived fields -- so at a uniform unit density the
    optimizer's first iterate is the plain elastic solve. They recover the fields
    through separate code today; this pins that they agree, so collapsing them
    onto one owner is a refactor rather than a change.
    """
    mesh = make_unit_square(6)
    _, solution = _solved(mesh)

    optimizer = TopologyOptimizer(
        mesh, LinearElastic(E=1.0, nu=0.3), _cantilever_bc(),
        iters=1, volume_frac=1.0, penalty=1.0,
    )
    # rho = 1 and p = 1 leaves E(rho) = E_0, so this is the same elastic problem.
    optimizer_solution = optimizer._solve()

    np.testing.assert_allclose(optimizer_solution.u, solution.u, rtol=1e-10)
    np.testing.assert_allclose(
        optimizer_solution.compliance, solution.compliance, rtol=1e-10
    )


# -- what the recovered tensors actually are ---------------------------------


def test_recovered_strain_is_the_analytic_symmetric_gradient():
    """Ties B, the Voigt unpacking, and the engineering-shear factor together.

    For u(x) = A x the strain is exactly the symmetric part of A. Getting this
    right requires the shear row of B, the Voigt ordering, and the division by
    two to all agree -- a factor-of-two error anywhere shows up here, where the
    old `norm`-of-a-Voigt-vector reduction could hide it.
    """
    A = np.array([[0.03, 0.08], [-0.01, -0.02]])
    geometry, u_elements = _geometry_and_nodal(A)

    fields = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
        geometry, u_elements
    )

    np.testing.assert_allclose(fields.strain[0][:2, :2], 0.5 * (A + A.T), atol=1e-12)
    # Plane strain: the restrained direction carries no strain, by definition.
    np.testing.assert_allclose(fields.strain[0][2, 2], 0.0, atol=1e-12)


def test_recovered_stress_satisfies_the_isotropic_law():
    """sigma = 2*mu*eps + lambda*tr(eps)*I, checked on the full 3x3 tensor
    including the reconstructed out-of-plane component."""
    from fem.materials import Enu_to_Lame

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
    """The two elastic paths must agree where they model the same physics.

    `SmallStrain` is the linearisation the direct assembly solves, so at small
    displacement the energy path's Cauchy stress and the linear path's stress
    describe one state. They agree to O(||grad u||) -- the pushforward J^-1 P F^T
    is what separates them -- so this checks the whole 3x3 tensor at a strain
    small enough for that difference to be negligible.

    This is also what pins the transposes. `ElementGeometry.gradients` returns the
    transpose of the usual displacement gradient, so dW_dF comes out transposed
    too; get the pushforward wrong and the off-diagonal terms disagree here.
    """
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
    """The reconstructed sigma_zz is the same number whichever path built it.

    The linear path gets it from the material as nu*(sxx + syy); the energy path
    gets it from the density as lambda*tr(S). Those are the same quantity written
    two ways, and if they ever drift the two solvers report different von Mises
    stresses for identical physics.
    """
    A = 1e-6 * np.array([[2.0, 1.0], [-1.0, 4.0]])
    geometry, u_elements = _geometry_and_nodal(A)

    linear = LinearElasticForm(LinearElasticMaterial(E, NU)).derived_fields(
        geometry, u_elements
    )
    energy = EnergyForm(SmallStrain(E, NU)).derived_fields(geometry, u_elements)

    assert linear.stress[0][2, 2] != pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(energy.stress[0][2, 2], linear.stress[0][2, 2], rtol=1e-5)


def test_green_lagrange_strain_vanishes_under_rigid_rotation():
    """The property that makes the finite-strain measure worth its cost: rotating
    a body rigidly stores no energy, so its strain must be exactly zero. Small
    strain does not have this -- it reports a spurious compression -- which is
    what the two measures differ on."""
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
    """Total compliance is a physical quantity, so refining the mesh must not
    move it wildly -- a coarse guard against a recovery that scales with element
    count rather than with volume."""
    _, coarse = _solved(make_unit_square(n))
    _, fine = _solved(make_unit_square(n + 4))
    assert coarse.compliance.sum() == pytest.approx(fine.compliance.sum(), rel=0.5)
