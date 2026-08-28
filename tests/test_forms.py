"""The forms produce correct element matrices.

The strain-displacement matrix B (`fem.physics.forms`) and the constitutive matrix D
(`fem.physics.materials`) share one Voigt ordering; these pin that pairing against references,
so a change that desynchronizes them fails here rather than as a broken convergence
rate. Forms are batched, so a single reference simplex is a batch of one.
"""
import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.physics.energies import StVenantKirchhoff
from fem.physics.forms import DiffusionForm, EnergyForm, GeometricStiffnessForm, LinearElasticForm, BoundaryMassForm, MassForm, PrecomputedForm, ScaledForm, strain_displacement
from fem.loads import Source
from fem.physics.materials import LinearElasticMaterial


def one(element_type, vertices):
    """A batch-of-one ElementGeometry for a single reference simplex."""
    return element_type.geometry(np.asarray(vertices, dtype=float)[None])


# Reference simplices: the unit right triangle and tet, and a unit line.
TRI = one(LinearTriangleElement, [[0, 0], [1, 0], [0, 1]])
TET = one(LinearTetrahedralElement, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
LINE = one(LinearLineElement, [[0], [1]])


def test_laplacian_matches_analytic_unit_triangle():
    """The P1 Laplacian on the unit right triangle is the textbook stiffness."""
    expected = np.array([[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]])
    np.testing.assert_allclose(DiffusionForm().element_matrices(TRI)[0], expected)


def test_laplacian_matches_analytic_unit_line():
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(DiffusionForm().element_matrices(LINE)[0], expected)


def test_mass_form_scalar_matches_consistent_mass():
    """The scalar mass form is the element's consistent P1 mass matrix, and it
    integrates a unit field to the element volume (its row sum)."""
    M = MassForm().element_matrices(TRI)[0]
    reference = LinearTriangleElement.reference_mass_matrix() * TRI.volumes[0]
    np.testing.assert_allclose(M, reference)
    np.testing.assert_allclose(M.sum(), TRI.volumes[0])


@pytest.mark.parametrize("geometry", [TRI, TET])
def test_mass_form_replicates_scalar_per_component(geometry):
    """A k-component mass form is the scalar mass matrix Kronecker the identity."""
    scalar = MassForm(1).element_matrices(geometry)[0]
    k = geometry.reference_dim
    np.testing.assert_allclose(
        MassForm(k).element_matrices(geometry)[0], np.kron(scalar, np.eye(k))
    )


@pytest.mark.parametrize("geometry", [TRI, TET, LINE])
def test_laplacian_is_symmetric_and_annihilates_constants(geometry):
    """A Laplacian stiffness is symmetric and has the constant vector in its
    null space (rows sum to zero), whatever the element."""
    K = DiffusionForm().element_matrices(geometry)[0]
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-12)


def test_strain_displacement_maps_uniform_stretch_to_uniform_strain():
    """A unit x-stretch (u_x = x, u_y = 0) has strain [1, 0, 0] everywhere."""
    u = np.array([0.0, 0, 1, 0, 0, 0])  # node coords are (0,0),(1,0),(0,1)
    # grad_phi is (n_el, n_qp, N, dim); index element 0 and its single quad point.
    B = strain_displacement(TRI.grad_phi)[0, 0]
    np.testing.assert_allclose(B @ u, [1.0, 0.0, 0.0])


def test_strain_displacement_is_batched_over_elements():
    """Each element's B is built from its own gradients, not the first element's."""
    pair = LinearTriangleElement.geometry(np.array([
        [[0.0, 0], [1, 0], [0, 1]],
        [[0.0, 0], [2, 0], [0, 2]],  # twice the size -> half the gradients
    ]))
    B = strain_displacement(pair.grad_phi)
    assert B.shape == (2, 1, 3, 6)   # (n_el, n_qp, n_strains, N*dim)
    np.testing.assert_allclose(B[1], 0.5 * B[0])


def test_elastic_stiffness_matches_reference_triangle():
    """Golden element stiffness for the unit triangle at E=200, nu=0.3, captured
    from B^T D B and independently reproducible from plane-strain Lame values."""
    form = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))
    expected = np.array([
        [173.076923, 96.153846, -134.615385, -38.461538, -38.461538, -57.692308],
        [96.153846, 173.076923, -57.692308, -38.461538, -38.461538, -134.615385],
        [-134.615385, -57.692308, 134.615385, 0.0, 0.0, 57.692308],
        [-38.461538, -38.461538, 0.0, 38.461538, 38.461538, 0.0],
        [-38.461538, -38.461538, 0.0, 38.461538, 38.461538, 0.0],
        [-57.692308, -134.615385, 57.692308, 0.0, 0.0, 134.615385],
    ])
    np.testing.assert_allclose(form.element_matrices(TRI)[0], expected, atol=1e-5)


@pytest.mark.parametrize("geometry", [TRI, TET])
def test_elastic_stiffness_is_symmetric_with_rigid_body_nullspace(geometry):
    """An elastic stiffness is symmetric, and a rigid translation stores no
    energy (a constant displacement in any one component sums to zero per row)."""
    form = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))
    K = form.element_matrices(geometry)[0]
    np.testing.assert_allclose(K, K.T)
    d = geometry.reference_dim
    for component in range(d):
        translation = np.zeros(K.shape[0])
        translation[component::d] = 1.0
        np.testing.assert_allclose(K @ translation, 0, atol=1e-10)


def test_elastic_form_reads_per_element_modulus():
    """A per-element E array is applied element-wise, as SIMP density design relies on."""
    triangles = LinearTriangleElement.geometry(
        np.repeat(np.array([[[0.0, 0], [1, 0], [0, 1]]]), 3, axis=0)
    )
    uniform = LinearElasticForm(
        LinearElasticMaterial(200.0, 0.3)
    ).element_matrices(triangles)

    form = LinearElasticForm(LinearElasticMaterial(np.array([100.0, 200.0, 300.0]), 0.3))
    varying = form.element_matrices(triangles)
    # Identical geometry, so only the modulus distinguishes the three. D is linear
    # in E at fixed nu, so the ratios carry straight through to the stiffness.
    np.testing.assert_allclose(varying[1], uniform[1])
    np.testing.assert_allclose(varying[0], 0.5 * uniform[0])
    np.testing.assert_allclose(varying[2], 1.5 * uniform[2])


def test_per_element_modulus_length_is_checked():
    """A modulus array that does not match the mesh is a caller error, not a broadcast."""
    material = LinearElasticMaterial(np.array([100.0, 200.0]), 0.3)
    with pytest.raises(ValueError, match='2 entries but the mesh has 1'):
        LinearElasticForm(material).element_matrices(TRI)


def test_diffusion_form_with_unit_coefficient_is_the_laplacian():
    """kappa == 1 recovers the constant-coefficient Laplacian, element for element --
    the variable-coefficient form's constant case is the form it generalizes."""
    np.testing.assert_allclose(
        DiffusionForm(1.0).element_matrices(TRI),
        DiffusionForm().element_matrices(TRI),
    )


def test_diffusion_form_scales_with_the_coefficient():
    """A constant kappa scales the stiffness by kappa, since it factors out of the integral."""
    np.testing.assert_allclose(
        DiffusionForm(5.0).element_matrices(TRI),
        5.0 * DiffusionForm().element_matrices(TRI),
    )


# Every form, with the components per node it is written for on the unit triangle.
EVERY_FORM = [
    (MassForm(2), 2),
    (BoundaryMassForm(2, np.array([True])), 2),
    (DiffusionForm(), 1),
    (DiffusionForm(2.0), 1),
    (LinearElasticForm(LinearElasticMaterial(200.0, 0.3)), 2),
    (GeometricStiffnessForm(np.zeros((1, 2, 2))), 2),
    (ScaledForm(3.0, DiffusionForm()), 1),
    (PrecomputedForm(np.eye(3)[None]), 1),
    (EnergyForm(StVenantKirchhoff(200.0, 0.3)), 2),
]


@pytest.mark.parametrize('form, n_components', EVERY_FORM, ids=lambda x: type(x).__name__ if not isinstance(x, int) else '')
def test_every_form_answers_what_it_declares(form, n_components):
    """A form's flags agree with its methods: `constant_tangent` means the tangent is the
    same at two states, `has_energy` means `element_energies` is defined, and the
    residual and tangent blocks have the shapes the assembly scatters."""
    rng = np.random.default_rng(0)
    u0 = 0.1 * rng.standard_normal((1, 3, n_components))
    u1 = 0.1 * rng.standard_normal((1, 3, n_components))
    k = 3 * n_components

    assert form.element_residuals(TRI, u0).shape == (1, k)
    t0, t1 = form.element_tangents(TRI, u0), form.element_tangents(TRI, u1)
    assert t0.shape == (1, k, k)
    assert np.allclose(t0, t1) == form.constant_tangent

    if form.has_energy:
        assert form.element_energies(TRI, u0).shape == (1,)
    else:
        with pytest.raises(NotImplementedError):
            form.element_energies(TRI, u0)


def test_linear_form_constant_source_integrates_the_hat_exactly():
    """For a constant source c, each node's load is c * volume / N and the loads sum to
    c * volume."""
    volume = float(TRI.volumes[0])
    b = Source(3.0, 1).element_vectors(TRI)[0]   # (N,)
    np.testing.assert_allclose(b, 3.0 * volume / 3)  # 3 nodes, integral of a P1 hat
    np.testing.assert_allclose(b.sum(), 3.0 * volume)
