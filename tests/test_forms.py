"""The forms produce correct element matrices.

The stiffness integrand lives in `fem.forms`, with the strain-displacement matrix
B (here) and the constitutive matrix D (`fem.materials`) in separate files sharing
one Voigt ordering. These tests pin that pairing against known-correct references,
so a change that desynchronizes B and D fails fast and locally rather than only
showing up as a broken convergence rate.

Forms are batched -- they take an `ElementGeometry` for a whole mesh and return
`(n_elements, k, k)` -- so a single reference simplex is a batch of one, and the
assertions index `[0]` out of the result.
"""
import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.forms import LaplacianForm, LinearElasticForm, MassForm, strain_displacement
from fem.materials import LinearElasticMaterial


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
    np.testing.assert_allclose(LaplacianForm().element_matrices(TRI)[0], expected)


def test_laplacian_matches_analytic_unit_line():
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(LaplacianForm().element_matrices(LINE)[0], expected)


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
    K = LaplacianForm().element_matrices(geometry)[0]
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-12)


def test_strain_displacement_maps_uniform_stretch_to_uniform_strain():
    """A unit x-stretch (u_x = x, u_y = 0) has strain [1, 0, 0] everywhere."""
    u = np.array([0.0, 0, 1, 0, 0, 0])  # node coords are (0,0),(1,0),(0,1)
    B = strain_displacement(TRI.grad_phi)[0]
    np.testing.assert_allclose(B @ u, [1.0, 0.0, 0.0])


def test_strain_displacement_is_batched_over_elements():
    """Each element's B is built from its own gradients, not the first element's."""
    pair = LinearTriangleElement.geometry(np.array([
        [[0.0, 0], [1, 0], [0, 1]],
        [[0.0, 0], [2, 0], [0, 2]],  # twice the size -> half the gradients
    ]))
    B = strain_displacement(pair.grad_phi)
    assert B.shape == (2, 3, 6)
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
    """A per-element E array is applied element-wise, as TopologyOptimizer relies on."""
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
