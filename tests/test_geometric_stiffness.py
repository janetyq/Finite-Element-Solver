"""The geometric (initial-stress) stiffness form produces correct element matrices.

`GeometricStiffnessForm` is the one operator whose "material" is a supplied stress
rather than a constitutive law, so these pin it against properties that hold for any
prestress (symmetry, linearity, the rigid-translation null space) plus two analytic
values: under a hydrostatic prestress it collapses to a scaled Laplacian, and a
uniaxial prestress on the unit triangle has a hand-computable block.

The physics -- that this is the matrix whose competition with the elastic stiffness
sets the buckling load -- is checked in `tests/test_buckling.py`.
"""
import numpy as np
import pytest

from fem.elements import LinearTetrahedralElement, LinearTriangleElement
from fem.forms import GeometricStiffnessForm, LaplacianForm


def one(element_type, vertices):
    """A batch-of-one ElementGeometry for a single reference simplex."""
    return element_type.geometry(np.asarray(vertices, dtype=float)[None])


TRI = one(LinearTriangleElement, [[0, 0], [1, 0], [0, 1]])
TET = one(LinearTetrahedralElement, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _prestress(geometry, sigma):
    """Broadcast one (d, d) prestress tensor across every element of `geometry`."""
    d = geometry.spatial_dim
    return np.broadcast_to(np.asarray(sigma, dtype=float), (geometry.n_elements, d, d))


@pytest.mark.parametrize("geometry", [TRI, TET])
def test_geometric_stiffness_is_symmetric(geometry):
    """K_g is symmetric for any prestress -- it is a Gᵀ Σ G form with Σ symmetric."""
    d = geometry.spatial_dim
    rng = np.random.default_rng(0)
    sigma = rng.normal(size=(d, d))
    sigma = 0.5 * (sigma + sigma.T)   # a stress tensor is symmetric
    K = GeometricStiffnessForm(_prestress(geometry, sigma)).element_matrices(geometry)[0]
    np.testing.assert_allclose(K, K.T, atol=1e-12)


@pytest.mark.parametrize("geometry", [TRI, TET])
def test_geometric_stiffness_is_linear_in_the_prestress(geometry):
    """The stress factors out of the integral, so scaling it scales K_g, and a zero
    prestress gives a zero matrix (an unstressed structure has no geometric stiffness)."""
    d = geometry.spatial_dim
    rng = np.random.default_rng(1)
    sigma = rng.normal(size=(d, d))
    base = GeometricStiffnessForm(_prestress(geometry, sigma)).element_matrices(geometry)
    scaled = GeometricStiffnessForm(_prestress(geometry, -2.5 * sigma)).element_matrices(geometry)
    zero = GeometricStiffnessForm(_prestress(geometry, 0 * sigma)).element_matrices(geometry)

    np.testing.assert_allclose(scaled, -2.5 * base, atol=1e-12)
    np.testing.assert_allclose(zero, 0.0, atol=1e-14)


@pytest.mark.parametrize("geometry", [TRI, TET])
def test_hydrostatic_prestress_is_a_scaled_laplacian(geometry):
    """Under a hydrostatic prestress p·I, Σ₀ = p·I and the form reduces to
    p ∫ ∇φ_a·∇φ_b per component -- exactly p times the block-expanded Laplacian
    stiffness. The cleanest analytic anchor: it ties K_g to a form already tested."""
    d = geometry.spatial_dim
    p = 3.0
    K_g = GeometricStiffnessForm(_prestress(geometry, p * np.eye(d))).element_matrices(geometry)[0]
    laplacian = LaplacianForm().element_matrices(geometry)[0]
    np.testing.assert_allclose(K_g, p * np.kron(laplacian, np.eye(d)), atol=1e-12)


def test_uniaxial_prestress_on_unit_triangle_matches_hand_value():
    """A pure σ_xx prestress on the unit right triangle couples only through the
    x-derivatives of the hats. With grad_phi_x = (-1, 1, 0) and area 1/2, the scalar
    block is (s·area)·outer(gx, gx), expanded per component."""
    s = 7.0
    K_g = GeometricStiffnessForm(_prestress(TRI, [[s, 0], [0, 0]])).element_matrices(TRI)[0]

    gx = np.array([-1.0, 1.0, 0.0])
    area = 0.5
    block = s * area * np.outer(gx, gx)
    np.testing.assert_allclose(K_g, np.kron(block, np.eye(2)), atol=1e-12)


def test_geometric_stiffness_annihilates_rigid_translations():
    """A rigid translation stores no geometric energy: ∑_b ∇φ_b = 0 (partition of
    unity), so each component's rows sum to zero, just as an elastic stiffness's do."""
    K_g = GeometricStiffnessForm(_prestress(TRI, [[2.0, 1.0], [1.0, -3.0]])).element_matrices(TRI)[0]
    for component in range(2):
        translation = np.zeros(K_g.shape[0])
        translation[component::2] = 1.0
        np.testing.assert_allclose(K_g @ translation, 0.0, atol=1e-12)


def test_geometric_stiffness_is_batched_per_element():
    """Each element uses its own prestress, not the first element's."""
    triangles = LinearTriangleElement.geometry(
        np.repeat(np.array([[[0.0, 0], [1, 0], [0, 1]]]), 2, axis=0)
    )
    prestress = np.stack([
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[5.0, 0.0], [0.0, 0.0]]),   # five times the first
    ])
    K = GeometricStiffnessForm(prestress).element_matrices(triangles)
    np.testing.assert_allclose(K[1], 5.0 * K[0], atol=1e-12)


def test_prestress_shape_is_checked():
    """A prestress that does not cover the mesh is a caller error, not a broadcast."""
    bad = np.zeros((2, 2, 2))   # two elements, but TRI has one
    with pytest.raises(ValueError, match='prestress must be'):
        GeometricStiffnessForm(bad).element_matrices(TRI)
