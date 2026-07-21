"""The forms produce correct element matrices.

The stiffness integrand lives in `fem.forms`, with the strain-displacement matrix
B (here) and the constitutive matrix D (`fem.materials`) in separate files sharing
one Voigt ordering. These tests pin that pairing against known-correct references,
so a change that desynchronizes B and D fails fast and locally rather than only
showing up as a broken convergence rate.
"""
import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.forms import LaplacianForm, LinearElasticForm, strain_displacement
from fem.materials import LinearElasticMaterial

# Reference simplices: the unit right triangle and tet, and a unit line.
TRI = LinearTriangleElement(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
TET = LinearTetrahedralElement(
    np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
)
LINE = LinearLineElement(np.array([[0.0], [1.0]]))


def test_laplacian_matches_analytic_unit_triangle():
    """The P1 Laplacian on the unit right triangle is the textbook stiffness."""
    expected = np.array([[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]])
    np.testing.assert_allclose(LaplacianForm().element_matrix(TRI, 0), expected)


def test_laplacian_matches_analytic_unit_line():
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(LaplacianForm().element_matrix(LINE, 0), expected)


@pytest.mark.parametrize("element", [TRI, TET, LINE])
def test_laplacian_is_symmetric_and_annihilates_constants(element):
    """A Laplacian stiffness is symmetric and has the constant vector in its
    null space (rows sum to zero), whatever the element."""
    K = LaplacianForm().element_matrix(element, 0)
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(K.sum(axis=1), 0, atol=1e-12)


def test_strain_displacement_maps_uniform_stretch_to_uniform_strain():
    """A unit x-stretch (u_x = x, u_y = 0) has strain [1, 0, 0] everywhere."""
    u = np.array([0.0, 0, 1, 0, 0, 0])  # node coords are (0,0),(1,0),(0,1)
    np.testing.assert_allclose(strain_displacement(TRI.grad_phi) @ u, [1.0, 0.0, 0.0])


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
    np.testing.assert_allclose(form.element_matrix(TRI, 0), expected, atol=1e-5)


@pytest.mark.parametrize("element", [TRI, TET])
def test_elastic_stiffness_is_symmetric_with_rigid_body_nullspace(element):
    """An elastic stiffness is symmetric, and a rigid translation stores no
    energy (a constant displacement in any one component sums to zero per row)."""
    form = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))
    K = form.element_matrix(element, 0)
    np.testing.assert_allclose(K, K.T)
    d = element.reference_dim
    for component in range(d):
        translation = np.zeros(K.shape[0])
        translation[component::d] = 1.0
        np.testing.assert_allclose(K @ translation, 0, atol=1e-10)


def test_elastic_form_reads_per_element_modulus():
    """A per-element E array indexes by e_idx, as TopologyOptimizer relies on."""
    stiff_uniform = LinearElasticForm(LinearElasticMaterial(200.0, 0.3)).element_matrix(TRI, 1)

    material = LinearElasticMaterial(np.array([100.0, 200.0, 300.0]), 0.3)
    form = LinearElasticForm(material)
    np.testing.assert_allclose(form.element_matrix(TRI, 1), stiff_uniform)
    # Element 0 sees E=100, not 200 -> a different (softer) matrix.
    assert not np.allclose(form.element_matrix(TRI, 0), stiff_uniform)
