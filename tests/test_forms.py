"""Forms reproduce the element matrices that used to live on Element.

The stiffness integrand moved off `Element.calculate_stiffness_matrix` into
`fem.forms`, and the strain-displacement matrix B (in forms) and the constitutive
matrix D (in materials) are now in separate files but must share a Voigt ordering.
These tests pin both: each form's element matrix equals the pre-refactor
`element.calculate_stiffness_matrix`, evaluated straight from the element geometry.
"""
import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.forms import LaplacianForm, LinearElasticForm, strain_displacement
from fem.materials import Enu_to_Lame, LinearElasticMaterial

TRI = LinearTriangleElement(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
TET = LinearTetrahedralElement(
    np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
)
LINE = LinearLineElement(np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("element", [TRI, TET, LINE])
def test_laplacian_form_matches_scalar_stiffness(element):
    expected = element.calculate_stiffness_matrix(n_components=1)
    np.testing.assert_allclose(LaplacianForm().element_matrix(element, 0), expected)


@pytest.mark.parametrize("element", [TRI, TET])
def test_strain_displacement_matches_element_B(element):
    np.testing.assert_allclose(strain_displacement(element.grad_phi), element.calculate_B())


@pytest.mark.parametrize("element", [TRI, TET])
def test_elastic_form_matches_vector_stiffness(element):
    E, nu = 200.0, 0.3
    mu, lamb = Enu_to_Lame(E, nu)
    n_components = element.reference_dim
    expected = element.calculate_stiffness_matrix(
        n_components, idx=0, mu=[mu], lamb=[lamb]
    )

    form = LinearElasticForm(LinearElasticMaterial(E, nu))
    np.testing.assert_allclose(form.element_matrix(element, 0), expected)


def test_elastic_form_reads_per_element_modulus():
    """A per-element E array indexes by e_idx, as TopologyOptimizer relies on."""
    material = LinearElasticMaterial(np.array([100.0, 200.0, 300.0]), 0.3)
    form = LinearElasticForm(material)

    stiff_1 = form.element_matrix(TRI, 1)
    mu, lamb = Enu_to_Lame(200.0, 0.3)
    expected = TRI.calculate_stiffness_matrix(2, idx=0, mu=[mu], lamb=[lamb])
    np.testing.assert_allclose(stiff_1, expected)

    # Different element index -> different modulus -> different matrix.
    assert not np.allclose(form.element_matrix(TRI, 0), stiff_1)
