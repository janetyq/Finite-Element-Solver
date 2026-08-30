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
from fem.physics.forms import (
    BoundaryMassForm, DiffusionForm, EnergyForm, GeometricStiffnessForm, LinearElasticForm, MassForm,
    PrecomputedForm, ScaledForm, SumForm, rigid_body_modes, strain_displacement, voigt_to_tensor,
)
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.space import FunctionSpace


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


def test_elastic_stiffness_matches_the_hand_assembled_reference_triangle():
    """K = area * B^T D B on the unit right triangle, with B written out from the P1
    gradients (-1,-1), (1,0), (0,1) and D from the plane-strain Lame constants, so the
    reference is derived here rather than recorded from the implementation."""
    E, nu = 200.0, 0.3
    mu, lam = Enu_to_Lame(E, nu)
    D = np.array([[lam + 2 * mu, lam, 0.0], [lam, lam + 2 * mu, 0.0], [0.0, 0.0, mu]])
    grads = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])   # d(phi_a)/dx, d(phi_a)/dy
    B = np.zeros((3, 6))
    for a, (dx, dy) in enumerate(grads):
        B[0, 2 * a] = dx
        B[1, 2 * a + 1] = dy
        B[2, 2 * a], B[2, 2 * a + 1] = dy, dx
    expected = 0.5 * B.T @ D @ B

    form = LinearElasticForm(LinearElasticMaterial(E, nu))
    np.testing.assert_allclose(form.element_matrices(TRI)[0], expected, atol=1e-10)


@pytest.mark.parametrize('form', [
    DiffusionForm(), MassForm(), LinearElasticForm(LinearElasticMaterial(200.0, 0.3)),
], ids=lambda f: type(f).__name__)
def test_element_matrices_do_not_depend_on_node_orientation(form):
    """The measure enters as |det J|, so a triangle listed clockwise assembles the same
    matrix as its counter-clockwise twin once the node blocks are put back in order."""
    ccw = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    K_ccw = form.element_matrices(LinearTriangleElement.geometry(ccw))[0]
    K_cw = form.element_matrices(LinearTriangleElement.geometry(ccw[:, ::-1]))[0]

    per_node = K_ccw.shape[0] // 3
    undo = np.concatenate([np.arange(per_node * a, per_node * (a + 1)) for a in (2, 1, 0)])
    np.testing.assert_allclose(K_cw[np.ix_(undo, undo)], K_ccw, atol=1e-12)


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
    b = Source(3.0).element_vectors(TRI, 1)[0]   # (N,)
    np.testing.assert_allclose(b, 3.0 * volume / 3)  # 3 nodes, integral of a P1 hat
    np.testing.assert_allclose(b.sum(), 3.0 * volume)


# -- refusals -------------------------------------------------------------------


def test_a_sum_needs_two_flat_terms():
    with pytest.raises(ValueError, match='at least two terms'):
        SumForm((DiffusionForm(),))
    with pytest.raises(ValueError, match='is flat'):
        SumForm((DiffusionForm() + MassForm(), MassForm()))


def test_a_scale_distributes_over_a_sum_rather_than_wrapping_it():
    """`factor * (a + b)` is a sum of scaled terms; wrapping the sum itself is refused."""
    scaled = 2.0 * (DiffusionForm() + MassForm())
    assert isinstance(scaled, SumForm)
    assert all(isinstance(term, ScaledForm) for term in scaled.terms)
    with pytest.raises(TypeError, match='scale the terms of a sum'):
        ScaledForm(2.0, DiffusionForm() + MassForm())


@pytest.mark.parametrize('method', ['element_matrices', 'element_residuals', 'element_tangents', 'element_energies'])
def test_a_sum_has_no_element_blocks_of_its_own(method):
    total = DiffusionForm() + MassForm()
    args = (TRI,) if method == 'element_matrices' else (TRI, np.zeros((1, 3)))
    with pytest.raises(TypeError, match='no element blocks'):
        getattr(total, method)(*args)


def test_a_sum_answers_for_one_physics_term_only():
    """Two terms naming a flux, or a near-null space, leave the sum unable to say which
    physics it solves; a boundary mass beside a Laplacian is the intended shape."""
    space = FunctionSpace(box_mesh(corners=[[0, 0], [1, 1]], resolution=(3, 3)), n_components=2)
    with pytest.raises(ValueError, match='more than one term of the sum names a flux'):
        (DiffusionForm() + DiffusionForm()).flux()
    elastic = LinearElasticForm(LinearElasticMaterial(200.0, 0.3))
    with pytest.raises(ValueError, match='more than one term of the sum names a near-null space'):
        (elastic + elastic).near_null_space(space)


def test_rigid_body_modes_need_a_plane_or_a_solid():
    with pytest.raises(ValueError, match='rigid-body modes are defined for 2D or 3D'):
        rigid_body_modes(np.zeros((4, 1)), 1)


def test_voigt_to_tensor_takes_three_or_six_components():
    with pytest.raises(ValueError, match=r'expected 3 \(2D\) or 6 \(3D\) Voigt components'):
        voigt_to_tensor(np.zeros((2, 4)))


def test_strain_displacement_is_defined_in_2d_and_3d_only():
    with pytest.raises(NotImplementedError, match='no strain-displacement matrix for dim=1'):
        strain_displacement(LINE.grad_phi)
