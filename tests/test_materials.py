"""The elastic constitutive law, and the plane-strain reduction it encodes."""
import numpy as np

from fem.materials import Enu_to_Lame, LinearElasticMaterial, hooke_matrix

E, NU = 210.0, 0.3


def _plane_strain_state(eps_xx, eps_yy, eps_xy):
    """An in-plane strain tensor and the full 3D stress the law gives it, the reference for
    the 2D shortcut."""
    mu, lamb = Enu_to_Lame(E, NU)
    voigt_3d = np.array([eps_xx, eps_yy, 0.0, 2 * eps_xy, 0.0, 0.0])
    stress_3d = hooke_matrix(3, mu, lamb) @ voigt_3d
    strain = np.array([[[eps_xx, eps_xy], [eps_xy, eps_yy]]])
    return strain, stress_3d


def test_out_of_plane_stress_matches_the_3d_law_under_zero_axial_strain():
    """`out_of_plane_stress` reproduces what the full 3D law gives for epsilon_zz = 0."""
    strain, stress_3d = _plane_strain_state(0.004, -0.002, 0.0015)

    sigma_zz = LinearElasticMaterial(E, NU).out_of_plane_stress(strain)
    np.testing.assert_allclose(sigma_zz, stress_3d[2], rtol=1e-12)


def test_lambda_trace_and_nu_sum_forms_agree():
    """`lambda * tr(eps)` and `nu * (sigma_xx + sigma_yy)` are the same number."""
    strain, stress_3d = _plane_strain_state(0.004, -0.002, 0.0015)

    lambda_form = LinearElasticMaterial(E, NU).out_of_plane_stress(strain)
    nu_form = NU * (stress_3d[0] + stress_3d[1])
    np.testing.assert_allclose(lambda_form, nu_form, rtol=1e-12)


def test_out_of_plane_stress_is_batched():
    """Post-processing hands it one tensor per element, not a single state."""
    rng = np.random.default_rng(0)
    strain = rng.normal(size=(5, 2, 2))
    strain = strain + np.swapaxes(strain, -2, -1)

    _, lamb = Enu_to_Lame(E, NU)
    got = LinearElasticMaterial(E, NU).out_of_plane_stress(strain)

    assert got.shape == (5,)
    np.testing.assert_allclose(got, lamb * np.trace(strain, axis1=-2, axis2=-1))


def test_per_element_modulus_gives_a_per_element_out_of_plane_stress():
    """SIMP design scales E per element, so lambda is an array there too."""
    moduli = np.array([1.0, 2.0, 4.0])
    strain = np.tile(np.eye(2) * 0.01, (3, 1, 1))

    got = LinearElasticMaterial(moduli, NU).out_of_plane_stress(strain)

    _, lamb = Enu_to_Lame(moduli, NU)
    np.testing.assert_allclose(got, lamb * 0.02)


def test_incompressible_limit_carries_the_full_in_plane_stress():
    """At nu -> 1/2 the material cannot change volume, so a purely deviatoric
    in-plane strain develops no out-of-plane stress, while a dilational one
    develops an unbounded lambda. Anchors which constant scales the trace."""
    deviatoric = np.array([[[0.01, 0.0], [0.0, -0.01]]])
    material = LinearElasticMaterial(E, 0.4999)

    np.testing.assert_allclose(
        material.out_of_plane_stress(deviatoric), 0.0, atol=1e-9
    )
