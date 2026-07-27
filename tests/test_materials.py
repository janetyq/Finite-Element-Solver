"""The elastic constitutive law, and the plane-strain reduction it encodes."""
import numpy as np

from fem.materials import Enu_to_Lame, LinearElasticMaterial, hooke_matrix


def test_out_of_plane_stress_matches_the_3d_law_under_zero_axial_strain():
    """`out_of_plane_stress` must agree with what the full 3D law produces for a
    strain state with epsilon_zz = 0 -- the definition of plane strain.

    Built independently: take an in-plane strain, run it through the *3D* Hooke
    matrix with a zero zz component, and read off sigma_zz. The 2D shortcut
    nu(sigma_xx + sigma_yy) must reproduce it.
    """
    E, nu = 210.0, 0.3
    material = LinearElasticMaterial(E, nu)
    mu, lamb = Enu_to_Lame(E, nu)

    # A plane-strain state: nonzero in-plane strain, epsilon_zz = 0.
    eps_xx, eps_yy, gamma_xy = 0.004, -0.002, 0.003
    strain_3d = np.array([eps_xx, eps_yy, 0.0, gamma_xy, 0.0, 0.0])
    stress_3d = hooke_matrix(3, mu, lamb) @ strain_3d

    sigma_zz = material.out_of_plane_stress(stress_3d[0], stress_3d[1])
    np.testing.assert_allclose(sigma_zz, stress_3d[2], rtol=1e-12)


def test_out_of_plane_stress_is_batched():
    """Post-processing hands it a per-element array, not a scalar."""
    material = LinearElasticMaterial(1.0, 0.25)
    sxx = np.array([1.0, 2.0, -3.0])
    syy = np.array([0.5, -1.0, 4.0])

    np.testing.assert_allclose(
        material.out_of_plane_stress(sxx, syy), 0.25 * (sxx + syy)
    )


def test_incompressible_limit_carries_the_full_in_plane_stress():
    """At nu -> 1/2 the material cannot change volume, so the restrained z
    direction carries the mean of the in-plane stresses. A sanity anchor on the
    coefficient: it is nu, not 1 - nu or nu/(1 - nu)."""
    material = LinearElasticMaterial(1.0, 0.5)
    np.testing.assert_allclose(material.out_of_plane_stress(2.0, 4.0), 3.0)
