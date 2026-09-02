"""Material property conversions and the isotropic elastic constitutive law.

Conversions between engineering constants (Young's modulus E, Poisson's ratio nu)
and Lame parameters (shear modulus mu, lambda), plus `LinearElasticMaterial`, the
map from strain to stress.

`hooke_matrix` fixes the Voigt ordering of the constitutive matrix D; the
strain-displacement matrix B in `fem.physics.forms` orders its strain rows the same way.

In 2D the law is plane strain throughout. `LinearElasticMaterial.out_of_plane_stress`
names that assumption and supplies the `sigma_zz` a 2D Voigt vector omits, which
post-processing needs to build a complete stress tensor.

`eigenstress` is the same reduction for a stress-free strain (thermal expansion, a
plastic strain): the eigenstrain is always a full 3x3 tensor, the eigenstress is the 3D
law applied to it, and a 2D form reads the in-plane rows and the `zz` entry off that.
The reduction is written here once, so no form has to remember it.
"""
from dataclasses import dataclass

import numpy as np

from fem.typing import ElementValues, FloatArray, Matrix


def Enu_to_Lame(E, nu):
    # mu - shear modulus, lambda - Lame constant
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lamb


def Lame_to_Enu(mu, lamb):
    # E - Young's modulus, nu - Poisson's ratio
    E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
    nu = lamb / (2 * (lamb + mu))
    return E, nu


def hooke_patterns(reference_dim: int) -> tuple[Matrix, Matrix]:
    '''The two constant matrices `D = mu * P_mu + lamb * P_lamb` is built from.

    The isotropic law is linear in the Lame parameters, so D decomposes into a
    part scaled by mu and a part scaled by lamb, neither depending on the
    material. That decomposition lets one element and a whole mesh of elements
    with different moduli share an implementation: scale by scalars for the first,
    by `(n_elements, 1, 1)` arrays for the second.

    In Voigt form with strain ordered [xx, yy, (zz,) engineering shears], P_mu is
    diagonal (2 on the normal components, 1 on the shears) and P_lamb is the
    all-ones block coupling the normal components, since lamb multiplies the
    trace of the strain. For reference_dim 2 that spells out to

        D = [[2mu+lamb, lamb, 0], [lamb, 2mu+lamb, 0], [0, 0, mu]]

    which is the form `tests/test_elasticity_models.py` checks against the second
    derivative of the small-strain energy.
    '''
    if reference_dim not in (2, 3):
        raise NotImplementedError(
            f'no elastic constitutive matrix for reference_dim={reference_dim}'
        )
    d = reference_dim
    n_shears = d * (d - 1) // 2
    P_mu = np.diag(np.array([2.0] * d + [1.0] * n_shears))
    P_lamb = np.zeros((d + n_shears, d + n_shears))
    P_lamb[:d, :d] = 1.0
    return P_mu, P_lamb


def hooke_matrix(reference_dim: int, mu: float, lamb: float) -> Matrix:
    '''Isotropic elastic constitutive matrix D (strain -> stress) in Voigt form.

    Strain and stress are ordered [xx, yy, (zz,) engineering shears], matching
    the rows of `fem.physics.forms.strain_displacement`. `reference_dim` is the element's
    own dimension (2 for a triangle, 3 for a tet), which for the planar meshes
    supported today equals the number of displacement components.
    '''
    P_mu, P_lamb = hooke_patterns(reference_dim)
    return mu * P_mu + lamb * P_lamb


@dataclass(frozen=True)
class LinearElasticMaterial:
    '''Isotropic linear-elastic constitutive law, parameterised by E and nu.

    E may be a scalar or a per-element array (SIMP density design scales it by a
    density cubed each iteration), so the constitutive matrix is requested per
    element rather than built once. nu is uniform.
    '''
    E: float | ElementValues
    nu: float

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        '''The stress in the restrained z direction, which a 2D solve omits.

        2D here means plane strain, the assumption `hooke_matrix(2, ...)`
        already encodes: the body is held fixed in z, so `epsilon_zz = 0` and the
        material develops `sigma_zz = lambda * tr(epsilon)` resisting that. The
        stress is real, but it falls outside the three Voigt components a 2D
        assembly produces, so von Mises or pressure built from those alone is
        computed on an incomplete tensor.

        Takes the in-plane strain tensor, matching the same method on the energy
        densities. The equivalent `nu(sigma_xx + sigma_yy)` is the same number;
        `tests/test_materials.py` pins both against the 3D law.
        '''
        _, lamb = Enu_to_Lame(self.E, self.nu)
        trace = np.einsum('eii->e', np.asarray(strain))
        return np.asarray(lamb) * trace

    def thermal_stress_modulus(self, alpha: float | ElementValues) -> float | ElementValues:
        '''`beta = (3 lambda + 2 mu) alpha`, the stress a fully restrained body develops
        per degree of heating: the coefficient of the thermal term in
        `sigma = C : eps - beta dT I`. Built from the 3D Lame constants in every
        dimension; `eigenstress` is what the solve path uses, this is the closed-form
        number the tests and the docs quote.'''
        mu, lamb = Enu_to_Lame(self.E, self.nu)
        return (3.0 * lamb + 2.0 * mu) * alpha

    def eigenstress(self, eigenstrain: FloatArray) -> FloatArray:
        '''The 3D eigenstress `sigma* = C_3D : eps*` of a stress-free strain, `(..., 3, 3)`
        in and out, batched over any leading axes (elements, then quadrature points).

        The eigenstrain is a full 3x3 tensor even for a 2D solve, and the law applied
        to it is the 3D one, with `lambda tr(eps*)` summing all three diagonal entries.
        That is the plane-strain reduction for an eigenstrain: with the total
        `eps_zz = 0`, restricting the 3D law to the plane gives

            sigma_ab = (D_2D eps)_ab - sigma*_ab,    sigma_zz = lambda tr(eps_in) - sigma*_zz

        so a 2D form subtracts the in-plane rows of this tensor from `D_2D eps` and the
        `zz` entry from `out_of_plane_stress`. Building the eigenstress from a 2D
        eigenstrain through `D_2D` instead drops `lambda eps*_zz`, the blocked
        out-of-plane expansion leaking into the plane through Poisson coupling: for a
        thermal strain it gives `(2 lambda + 2 mu) alpha dT` where `(3 lambda + 2 mu)
        alpha dT` is right. A per-element `E` broadcasts over the leading element axis.
        '''
        eigenstrain = np.asarray(eigenstrain, dtype=float)
        if eigenstrain.shape[-2:] != (3, 3):
            raise ValueError(
                f'an eigenstrain is a full (..., 3, 3) tensor in every dimension, got '
                f'shape {eigenstrain.shape}'
            )
        mu, lamb = (np.asarray(m, dtype=float) for m in Enu_to_Lame(self.E, self.nu))
        if mu.ndim:
            if eigenstrain.ndim < 3 or eigenstrain.shape[0] != len(mu):
                raise ValueError(
                    f'per-element modulus has {len(mu)} entries but the eigenstrain has '
                    f'leading shape {eigenstrain.shape[:-2]}'
                )
            # The element axis leads; any axes between it and the tensor broadcast.
            shape = (len(mu),) + (1,) * (eigenstrain.ndim - 3)
            mu, lamb = mu.reshape(shape), lamb.reshape(shape)
        trace = np.einsum('...ii->...', eigenstrain)
        return ((lamb * trace)[..., None, None] * np.eye(3)
                + (2.0 * mu)[..., None, None] * eigenstrain)

    def constitutive_matrices(self, reference_dim: int, n_elements: int) -> FloatArray:
        '''(n_elements, s, s) Voigt D, one per element: the batched assembly path.

        A uniform modulus returns a broadcast view of the single matrix rather
        than n_elements copies of it, so the common case costs no extra memory
        and `np.einsum` still contracts it against a per-element B.
        '''
        P_mu, P_lamb = hooke_patterns(reference_dim)
        if isinstance(self.E, np.ndarray):
            if len(self.E) != n_elements:
                raise ValueError(
                    f'per-element modulus has {len(self.E)} entries but the mesh has '
                    f'{n_elements} elements'
                )
            mu, lamb = Enu_to_Lame(self.E, self.nu)
            return mu[:, None, None] * P_mu + lamb[:, None, None] * P_lamb

        mu, lamb = Enu_to_Lame(self.E, self.nu)
        return np.broadcast_to(mu * P_mu + lamb * P_lamb, (n_elements, *P_mu.shape))
