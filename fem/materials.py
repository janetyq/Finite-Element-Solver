"""Material property conversions and the isotropic elastic constitutive law.

Conversions between engineering constants (Young's modulus E, Poisson's ratio nu)
and Lame parameters (shear modulus mu, lambda), plus `LinearElasticMaterial`: the
map from strain to stress, which used to live on `Element` as `calculate_D`. It is
a material law, not element geometry, so it belongs here rather than on the shape.

`hooke_matrix` fixes the Voigt ordering of the constitutive matrix D; the
strain-displacement matrix B in `fem.forms` must order its strain rows the same
way, since D and B are contracted against each other. The two are the shared
convention referred to in both files.
"""
from dataclasses import dataclass

import numpy as np

from fem.typing import ElementField, FloatArray, Matrix


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
    material. Writing it this way is what lets one element and a whole mesh of
    elements with different moduli share an implementation: scale by scalars for
    the first, by `(n_elements, 1, 1)` arrays for the second.

    In Voigt form with strain ordered [xx, yy, (zz,) engineering shears], P_mu is
    diagonal -- 2 on the normal components, 1 on the shears -- and P_lamb is the
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
    the rows of `fem.forms.strain_displacement`. `reference_dim` is the element's
    own dimension (2 for a triangle, 3 for a tet), which for the planar meshes
    supported today equals the number of displacement components.
    '''
    P_mu, P_lamb = hooke_patterns(reference_dim)
    return mu * P_mu + lamb * P_lamb


@dataclass(frozen=True)
class LinearElasticMaterial:
    '''Isotropic linear-elastic constitutive law, parameterised by E and nu.

    E may be a scalar or a per-element array -- TopologyOptimizer scales it by a
    density cubed each iteration -- so the constitutive matrix is requested per
    element rather than built once. nu is uniform.
    '''
    E: float | ElementField
    nu: float

    def constitutive_matrices(self, reference_dim: int, n_elements: int) -> FloatArray:
        '''(n_elements, s, s) Voigt D, one per element -- the batched assembly path.

        A uniform modulus returns a broadcast *view* of the single matrix rather
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
