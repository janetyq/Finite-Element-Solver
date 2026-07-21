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

from fem.typing import ElementField, Matrix


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


def hooke_matrix(reference_dim: int, mu: float, lamb: float) -> Matrix:
    '''Isotropic elastic constitutive matrix D (strain -> stress) in Voigt form.

    Strain and stress are ordered [xx, yy, (zz,) engineering shears], matching
    the rows of `fem.forms.strain_displacement`. `reference_dim` is the element's
    own dimension (2 for a triangle, 3 for a tet), which for the planar meshes
    supported today equals the number of displacement components.
    '''
    if reference_dim == 2:
        return np.array([
            [2 * mu + lamb, lamb, 0],
            [lamb, 2 * mu + lamb, 0],
            [0, 0, mu],
        ], dtype=np.float64)
    if reference_dim == 3:
        return np.array([
            [2 * mu + lamb, lamb, lamb, 0, 0, 0],
            [lamb, 2 * mu + lamb, lamb, 0, 0, 0],
            [lamb, lamb, 2 * mu + lamb, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ], dtype=np.float64)
    raise NotImplementedError(
        f'no elastic constitutive matrix for reference_dim={reference_dim}'
    )


@dataclass(frozen=True)
class LinearElasticMaterial:
    '''Isotropic linear-elastic constitutive law, parameterised by E and nu.

    E may be a scalar or a per-element array -- TopologyOptimizer scales it by a
    density cubed each iteration -- so the constitutive matrix is requested per
    element rather than built once. nu is uniform.
    '''
    E: float | ElementField
    nu: float

    def constitutive_matrix(self, reference_dim: int, e_idx: int) -> Matrix:
        '''The Voigt D for element `e_idx`, at the element's `reference_dim`.'''
        E = self.E[e_idx] if isinstance(self.E, np.ndarray) else self.E
        mu, lamb = Enu_to_Lame(E, self.nu)
        return hooke_matrix(reference_dim, mu, lamb)
