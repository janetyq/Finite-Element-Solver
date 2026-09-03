"""Material property conversions and the isotropic elastic constitutive law.

Conversions between engineering constants (Young's modulus E, Poisson's ratio nu)
and Lame parameters (shear modulus mu, lambda), plus `LinearElasticMaterial`, the
map from strain to stress.

`hooke_matrix` fixes the Voigt ordering of the constitutive matrix D; the
strain-displacement matrix B in `fem.physics.forms` orders its strain rows the same way.

A 2D solve reduces a 3D body, and the material owns the `reduction`: plane strain (the
default, a long body held fixed in z) or plane stress (a thin plate free to contract in
z). The two differ in the in-plane law, in which out-of-plane component is nonzero
(`out_of_plane_stress` under plane strain, `out_of_plane_strain` under plane stress),
and in how a stress-free strain such as thermal expansion loads the plane
(`constrained_stress`). Everything else reads the material, so no form has to know which.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fem.typing import ElementValues, FloatArray, Matrix

Reduction = Literal['plane_strain', 'plane_stress']


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



def isotropic_stress(mu: FloatArray, lamb: FloatArray, strain: FloatArray) -> FloatArray:
    '''`lamb tr(eps) I + 2 mu eps` on `(..., d, d)` tensors, `mu` and `lamb` scalars or
    arrays that broadcast against the leading axes.'''
    d = strain.shape[-1]
    trace = np.einsum('...ii->...', strain)
    return (lamb * trace)[..., None, None] * np.eye(d) + (2.0 * mu)[..., None, None] * strain

@dataclass(frozen=True)
class LinearElasticMaterial:
    '''Isotropic linear-elastic constitutive law, parameterised by E and nu.

    E may be a scalar or a per-element array (SIMP density design scales it by a
    density cubed each iteration), so the constitutive matrix is requested per
    element rather than built once. nu is uniform.

    `reduction` is what a 2D solve means for the third direction. Plane strain, the
    default, holds the body fixed in z, as in a long body or a thick section: the
    strain `eps_zz` is zero and a stress `sigma_zz` develops. Plane stress leaves it
    free, as in a thin plate: `sigma_zz` is zero and the plate thins or thickens by
    `eps_zz`. A 3D solve has no reduction and accepts only the default.
    '''
    E: float | ElementValues
    nu: float
    reduction: Reduction = 'plane_strain'

    def __post_init__(self) -> None:
        if self.reduction not in ('plane_strain', 'plane_stress'):
            raise ValueError(
                f"reduction must be 'plane_strain' or 'plane_stress', got {self.reduction!r}"
            )

    @property
    def plane_stress(self) -> bool:
        return self.reduction == 'plane_stress'

    def lame(self) -> tuple[FloatArray, FloatArray]:
        '''The 3D `(mu, lamb)` as arrays, one entry per element for a per-element `E`.'''
        mu, lamb = Enu_to_Lame(self.E, self.nu)
        return np.asarray(mu, dtype=float), np.asarray(lamb, dtype=float)

    def require_dimension(self, spatial_dim: int) -> None:
        '''Refuse a reduction a solve in `spatial_dim` has no meaning for.'''
        if spatial_dim == 3 and self.plane_stress:
            raise ValueError('plane stress is a 2D reduction; a 3D solve has none')

    def in_plane_lame(self, reference_dim: int) -> tuple[FloatArray, FloatArray]:
        '''`(mu, lamb)` of the law a solve in `reference_dim` assembles.

        The 3D constants, except under plane stress, where the in-plane law is the
        3D one with `lamb` replaced by `2 lamb mu / (lamb + 2 mu)`: the free z
        direction relieves part of the volumetric coupling. Every in-plane formula
        (`D`, the Navier operator, the constrained stress) reads its constants here.
        '''
        self.require_dimension(reference_dim)
        mu, lamb = self.lame()
        if reference_dim == 2 and self.plane_stress:
            return mu, 2.0 * lamb * mu / (lamb + 2.0 * mu)
        return mu, lamb

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        '''`sigma_zz` from `(n_elements, ..., 2, 2)` in-plane strain tensors, one per
        point of each element; a per-element `E` broadcasts over the leading axis.

        Under plane strain the body is held fixed in z, and holding it costs
        `sigma_zz = lambda * tr(epsilon)`, a real stress that falls outside the three
        Voigt components a 2D assembly produces, so von Mises built from those alone
        would be wrong. The equivalent `nu(sigma_xx + sigma_yy)` is the same number;
        `tests/test_materials.py` pins both against the 3D law. Under plane stress it
        is zero by definition.
        '''
        trace = self._trace(strain, 'the strain')
        if self.plane_stress:
            return np.zeros_like(trace)
        _, lamb = self._per_element(self.lame(), trace.ndim)
        return lamb * trace

    def out_of_plane_strain(self, strain: FloatArray,
                            eigenstrain: FloatArray | None = None) -> FloatArray:
        '''`eps_zz` from `(n_elements, ..., 2, 2)` in-plane strain tensors, laid out
        as for `out_of_plane_stress`.

        Zero under plane strain. Under plane stress the plate is free in z and thins
        by Poisson's effect, `eps_zz = -lambda tr(epsilon) / (lambda + 2 mu)`; with an
        `eigenstrain` it also takes on that strain's z component and thins only
        against the in-plane mechanical strain.
        '''
        trace = self._trace(strain, 'the strain')
        if not self.plane_stress:
            return np.zeros_like(trace)
        mu, lamb = self._per_element(self.lame(), trace.ndim)
        ratio = lamb / (lamb + 2.0 * mu)
        if eigenstrain is None:
            return -ratio * trace
        eigenstrain = np.asarray(eigenstrain, dtype=float)
        mechanical = trace - np.einsum('...ii->...', eigenstrain[..., :2, :2])
        return eigenstrain[..., 2, 2] - ratio * mechanical

    @property
    def out_of_plane_ratio(self) -> float:
        '''`sigma_zz / (sigma_xx + sigma_yy)`: `nu` under plane strain, 0 under plane
        stress. What a formula in the in-plane Voigt stress needs to complete the
        tensor (the von Mises quantities of interest).'''
        return 0.0 if self.plane_stress else self.nu

    def constrained_stress(self, eigenstrain: FloatArray) -> FloatArray:
        '''The stress an eigenstrain produces in a body held at its reference shape,
        on `(n_elements, ..., 3, 3)` tensors, one per point of each element; a
        per-element `E` broadcasts over the leading axis.

        Under plane strain the body is held in every direction, so this is the 3D
        law `C : eps*` on the full tensor. The z component matters: the expansion
        denied in z pushes back on the plane through Poisson's effect and supplies a
        third of the in-plane thermal stress, which a 2D thermal strain fed through
        the 2D Hooke matrix misses (`(2 lambda + 2 mu) alpha dT` instead of
        `(3 lambda + 2 mu) alpha dT`). Under plane stress the plate is free in z, so
        its z component costs nothing: the in-plane block is the plane-stress law on
        the in-plane eigenstrain and the third row and column are zero.
        '''
        eigenstrain = np.asarray(eigenstrain, dtype=float)
        if eigenstrain.shape[-2:] != (3, 3):
            raise ValueError(
                f'an eigenstrain is a full (..., 3, 3) tensor in every dimension, got '
                f'shape {eigenstrain.shape}'
            )
        self._check_element_axis(eigenstrain.shape[:-2], 'the eigenstrain')
        if not self.plane_stress:
            mu, lamb = self._per_element(self.lame(), eigenstrain.ndim - 2)
            return isotropic_stress(mu, lamb, eigenstrain)
        mu, lamb = self._per_element(self.in_plane_lame(2), eigenstrain.ndim - 2)
        sigma = np.zeros_like(eigenstrain)
        sigma[..., :2, :2] = isotropic_stress(mu, lamb, eigenstrain[..., :2, :2])
        return sigma

    def _trace(self, tensors: FloatArray, what: str) -> FloatArray:
        '''The trace of `(n_elements, ..., d, d)` tensors, the element axis checked.'''
        trace = np.einsum('...ii->...', np.asarray(tensors, dtype=float))
        self._check_element_axis(trace.shape, what)
        return trace

    def _check_element_axis(self, leading: tuple[int, ...], what: str) -> None:
        E = np.asarray(self.E)
        if E.ndim and (not leading or leading[0] != len(E)):
            raise ValueError(
                f'per-element modulus has {len(E)} entries but {what} has leading '
                f'shape {leading}'
            )

    @staticmethod
    def _per_element(lame: tuple[FloatArray, FloatArray], ndim: int) -> tuple[FloatArray, FloatArray]:
        '''`lame` shaped to broadcast against an `ndim`-dimensional array of per-point
        values whose leading axis runs over the elements: scalars for a uniform
        modulus, `(n_elements, 1, ...)` for a per-element one.'''
        mu, lamb = lame
        if mu.ndim:
            shape = (len(mu),) + (1,) * (ndim - 1)
            mu, lamb = mu.reshape(shape), lamb.reshape(shape)
        return mu, lamb

    def constitutive_matrices(self, reference_dim: int, n_elements: int) -> FloatArray:
        '''(n_elements, s, s) Voigt D, one per element: the batched assembly path.

        A uniform modulus returns a broadcast view of the single matrix rather
        than n_elements copies of it, so the common case costs no extra memory
        and `np.einsum` still contracts it against a per-element B.
        '''
        P_mu, P_lamb = hooke_patterns(reference_dim)
        mu, lamb = self.in_plane_lame(reference_dim)
        if mu.ndim:
            if len(mu) != n_elements:
                raise ValueError(
                    f'per-element modulus has {len(mu)} entries but the mesh has '
                    f'{n_elements} elements'
                )
            return mu[:, None, None] * P_mu + lamb[:, None, None] * P_lamb
        return np.broadcast_to(mu * P_mu + lamb * P_lamb, (n_elements, *P_mu.shape))
