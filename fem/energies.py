"""Hyperelastic strain energy densities: the material law for nonlinear FEM.

An energy density maps the deformation gradient F = I + grad u to a scalar energy W
and the derivative chain the Newton solver needs (dW/dF, d²W/dF² decomposed through
the strain tensor S). Every quantity is batched over the mesh: the primary interface
is `evaluate`, which takes `(n_elements, d, d)` gradients and returns a
`StrainEnergyDerivatives` bundle with a leading element axis on each array.

Two strain measures share one energy function W(S):

    Green-Lagrange  S = ½(FᵀF - I)   geometrically exact, frame indifferent
    small strain    ε = ½(F + Fᵀ) - I  linearisation, constant Hessian

They differ only in how S depends on F, so the derivative chain factorises as
dW/dF = dW/dS : dS/dF. The parent-child split below mirrors that: `StVenantKirchhoff`
owns the S-to-F map, `SmallStrain` overrides it with the linear one.

Dimension-general: every tensor is built from `d = grad_u.shape[-1]`, not a fixed
DIM = 2. The constant tensors (d²S/dF² for Green-Lagrange, d²W/dS²) are precomputed
once at construction and broadcast over elements.


Solving versus reporting
------------------------

`evaluate` feeds the Newton solve; `strain` and `out_of_plane_stress` feed
post-processing. Two conventions differ between the two jobs.

**Gradient orientation.** `ElementGeometry.gradients` puts `du_c/dx_i` at entry
`[i, c]`, the transpose of the usual convention, so the `F` built here is
`F_standardᵀ` and `evaluate`'s whole chain works in that orientation. W is blind to
it, using S only through `tr S` and `tr(SᵀS)`, which `½(FFᵀ - I)` and `½(FᵀF - I)`
share. A reported tensor is not blind to it, so `strain` transposes back and returns
the textbook `½(FᵀF - I)`.

**Plane strain.** A 2D density models a 3D body held fixed in z, so `S_zz = 0`.
That is why a stress appears there: material squeezed in x and y pushes outward
in z, the restraint pushes back, and the law gives `sigma_zz = lambda * tr(S)`.
A 2D assembly produces only the three in-plane Voigt components, so von Mises
built from those alone is missing this one. `out_of_plane_stress` supplies it.
"""
from dataclasses import dataclass

import numpy as np

from fem.materials import Enu_to_Lame
from fem.typing import FloatArray


@dataclass(frozen=True)
class StrainEnergyDerivatives:
    """The derivative chain of a strain energy, batched over elements.

    Every array has a leading `(n_elements,)` axis. `EnergyForm` contracts these
    against the shape-function-derived `dF_dx` to assemble the element energy,
    residual, and tangent in one vectorised pass.
    """
    W: FloatArray          # (n_el,)
    dW_dF: FloatArray      # (n_el, d, d)
    dW_dS: FloatArray      # (n_el, d, d)
    dS_dF: FloatArray      # (n_el, d, d, d, d)
    d2S_dF2: FloatArray    # (d, d, d, d, d, d), constant, broadcast
    d2W_dS2: FloatArray    # (d, d, d, d), constant, broadcast


class StVenantKirchhoff:
    """St Venant-Kirchhoff strain energy density.

    Green-Lagrange strain S = ½(FᵀF - I) paired with the isotropic energy
    W = ½λ tr(S)² + μ tr(SᵀS).

    Not linear elasticity, despite pairing the same W with the same Lame
    parameters. Green-Lagrange keeps the quadratic grad_uᵀ grad_u term that
    infinitesimal strain theory drops, which makes the model geometrically
    nonlinear: it is frame indifferent (a rigid rotation produces no strain,
    where small strain produces a spurious ~θ²/2 compression) at the cost of a
    Newton solve.

    Small strain is its linearisation, so the two agree to O(‖grad u‖²); see
    tests/test_elasticity_models.py, which pins both halves of that statement.
    """

    def __init__(self, E: float, nu: float) -> None:
        self.mu, self.lamb = Enu_to_Lame(E, nu)

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        """Evaluate the full derivative chain at `(n_elements, d, d)` gradients."""
        d = grad_u.shape[-1]
        eye = np.eye(d)
        F = eye + grad_u
        S = self._strain(F, eye)
        dW_dS = self._dW_dS(S, eye)
        dS_dF = self._dS_dF(F, d)
        return StrainEnergyDerivatives(
            W=self._energy(S),
            dW_dF=np.einsum('eij,eijmn->emn', dW_dS, dS_dF),
            dW_dS=dW_dS,
            dS_dF=dS_dF,
            d2S_dF2=self._d2S_dF2(d),
            d2W_dS2=self._d2W_dS2(d),
        )

    # -- strain measure (overridden by SmallStrain) -------------------------

    def strain(self, grad_u: FloatArray) -> FloatArray:
        """This density's own strain measure, for reporting rather than solving.

        Subclasses override `_strain`, so both strain measures answer through
        this. F is built in the standard orientation; see "Gradient orientation".
        """
        d = grad_u.shape[-1]
        eye = np.eye(d)
        return self._strain(eye + np.swapaxes(grad_u, -2, -1), eye)

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        """The stress in the restrained z direction, which a 2D solve omits.

        See "Plane strain" above. This is the second Piola-Kirchhoff component;
        `EnergyForm` converts it to Cauchy, which here is just a division by J
        since the material does not move in z.

        `'eii->e'` is a batched trace: sum each element's diagonal.
        """
        return self.lamb * np.einsum('eii->e', strain)

    def _strain(self, F: FloatArray, eye: FloatArray) -> FloatArray:
        # Green-Lagrange. The quadratic term makes this nonlinear in u, so Newton
        # takes several iterations rather than the single step a quadratic energy
        # would need.
        return 0.5 * (np.einsum('eji,ejk->eik', F, F) - eye)

    def _dS_dF(self, F: FloatArray, d: int) -> FloatArray:
        # dS_dF[e,i,j,m,n] = ½(F[e,m,i]δ(j,n) + F[e,m,j]δ(i,n))
        eye = np.eye(d)
        return 0.5 * (
            np.einsum('emi,jn->eijmn', F, eye) +
            np.einsum('emj,in->eijmn', F, eye)
        )

    def _d2S_dF2(self, d: int) -> FloatArray:
        # d²S/dF²[i,j,m,n,k,q] = ½(δ(j,n)δ(k,m)δ(i,q) + δ(i,n)δ(k,m)δ(j,q))
        eye = np.eye(d)
        return 0.5 * (
            np.einsum('jn,km,iq->ijmnkq', eye, eye, eye) +
            np.einsum('in,km,jq->ijmnkq', eye, eye, eye)
        )

    # -- energy function (shared by both strain measures) -------------------

    def _energy(self, S: FloatArray) -> FloatArray:
        tr = np.einsum('eii->e', S)
        tr_STS = np.einsum('eij,eij->e', S, S)
        return 0.5 * (self.lamb * tr ** 2 + 2 * self.mu * tr_STS)

    def _dW_dS(self, S: FloatArray, eye: FloatArray) -> FloatArray:
        tr = np.einsum('eii->e', S)
        return self.lamb * tr[:, None, None] * eye + 2 * self.mu * S

    def _d2W_dS2(self, d: int) -> FloatArray:
        eye = np.eye(d)
        return (self.lamb * np.einsum('ij,mn->ijmn', eye, eye)
                + 2 * self.mu * np.einsum('im,jn->ijmn', eye, eye))

    # -- single-element energy, for the test cross-check --------------------

    def calculate_W_from_S(self, S: FloatArray) -> float:
        """Single-element W(S), pinning `Material`'s D as d2W/de2 in the tests."""
        return float(self._energy(S[None])[0])


class SmallStrain(StVenantKirchhoff):
    """Infinitesimal-strain elasticity: St-VK with ε = ½(F + Fᵀ) - I.

    The linearisation of Green-Lagrange. The strain is affine in F, so dS/dF is
    constant, d²S/dF² vanishes, the energy is quadratic in u, and Newton converges
    in one step. This is the same physics `Solver` solves by direct assembly; its
    value is as the independent cross-check and as the small-strain member of the
    strain-measure axis.
    """

    def _strain(self, F: FloatArray, eye: FloatArray) -> FloatArray:
        return 0.5 * (F + np.swapaxes(F, -2, -1)) - eye

    def _dS_dF(self, F: FloatArray, d: int) -> FloatArray:
        n = F.shape[0]
        eye = np.eye(d)
        single = 0.5 * (np.einsum('im,jn->ijmn', eye, eye)
                        + np.einsum('jm,in->ijmn', eye, eye))
        return np.broadcast_to(single, (n, d, d, d, d))

    def _d2S_dF2(self, d: int) -> FloatArray:
        return np.zeros((d, d, d, d, d, d))


class NeohookeanEnergyDensity:
    def __init__(self, E: float, nu: float) -> None:
        self.mu, self.lamb = Enu_to_Lame(E, nu)

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        raise NotImplementedError(
            "NeohookeanEnergyDensity is not implemented yet; "
            "use StVenantKirchhoff for now."
        )
