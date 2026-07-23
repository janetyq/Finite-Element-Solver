"""Hyperelastic strain energy densities: the material law for nonlinear FEM.

An energy density maps the deformation gradient F = I + grad u to a scalar
energy W and the derivative chain the Newton solver needs (dW/dF, d²W/dF²
decomposed through the strain tensor S).  Every quantity is batched over the
mesh: the primary interface is `evaluate`, which takes `(n_elements, d, d)`
gradients and returns a `StrainEnergyDerivatives` bundle with a leading element
axis on each array.

Two strain measures share one energy function W(S):

    Green-Lagrange  S = ½(FᵀF - I)   geometrically exact, frame indifferent
    small strain    ε = ½(F + Fᵀ) - I  linearisation, constant Hessian

They differ only in how S depends on F, so the derivative chain factorises as
dW/dF = dW/dS : dS/dF, and the parent–child split below mirrors that exactly:
`StVenantKirchhoff` owns the S-to-F map, `SmallStrain` overrides it with the
linear one.

Dimension-general: every tensor is built from `d = grad_u.shape[-1]`, not a
fixed DIM = 2. The constant tensors (d²S/dF² for Green-Lagrange, d²W/dS²) are
precomputed once at construction and broadcast over elements.
"""
import logging
from dataclasses import dataclass

import numpy as np

from fem.materials import Enu_to_Lame
from fem.typing import FloatArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrainEnergyDerivatives:
    """The derivative chain of a strain energy, batched over elements.

    Every array has a leading `(n_elements,)` axis.  `EnergyForm` contracts
    these against the shape-function-derived `dF_dx` to assemble the element
    energy, residual, and tangent in one vectorised pass.
    """
    W: FloatArray          # (n_el,)
    dW_dF: FloatArray      # (n_el, d, d)
    dW_dS: FloatArray      # (n_el, d, d)
    dS_dF: FloatArray      # (n_el, d, d, d, d)
    d2S_dF2: FloatArray    # (d, d, d, d, d, d)  -- constant, broadcast
    d2W_dS2: FloatArray    # (d, d, d, d)         -- constant, broadcast


class StVenantKirchhoff:
    """St Venant-Kirchhoff strain energy density.

    Green-Lagrange strain S = ½(FᵀF - I) paired with the isotropic energy
    W = ½λ tr(S)² + μ tr(SᵀS).

    Not linear elasticity, despite pairing the same W with the same Lame
    parameters. Green-Lagrange keeps the quadratic grad_uᵀ grad_u term that
    infinitesimal strain theory drops, which makes the model *geometrically*
    nonlinear: it is frame indifferent (a rigid rotation produces no strain,
    where small strain produces a spurious ~θ²/2 compression) at the cost of a
    Newton solve.

    Small strain is its linearisation, so the two agree to O(‖grad u‖²) --
    see tests/test_elasticity_models.py, which pins both halves of that
    statement.
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

    def _strain(self, F: FloatArray, eye: FloatArray) -> FloatArray:
        # Green-Lagrange. The quadratic term is what makes this nonlinear in u,
        # so Newton takes several iterations rather than the single step a
        # quadratic energy would need.
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

    # -- single-element interface for gradient checks -----------------------

    def calculate_S_from_F(self, F: FloatArray) -> FloatArray:
        """Single-element S(F), for the parked gradient checks."""
        d = F.shape[-1]
        return self._strain(F[None], np.eye(d))[0]

    def calculate_W_from_S(self, S: FloatArray) -> float:
        return float(self._energy(S[None])[0])

    def calculate_W_from_F(self, F: FloatArray) -> float:
        d = F.shape[-1]
        return self.calculate_W_from_S(self._strain(F[None], np.eye(d))[0])

    def calculate_dS_dF(self, F: FloatArray) -> FloatArray:
        d = F.shape[-1]
        return self._dS_dF(F[None], d)[0]

    def calculate_dW_dS(self, S: FloatArray) -> FloatArray:
        d = S.shape[-1]
        return self._dW_dS(S[None], np.eye(d))[0]

    def calculate_dW_dF(self, F: FloatArray) -> FloatArray:
        d = F.shape[-1]
        eye = np.eye(d)
        S = self._strain(F[None], eye)
        return np.einsum('ij,ijmn->mn', self._dW_dS(S, eye)[0], self._dS_dF(F[None], d)[0])

    def calculate_d2S_dF2(self, F: FloatArray) -> FloatArray:
        return self._d2S_dF2(F.shape[-1])

    def calculate_d2W_dS2(self, S: FloatArray) -> FloatArray:
        return self._d2W_dS2(S.shape[-1])

    def check_gradients(self) -> None:
        from fem.numerics import check_gradient
        check_gradient(self.calculate_S_from_F, self.calculate_dS_dF, (2, 2))
        check_gradient(self.calculate_W_from_S, self.calculate_dW_dS, (2, 2))
        check_gradient(self.calculate_W_from_F, self.calculate_dW_dF, (2, 2))
        logger.info("Gradient checks completed")


class SmallStrain(StVenantKirchhoff):
    """Infinitesimal-strain elasticity: St-VK with ε = ½(F + Fᵀ) - I.

    The linearisation of Green-Lagrange.  The strain is affine in F, so dS/dF
    is constant, d²S/dF² vanishes, the energy is quadratic in u, and Newton
    converges in one step.  This is the same physics `Solver` solves by direct
    assembly — its value is as the independent cross-check and as the
    small-strain member of the strain-measure axis.
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
