"""Hyperelastic strain energy densities: the material law for nonlinear FEM.

An energy density maps the deformation gradient F = I + grad u to a scalar energy W
and the two derivatives the Newton solver contracts: the first Piola-Kirchhoff stress
P = dW/dF and the material tangent A = d²W/dF². Every quantity is batched over the
mesh: the primary interface is `evaluate`, which takes `(n_elements, d, d)` gradients
and returns a `StrainEnergyDerivatives` bundle with a leading element axis on each
array.

The interface is `F`-based on purpose: it is what `EnergyForm` actually contracts,
and it is the one shape every hyperelastic law shares. A model built on a strain
measure (St-Venant-Kirchhoff below) folds its own chain down to `P` and `A`
internally; a model built on the invariants of `C = FᵀF` (Neo-Hookean) writes them
directly. Both answer the same three arrays, so `EnergyForm` needs to know nothing
about how the density is parameterised.

Dimension-general: every tensor is built from `d = grad_u.shape[-1]`, so the same
class serves 2D (plane strain) and 3D.

The gradient is in the standard orientation `F[c, i] = ∂u_c/∂x_i`
(`ElementGeometry.gradients`), so `P` and `A` come out in that orientation too and
`strain` returns the textbook `½(FᵀF - I)` with no transpose. A 2D density models a
3D body held fixed in z, so `S_zz = 0` and the restraint develops an out-of-plane
stress; `out_of_plane_stress` supplies the one component a 2D assembly omits, which
`EnergyForm` needs for von Mises.
"""
from dataclasses import dataclass

import numpy as np

from fem.materials import Enu_to_Lame
from fem.typing import FloatArray


@dataclass(frozen=True)
class StrainEnergyDerivatives:
    """A strain energy and its first two derivatives in F, batched over elements.

    Every array has a leading `(n_elements,)` axis. `EnergyForm` contracts `P` and
    `A` against the shape-function-derived `dF_dx` to assemble the element energy,
    residual, and tangent in one vectorised pass.
    """
    W: FloatArray    # (n_el,)
    P: FloatArray    # (n_el, d, d), first Piola-Kirchhoff stress dW/dF
    A: FloatArray    # (n_el, d, d, d, d), material tangent d²W/dF²


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

    `P` and `A` are assembled internally from this model's strain chain
    (`dW/dS`, `dS/dF`, and their second derivatives); a consumer sees only the
    F-derivatives every density shares.
    """

    # Polynomial degree of W in the displacement gradient: the Green strain is quadratic
    # in grad_u and W is quadratic in the strain, so W is quartic. This sets the rule the
    # energy is integrated at, above the linear stiffness's default on P2 and higher (see
    # FunctionSpace._geometry_for).
    energy_degree = 4

    def __init__(self, E: float, nu: float) -> None:
        self.mu, self.lamb = Enu_to_Lame(E, nu)

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        """Evaluate W, P = dW/dF, and A = d²W/dF² at `(n_elements, d, d)` gradients."""
        d = grad_u.shape[-1]
        eye = np.eye(d)
        F = eye + grad_u
        S = self._strain(F, eye)
        dW_dS = self._dW_dS(S, eye)
        dS_dF = self._dS_dF(F, d)
        d2S_dF2 = self._d2S_dF2(d)
        d2W_dS2 = self._d2W_dS2(d)

        # dW/dF = dW/dS : dS/dF, and d²W/dF² = dW/dS : d²S/dF² + dS/dF : d²W/dS² : dS/dF,
        # the same chain rule the tangent used to expand at assembly time, folded down to
        # F-derivatives here so `EnergyForm` contracts one material tangent.
        P = np.einsum('eij,eijmn->emn', dW_dS, dS_dF)
        A = (np.einsum('eij,ijmnkq->emnkq', dW_dS, d2S_dF2)
             + np.einsum('eijmn,ijab,eabkq->emnkq', dS_dF, d2W_dS2, dS_dF))
        return StrainEnergyDerivatives(W=self._energy(S), P=P, A=A)

    # -- strain measure (overridden by SmallStrain) -------------------------

    def strain(self, grad_u: FloatArray) -> FloatArray:
        """This density's own strain measure, for reporting rather than solving.

        Subclasses override `_strain`, so both strain measures answer through this.
        """
        d = grad_u.shape[-1]
        eye = np.eye(d)
        return self._strain(eye + grad_u, eye)

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        """The stress in the restrained z direction, which a 2D solve omits.

        This is the second Piola-Kirchhoff component `S_zz = λ tr(S)`, which the
        plane-strain restraint `S_zz = 0` develops; `EnergyForm` converts it to
        Cauchy, which here is just a division by J since the material does not move
        in z.

        `'eii->e'` is a batched trace: sum each element's diagonal.
        """
        return self.lamb * np.einsum('eii->e', strain)

    def _strain(self, F: FloatArray, eye: FloatArray) -> FloatArray:
        # Green-Lagrange. The quadratic term makes this nonlinear in u, so Newton
        # takes several iterations rather than the single step a quadratic energy
        # would need.
        return 0.5 * (np.einsum('eji,ejk->eik', F, F) - eye)

    def _dS_dF(self, F: FloatArray, d: int) -> FloatArray:
        # dS_dF[e,i,j,m,n] = ∂S_ij/∂F_mn = ½(F[e,m,i]δ(j,n) + F[e,m,j]δ(i,n))
        eye = np.eye(d)
        return 0.5 * (
            np.einsum('emi,jn->eijmn', F, eye) +
            np.einsum('emj,in->eijmn', F, eye)
        )

    def _d2S_dF2(self, d: int) -> FloatArray:
        # d²S/dF²[i,j,m,n,k,q] = ∂²S_ij/∂F_mn∂F_kq
        #                      = ½(δ(j,n)δ(k,m)δ(i,q) + δ(i,n)δ(k,m)δ(j,q))
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

    # The small strain is affine in grad_u and W quadratic in it, so W is quadratic:
    # the default P2 rule already integrates it exactly, unlike the quartic Green-Lagrange.
    energy_degree = 2

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
    """Compressible Neo-Hookean strain energy density.

    W = ½μ (I₁ - d) - μ ln J + ½λ ln² J,   I₁ = tr(FᵀF),   J = det F.

    Written in the invariants of C = FᵀF rather than a strain tensor, so it has no
    constant Hessian to factor through a strain measure the way St-VK does; it writes
    P and A in F directly. It agrees with St-Venant-Kirchhoff (and small strain) to
    O(‖grad u‖²), and stays frame indifferent and stable in compression where the
    polynomial St-VK energy loses ellipticity.
    """

    # Non-polynomial (it carries a log J term), so no rule is exact; 4 is a reasonable
    # integration order, matching St-VK.
    energy_degree = 4

    def __init__(self, E: float, nu: float) -> None:
        self.mu, self.lamb = Enu_to_Lame(E, nu)

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        """Evaluate W, P = dW/dF, and A = d²W/dF² at `(n_elements, d, d)` gradients."""
        d = grad_u.shape[-1]
        eye = np.eye(d)
        F = eye + grad_u
        J = np.linalg.det(F)
        F_invT = np.swapaxes(np.linalg.inv(F), -2, -1)   # F⁻ᵀ[e,i,j] = (F⁻¹)_ji

        # J <= 0 is an inverted (non-physical) element, which the log cannot take. Its
        # energy is infinite, which is exactly what makes a line search reject the step;
        # `where` supplies the +inf and the guarded log keeps the invalid value quiet.
        inverted = J <= 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_J = np.log(J)
        I1 = np.einsum('eij,eij->e', F, F)
        W = 0.5 * self.mu * (I1 - d) - self.mu * ln_J + 0.5 * self.lamb * ln_J ** 2
        W = np.where(inverted, np.inf, W)

        # P = μ F + (λ ln J - μ) F⁻ᵀ.
        P = self.mu * F + (self.lamb * ln_J - self.mu)[:, None, None] * F_invT

        # A_{ij,mn} = μ δ_im δ_jn + λ F⁻ᵀ_ij F⁻ᵀ_mn + (μ - λ ln J) F⁻ᵀ_in F⁻ᵀ_mj.
        A = (self.mu * np.einsum('im,jn->ijmn', eye, eye)[None]
             + self.lamb * np.einsum('eij,emn->eijmn', F_invT, F_invT)
             + (self.mu - self.lamb * ln_J)[:, None, None, None, None]
             * np.einsum('ein,emj->eijmn', F_invT, F_invT))
        return StrainEnergyDerivatives(W=W, P=P, A=A)

    def strain(self, grad_u: FloatArray) -> FloatArray:
        """Green-Lagrange strain, the finite measure reported for a finite-strain law."""
        d = grad_u.shape[-1]
        eye = np.eye(d)
        F = eye + grad_u
        return 0.5 * (np.einsum('eji,ejk->eik', F, F) - eye)

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        """The restrained-z second Piola-Kirchhoff stress a 2D (plane-strain) solve omits.

        In plane strain F_zz = 1, so the 3D law gives `S_zz = λ ln J`; the in-plane
        `J` is recovered from `det C = det(2E + I)`. `EnergyForm` divides by J for the
        Cauchy component von Mises needs.
        """
        d = strain.shape[-1]
        C = 2.0 * strain + np.eye(d)
        ln_J = 0.5 * np.log(np.linalg.det(C))
        return self.lamb * ln_J
