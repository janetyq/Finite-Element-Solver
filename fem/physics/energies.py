"""Hyperelastic strain energy densities: the material law for nonlinear FEM.

An energy density maps the deformation gradient F = I + grad u to a scalar energy W
and the two derivatives the Newton solver contracts: the first Piola-Kirchhoff stress
P = dW/dF and the material tangent A = d²W/dF². Every quantity is batched over the
mesh: `evaluate` takes `(n_elements, d, d)` gradients and returns a
`StrainEnergyDerivatives` bundle with a leading element axis on each array.

The answers come in three tiers, because their consumers do: a line search needs only
`energy`, a residual needs `stress` (W and P), and only the tangent needs `evaluate`
(W, P, and A), which is most of the cost. `HyperelasticDensity` is the base every
density here extends: it defaults the cheaper tiers to slices of `evaluate`, so a
density that writes only the full chain is complete, and each density below overrides
them with the direct computation.

The interface is `F`-based on purpose: it is what `EnergyForm` actually contracts,
and it is the one shape every hyperelastic law shares. A model built on a strain
measure (St-Venant-Kirchhoff below) writes its `P` and `A` in closed form from that
measure's derivatives; a model built on the invariants of `C = FᵀF` (Neo-Hookean)
writes them directly. Both answer the same arrays, so `EnergyForm` needs to know
nothing about how the density is parameterised.

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

from fem.physics.materials import Enu_to_Lame
from fem.typing import FloatArray


@dataclass(frozen=True)
class StrainEnergyDerivatives:
    """A strain energy and its first two derivatives in F, batched over elements.

    `W` is the stored energy, `P = dW/dF` the stress, and `A = d²W/dF²` the stiffness:
    the three things a Newton solve needs from a material. Every array has a leading
    `(n_elements,)` axis; `EnergyForm` contracts `P` and `A` against the
    shape-function-derived `dF_dx` to assemble the element energy, residual, and tangent
    in one vectorised pass.
    """
    W: FloatArray    # (n_el,), stored strain energy
    P: FloatArray    # (n_el, d, d), stress: first Piola-Kirchhoff, dW/dF
    A: FloatArray    # (n_el, d, d, d, d), stiffness (material tangent): d²W/dF²


class HyperelasticDensity:
    """The three tiers of an energy density, each defaulting to the one above it.

    A subclass writes `evaluate`, the full chain `(W, P, A)`; `energy` and `stress`
    then work by slicing it. A density whose energy or stress is much cheaper than its
    tangent (every density in this module) overrides them, so a residual or a
    line-search merit never builds the tangent it will not read.
    """

    energy_degree: int

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        """W, P = dW/dF, and A = d²W/dF² at `(n_elements, d, d)` gradients."""
        raise NotImplementedError

    def energy(self, grad_u: FloatArray) -> FloatArray:
        """`(n_elements,)` stored energy W at the gradients."""
        return self.evaluate(grad_u).W

    def stress(self, grad_u: FloatArray) -> tuple[FloatArray, FloatArray]:
        """`(W, P)` at the gradients: the energy and the first Piola-Kirchhoff stress."""
        derivatives = self.evaluate(grad_u)
        return derivatives.W, derivatives.P


class StVenantKirchhoff(HyperelasticDensity):
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

    `P` and `A` are the chain rule through the strain, written out in closed form:
    with `S' = dW/dS` (symmetric) and `C = d²W/dS²`,

        P = F S'                                      (dS/dF is ½(F δ + F δ))
        A[m,n,k,q] = δ_mk S'[n,q] + F[m,a] C[a,n,b,q] F[k,b]

    the first term of `A` the geometric stiffness (the strain's own curvature in F),
    the second the material stiffness carried through the deformation. Neither needs
    the `(d, d, d, d)`-per-element `dS/dF` tensor the chain rule would materialise.
    """

    # Polynomial degree of W in the displacement gradient: the Green strain is quadratic
    # in grad_u and W is quadratic in the strain, so W is quartic. This sets the rule the
    # energy is integrated at, above the linear stiffness's default on P2 and higher (see
    # FunctionSpace._term_geometry).
    energy_degree = 4

    def __init__(self, E: float, nu: float) -> None:
        self.mu, self.lamb = Enu_to_Lame(E, nu)

    def _kinematics(self, grad_u: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """`(F, S, eye)` at the gradients: the deformation gradient, this density's strain
        measure, and the identity of the right dimension."""
        eye = np.eye(grad_u.shape[-1])
        F = eye + grad_u
        return F, self._strain(F, eye), eye

    def energy(self, grad_u: FloatArray) -> FloatArray:
        _, S, _ = self._kinematics(grad_u)
        return self._energy(S)

    def stress(self, grad_u: FloatArray) -> tuple[FloatArray, FloatArray]:
        F, S, eye = self._kinematics(grad_u)
        return self._energy(S), self._stress(F, self._dW_dS(S, eye))

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        """Evaluate W, P = dW/dF, and A = d²W/dF² at `(n_elements, d, d)` gradients."""
        F, S, eye = self._kinematics(grad_u)
        dW_dS = self._dW_dS(S, eye)
        return StrainEnergyDerivatives(
            W=self._energy(S), P=self._stress(F, dW_dS), A=self._tangent(F, dW_dS, eye))

    # -- strain measure and its derivatives in F (overridden by SmallStrain) -------

    def strain(self, grad_u: FloatArray) -> FloatArray:
        """This density's own strain measure, for reporting rather than solving.

        Subclasses override `_strain`, so both strain measures answer through this.
        """
        _, S, _ = self._kinematics(grad_u)
        return S

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
        return 0.5 * (np.swapaxes(F, -2, -1) @ F - eye)

    def _stress(self, F: FloatArray, dW_dS: FloatArray) -> FloatArray:
        # P_mn = S'_ij dS_ij/dF_mn with dS_ij/dF_mn = ½(F_mi δ_jn + F_mj δ_in) and S'
        # symmetric collapses to P = F S', a batched matrix product.
        return F @ dW_dS

    def _tangent(self, F: FloatArray, dW_dS: FloatArray, eye: FloatArray) -> FloatArray:
        # A_mnkq = dP_mn/dF_kq = δ_mk S'_nq + F_ma (dS'_an/dS_bc)(dS_bc/dF_kq)
        #        = δ_mk S'_nq + F_ma C_anbq F_kb,
        # with C the symmetrised d²W/dS² (see `_d2W_dS2`): the second factor of the
        # chain is symmetric in (b, c), so only C's (b, q)-symmetric part survives.
        # The outer product δ_mk S'_nq is `outer(eye, S')` with its axes put in A's order.
        geometric = np.multiply.outer(eye, dW_dS).transpose(2, 0, 3, 1, 4)
        material = np.einsum('ema,anbq,ekb->emnkq', F, self._d2W_dS2(eye), F, optimize=True)
        return geometric + material

    # -- energy function (shared by both strain measures) -------------------

    def _energy(self, S: FloatArray) -> FloatArray:
        tr = np.einsum('eii->e', S)
        tr_STS = np.einsum('eij,eij->e', S, S)
        return 0.5 * (self.lamb * tr ** 2 + 2 * self.mu * tr_STS)

    def _dW_dS(self, S: FloatArray, eye: FloatArray) -> FloatArray:
        tr = np.einsum('eii->e', S)
        return self.lamb * tr[:, None, None] * eye + 2 * self.mu * S

    def _d2W_dS2(self, eye: FloatArray) -> FloatArray:
        """`C[a,n,b,q] = dS'_an / dS_bq` on symmetric strains: `λ δ_an δ_bq + μ (δ_ab δ_nq
        + δ_aq δ_nb)`, the isotropic elasticity tensor, symmetric in each index pair and
        under their exchange. Constant, so it carries no element axis."""
        return (self.lamb * np.einsum('an,bq->anbq', eye, eye)
                + self.mu * (np.einsum('ab,nq->anbq', eye, eye)
                             + np.einsum('aq,nb->anbq', eye, eye)))

    # -- single-element energy, for the test cross-check --------------------

    def calculate_W_from_S(self, S: FloatArray) -> float:
        """Single-element W(S), pinning `Material`'s D as d2W/de2 in the tests."""
        return float(self._energy(S[None])[0])


class SmallStrain(StVenantKirchhoff):
    """Infinitesimal-strain elasticity: St-VK with ε = ½(F + Fᵀ) - I.

    The linearisation of Green-Lagrange. The strain is affine in F, so `P` is `dW/dε`
    itself and `A` is the constant `d²W/dε²`: the energy is quadratic in u and Newton
    converges in one step. This is the same physics `LinearElastic` solves by direct
    assembly; its value is as the independent cross-check and as the small-strain
    member of the strain-measure axis.
    """

    # The small strain is affine in grad_u and W quadratic in it, so W is quadratic:
    # the default P2 rule already integrates it exactly, unlike the quartic Green-Lagrange.
    energy_degree = 2

    def _strain(self, F: FloatArray, eye: FloatArray) -> FloatArray:
        return 0.5 * (F + np.swapaxes(F, -2, -1)) - eye

    def _stress(self, F: FloatArray, dW_dS: FloatArray) -> FloatArray:
        # dε_ij/dF_mn = ½(δ_im δ_jn + δ_jm δ_in), so P is the symmetric S' itself.
        return dW_dS

    def _tangent(self, F: FloatArray, dW_dS: FloatArray, eye: FloatArray) -> FloatArray:
        # Affine strain: no geometric term, and the material term is C itself.
        d = eye.shape[0]
        return np.broadcast_to(self._d2W_dS2(eye), (F.shape[0], d, d, d, d))


class NeohookeanEnergyDensity(HyperelasticDensity):
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

    def _invariants(self, grad_u: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """`(F, ln J, W, eye)` at the gradients, with `W = +inf` on an inverted element.

        J <= 0 is an inverted (non-physical) element, which the log cannot take. Its
        energy is infinite, which is exactly what makes a line search reject the step;
        `where` supplies the +inf and the guarded log keeps the invalid value quiet.
        """
        d = grad_u.shape[-1]
        eye = np.eye(d)
        F = eye + grad_u
        J = np.linalg.det(F)
        inverted = J <= 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            ln_J = np.log(J)
        I1 = np.einsum('eij,eij->e', F, F)
        W = 0.5 * self.mu * (I1 - d) - self.mu * ln_J + 0.5 * self.lamb * ln_J ** 2
        return F, ln_J, np.where(inverted, np.inf, W), eye

    @staticmethod
    def _inverse_transpose(F: FloatArray) -> FloatArray:
        '''F⁻ᵀ[e,i,j] = (F⁻¹)_ji, batched.'''
        return np.asarray(np.swapaxes(np.linalg.inv(F), -2, -1), dtype=np.float64)

    def _stress(self, F: FloatArray, ln_J: FloatArray, F_invT: FloatArray) -> FloatArray:
        # P = μ F + (λ ln J - μ) F⁻ᵀ.
        return self.mu * F + (self.lamb * ln_J - self.mu)[:, None, None] * F_invT

    def energy(self, grad_u: FloatArray) -> FloatArray:
        return self._invariants(grad_u)[2]

    def stress(self, grad_u: FloatArray) -> tuple[FloatArray, FloatArray]:
        F, ln_J, W, _ = self._invariants(grad_u)
        return W, self._stress(F, ln_J, self._inverse_transpose(F))

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        """Evaluate W, P = dW/dF, and A = d²W/dF² at `(n_elements, d, d)` gradients."""
        F, ln_J, W, eye = self._invariants(grad_u)
        F_invT = self._inverse_transpose(F)
        P = self._stress(F, ln_J, F_invT)

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
