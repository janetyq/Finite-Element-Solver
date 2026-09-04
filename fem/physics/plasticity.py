"""Deformation-theory plasticity: a metal's stress-strain curve as an energy density.

`RambergOsgood` is J2 (von Mises) deformation-theory plasticity under the
Ramberg-Osgood hardening curve: the stress follows Hooke's law below yield and the
power-law hardening curve beyond it, with the plastic part of the strain aligned with
the deviatoric stress. Deformation theory makes stress a function of the *current*
strain alone (Hencky), so the model is history-free and, being derived from a stored
energy, solves through the standard `EnergyForm` Newton path with no state to carry.

That is also its limit: it agrees with incremental (flow-theory) plasticity exactly
under monotonic, proportional loading, and is wrong under unloading, which it retraces
elastically down the same curve. Load once, read the redistributed stress and the
plastic zone; do not cycle it.

The uniaxial Ramberg-Osgood curve is

    eps = sigma/E + offset * (sigma_y/E) * (sigma/sigma_y)^n,

smooth everywhere (no yield corner for Newton to chatter on), with `offset` the
plastic strain at `sigma_y` in units of `sigma_y/E` (the classic fit uses 3/7, making
`sigma_y` the 0.2%-offset yield strength for typical E/sigma_y). The J2 generalization
splits the strain into volumetric and deviatoric parts: the volumetric response stays
elastic (`tr sigma = 3K tr eps`), and the deviatoric stress is coaxial with the
deviatoric strain at the magnitude the scalar curve dictates,

    e_eq = sigma_eq/(3 mu) + eps_p(sigma_eq),   eps_p(s) = offset*(sigma_y/E)*(s/sigma_y)^n,

with `sigma_eq = sqrt(3/2 s:s)` the von Mises stress and `e_eq = sqrt(2/3 e:e)` the
equivalent deviatoric strain. The curve is written stress-explicit, so the
strain-driven evaluation inverts it: one scalar Newton solve per point, vectorized
over the mesh, monotone from the elastic overestimate (the equation is convex and
increasing in `sigma_eq`).

Everything is computed on full 3x3 tensors internally: under plane strain the plastic
flow is volume-preserving, so the out-of-plane deviatoric strain `e_zz = -tr(eps)/3`
carries stress even though `eps_zz = 0`, and dropping it would put the return onto the
wrong yield surface. The in-plane blocks are what `evaluate` hands back, the same
plane-strain reduction the other densities make.
"""
from typing import NamedTuple

import numpy as np

from fem.physics.energies import HyperelasticDensity, StrainEnergyDerivatives
from fem.physics.materials import Enu_to_Lame
from fem.typing import FloatArray


class _DeviatoricState(NamedTuple):
    '''The J2 state at a batch of 3D strains: what stress and tangent are built from.'''
    trace: FloatArray       # (n,) tr(eps)
    deviator: FloatArray    # (n, 3, 3) e = dev(eps)
    e_eq: FloatArray        # (n,) equivalent deviatoric strain sqrt(2/3 e:e)
    sigma_eq: FloatArray    # (n,) von Mises stress solving the scalar curve
    secant: FloatArray      # (n,) c = 2 sigma_eq / (3 e_eq): s = c e (2 mu at e = 0)


class RambergOsgood(HyperelasticDensity):
    """J2 deformation-theory plasticity with Ramberg-Osgood hardening: an `EnergyDensity`.

    Hyperelastic in the small strain `eps = sym(grad u)`: `W` is the strain energy the
    stress-strain curve integrates to, so `P = dW/dF` is the stress and `A = d^2W/dF^2`
    the (secant-consistent) tangent, and Newton on the assembled residual has an energy
    to line-search on. Valid for monotonic, proportional loading (see the module
    docstring).

    `yield_stress` sets where the curve bends, `hardening_exponent` (n >= 1) how
    sharply: n = 1 is a second linear branch and large n approaches elastic-perfectly-
    plastic. `offset` is the plastic strain at `yield_stress` in units of
    `yield_stress/E`; the classic Ramberg-Osgood fit is 3/7.
    """

    # Non-polynomial in the displacement gradient (a rational power of the strain), so
    # no rule is exact; 4 matches the other non-polynomial density (Neo-Hookean).
    energy_degree = 4

    def __init__(self, E: float, nu: float, yield_stress: float,
                 hardening_exponent: float, offset: float = 3.0 / 7.0) -> None:
        if yield_stress <= 0:
            raise ValueError(f'yield_stress must be positive, got {yield_stress}')
        if hardening_exponent < 1:
            raise ValueError(
                f'hardening_exponent must be at least 1, got {hardening_exponent}')
        if offset <= 0:
            raise ValueError(f'offset must be positive, got {offset}')
        self.E = float(E)
        self.nu = float(nu)
        self.yield_stress = float(yield_stress)
        self.hardening_exponent = float(hardening_exponent)
        self.offset = float(offset)
        self.mu, self.lamb = Enu_to_Lame(E, nu)
        self.bulk = self.lamb + 2.0 * self.mu / 3.0

    # -- the scalar curve ----------------------------------------------------

    def plastic_strain(self, sigma_eq: FloatArray) -> FloatArray:
        '''Equivalent plastic strain at von Mises stress `sigma_eq` (elementwise).'''
        E, sy = self.E, self.yield_stress
        return self.offset * (sy / E) * (np.asarray(sigma_eq) / sy) ** self.hardening_exponent

    def _plastic_slope(self, sigma_eq: FloatArray) -> FloatArray:
        '''d(plastic strain)/d(sigma_eq).'''
        E, sy, n = self.E, self.yield_stress, self.hardening_exponent
        return (self.offset * n / E) * (sigma_eq / sy) ** (n - 1.0)

    def equivalent_stress(self, e_eq: FloatArray) -> FloatArray:
        '''Invert the curve: the von Mises stress at equivalent deviatoric strain `e_eq`.

        Solves `sigma/(3 mu) + eps_p(sigma) = e_eq` by Newton, elementwise. The left
        side is convex and increasing, so from a seed at or above the root the
        iteration decreases monotonically onto it; the seed is the smaller of the two
        upper bounds (all-elastic and all-plastic), which also keeps the power term
        in a safe floating range for a sharp exponent.
        '''
        e_eq = np.asarray(e_eq, dtype=float)
        mu, sy, n = self.mu, self.yield_stress, self.hardening_exponent
        elastic_bound = 3.0 * mu * e_eq
        with np.errstate(divide='ignore'):
            plastic_bound = sy * (self.E * e_eq / (self.offset * sy)) ** (1.0 / n)
        sigma = np.minimum(elastic_bound, plastic_bound)
        for _ in range(100):
            residual = sigma / (3.0 * mu) + self.plastic_strain(sigma) - e_eq
            slope = 1.0 / (3.0 * mu) + self._plastic_slope(sigma)
            step = residual / slope
            sigma = sigma - step
            if np.all(np.abs(step) <= 1e-14 * np.maximum(sigma, sy)):
                return sigma
        raise RuntimeError(
            'the Ramberg-Osgood scalar inversion did not converge; the strain state '
            'is far outside the floating range of the hardening curve'
        )

    # -- the tensor law ------------------------------------------------------

    def _state(self, eps3: FloatArray) -> _DeviatoricState:
        '''The J2 state at `(n, 3, 3)` strains: split, invert the curve, and form the
        secant `c` relating `s = c e` (its elastic limit `2 mu` where `e = 0`).'''
        trace = np.einsum('eii->e', eps3)
        deviator = eps3 - (trace / 3.0)[:, None, None] * np.eye(3)
        e_eq = np.sqrt(np.einsum('eij,eij->e', deviator, deviator) * (2.0 / 3.0))
        sigma_eq = self.equivalent_stress(e_eq)
        small = e_eq <= 1e-16
        with np.errstate(divide='ignore', invalid='ignore'):
            secant = np.where(small, 2.0 * self.mu, 2.0 * sigma_eq / (3.0 * e_eq))
        return _DeviatoricState(trace, deviator, e_eq, sigma_eq, secant)

    def _stress3(self, state: _DeviatoricState) -> FloatArray:
        '''(n, 3, 3) stress: elastic volumetric part plus the secant deviatoric part.'''
        return (self.bulk * state.trace[:, None, None] * np.eye(3)
                + state.secant[:, None, None] * state.deviator)

    def _energy3(self, state: _DeviatoricState) -> FloatArray:
        '''(n,) stored energy: `1/2 K tr^2` plus the integral of the deviatoric curve.

        The deviatoric part is `int_0^e_eq sigma_eq(x) dx`, integrated by parts so it
        is closed-form in the already-solved `sigma_eq`:
        `sigma_eq e_eq - sigma_eq^2/(6 mu) - offset sigma_y^2/(E (n+1)) (sigma_eq/sigma_y)^(n+1)`.
        '''
        s, sy, n = state.sigma_eq, self.yield_stress, self.hardening_exponent
        deviatoric = (s * state.e_eq - s ** 2 / (6.0 * self.mu)
                      - self.offset * sy ** 2 / (self.E * (n + 1.0)) * (s / sy) ** (n + 1.0))
        return 0.5 * self.bulk * state.trace ** 2 + deviatoric

    def _tangent3(self, state: _DeviatoricState) -> FloatArray:
        '''(n, 3, 3, 3, 3) tangent d(sigma)/d(eps) at the state.

        `C = K I(x)I + c P_dev + (4/9)(sigma_eq' e_eq - sigma_eq)/e_eq^3 (e(x)e)`, with
        `sigma_eq' = d sigma_eq/d e_eq` the slope of the inverted curve and `P_dev` the
        symmetrized deviatoric projector. At `e_eq = 0` the correction vanishes and the
        first two terms are exactly Hooke's tensor; past yield `sigma_eq' < c`'s secant
        slope, so the correction softens the response along the loading direction while
        the transverse response keeps the secant stiffness. Symmetric in both pairs and
        under their swap: the Hessian of the stored energy.
        '''
        eye = np.eye(3)
        I_sym = 0.5 * (np.einsum('ik,jl->ijkl', eye, eye) + np.einsum('il,jk->ijkl', eye, eye))
        I_vol = np.einsum('ij,kl->ijkl', eye, eye)
        P_dev = I_sym - I_vol / 3.0

        slope = 1.0 / (1.0 / (3.0 * self.mu) + self._plastic_slope(state.sigma_eq))
        small = state.e_eq <= 1e-16
        with np.errstate(divide='ignore', invalid='ignore'):
            correction = np.where(
                small, 0.0,
                (4.0 / 9.0) * (slope * state.e_eq - state.sigma_eq) / state.e_eq ** 3)
        return (self.bulk * I_vol[None]
                + state.secant[:, None, None, None, None] * P_dev[None]
                + correction[:, None, None, None, None]
                * np.einsum('eij,ekl->eijkl', state.deviator, state.deviator))

    # -- the EnergyDensity interface ------------------------------------------

    def _lifted_state(self, grad_u: FloatArray) -> _DeviatoricState:
        '''The J2 state at the gradients' small strain, lifted to 3D.

        Plane strain: `eps_zz = 0`. The law is evaluated there and the in-plane blocks
        returned by the tiers below: with the out-of-plane strain held fixed, the
        in-plane stress is exactly the in-plane derivative of the 3D energy. The strain
        is affine in F, so P is the stress itself and A the strain tangent, as for the
        other small-strain density (`SmallStrain`).
        '''
        d = grad_u.shape[-1]
        eps = 0.5 * (grad_u + np.swapaxes(grad_u, -2, -1))
        eps3 = np.zeros((len(eps), 3, 3))
        eps3[:, :d, :d] = eps
        return self._state(eps3)

    def energy(self, grad_u: FloatArray) -> FloatArray:
        return self._energy3(self._lifted_state(grad_u))

    def stress(self, grad_u: FloatArray) -> tuple[FloatArray, FloatArray]:
        d = grad_u.shape[-1]
        state = self._lifted_state(grad_u)
        return self._energy3(state), self._stress3(state)[:, :d, :d]

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        '''Evaluate W, P = dW/dF, and A = d²W/dF² at `(n_elements, d, d)` gradients;
        see `_lifted_state` for the plane-strain lift.'''
        d = grad_u.shape[-1]
        state = self._lifted_state(grad_u)
        P = self._stress3(state)[:, :d, :d]
        A = self._tangent3(state)[:, :d, :d, :d, :d]
        return StrainEnergyDerivatives(W=self._energy3(state), P=P, A=A)

    def strain(self, grad_u: FloatArray) -> FloatArray:
        '''The small strain `eps = sym(grad u)`: this density's own measure.'''
        return 0.5 * (grad_u + np.swapaxes(grad_u, -2, -1))

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        '''The stress in the restrained z direction, which a 2D solve omits.

        Plane strain holds `eps_zz = 0`, but the deviatoric strain `e_zz = -tr(eps)/3`
        still carries the secant stress, so `sigma_zz = K tr(eps) + c e_zz`: read off
        the 3D law at the lifted strain, like the in-plane components.
        '''
        eps3 = np.zeros((len(strain), 3, 3))
        eps3[:, :2, :2] = strain
        state = self._state(eps3)
        return self._stress3(state)[:, 2, 2]
