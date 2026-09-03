"""Ramberg-Osgood deformation-theory plasticity on the shared `EnergyForm` machinery.

The law is history-free (stress is a function of the current strain), so the checks
mirror the hyperelastic ones: the scalar curve inverts exactly, P and A are the energy
derivatives (finite differences, at a state well past yield), the law collapses to
Hooke below yield, and a homogeneous simple-shear boundary-value problem reproduces
the scalar curve through the full Newton path, an analytic anchor since a constant
stress field satisfies equilibrium exactly on any mesh. The inhomogeneous anchor is
Hill's pressurized thick-walled cylinder, run through the demo physics.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.numerics import central_difference_order
from fem.physics.energies import SmallStrain
from fem.physics.equations import DeformationPlasticity, LinearElastic
from fem.physics.plasticity import RambergOsgood
from fem.post.solution import ElasticSolution
from fem.regions import everywhere, intersect, on_plane
from fem.algebra.integrators import ThetaMethod
from fem.algebra.solve import BacktrackingLineSearch, NewtonSolve

E, NU = 200.0, 0.3
SIGMA_Y = 0.5           # yield strain ~2.5e-3: well inside small strain
HARDENING = 6.0


def _density(**kw) -> RambergOsgood:
    return RambergOsgood(E, NU, SIGMA_Y, HARDENING, **kw)


# -- the scalar curve --------------------------------------------------------


@pytest.mark.parametrize('n', [1.0, 5.0, 12.0])
def test_scalar_inversion_round_trips(n):
    """equivalent_stress inverts the strain-explicit curve: mapping stresses through
    e_eq(sigma) and back recovers them to full precision, zero and deep-plastic
    values included."""
    density = RambergOsgood(E, NU, SIGMA_Y, n)
    sigma = SIGMA_Y * np.array([0.0, 1e-3, 0.3, 1.0, 1.7, 4.0])
    e_eq = sigma / (3.0 * density.mu) + density.plastic_strain(sigma)
    np.testing.assert_allclose(density.equivalent_stress(e_eq), sigma,
                               rtol=1e-10, atol=1e-14 * SIGMA_Y)


def test_pure_shear_matches_an_independent_root():
    """At pure shear past yield, sigma_xy = sigma_eq / sqrt(3) with sigma_eq the root of
    the scalar curve, found here by an independent solver (brentq)."""
    density = _density()
    gamma = 0.01   # engineering shear strain, ~5x the yield strain in e_eq
    grad = np.zeros((1, 3, 3))
    grad[0, 0, 1] = grad[0, 1, 0] = gamma / 2.0
    e_eq = gamma / np.sqrt(3.0)

    root = brentq(lambda s: s / (3 * density.mu) + density.plastic_strain(s) - e_eq,
                  0.0, 3 * density.mu * e_eq)
    P = density.evaluate(grad).P[0]
    assert root < 3 * density.mu * e_eq * 0.99, 'the state must actually be plastic'
    np.testing.assert_allclose(P[0, 1], root / np.sqrt(3.0), rtol=1e-9)
    np.testing.assert_allclose(P[1, 0], P[0, 1], rtol=1e-12)


# -- the tensor law and its derivatives ---------------------------------------


@pytest.mark.parametrize('d', [2, 3])
def test_stress_and_tangent_are_the_energy_derivatives(d):
    """P is the gradient of W and A the gradient of P, each to O(eps^2) under central
    differences, at a state well past yield where all three curve terms are active.
    The eps sweep stays far below the state's own scale so it never crosses the
    e_eq = 0 kink of the deviatoric norm."""
    density = _density()
    rng = np.random.default_rng(0)
    grad0 = 0.05 * rng.standard_normal(d * d)
    eps = np.logspace(-5.0, -2.5, 8)

    def W(g):
        return float(density.evaluate(g.reshape(1, d, d)).W[0])

    def P(g):
        return density.evaluate(g.reshape(1, d, d)).P[0].ravel()

    p0 = density.evaluate(grad0.reshape(1, d, d)).P[0].ravel()
    a0 = density.evaluate(grad0.reshape(1, d, d)).A[0].reshape(d * d, d * d)

    grad_order = central_difference_order(W, lambda dv: p0 @ dv, grad0, eps=eps)
    hess_order = central_difference_order(P, lambda dv: a0 @ dv, grad0, eps=eps)
    assert 1.9 < grad_order < 2.1, f'P disagrees with dW/dF: order {grad_order:.3f}'
    assert 1.9 < hess_order < 2.1, f'A disagrees with dP/dF: order {hess_order:.3f}'


@pytest.mark.parametrize('d', [2, 3])
def test_tangent_is_symmetric(d):
    """A is the Hessian of a stored energy, so it is symmetric under the pair swap,
    at plastic states included (the softening correction is a symmetric dyad)."""
    density = _density()
    grad = 0.05 * np.random.default_rng(1).standard_normal((3, d, d))
    A = density.evaluate(grad).A
    np.testing.assert_allclose(A, A.transpose(0, 3, 4, 1, 2), atol=1e-12 * E)


@pytest.mark.parametrize('d', [2, 3])
def test_reduces_to_hooke_below_yield(d):
    """Far below yield the plastic term is O((sigma/sigma_y)^(n-1)) and the law is
    Hooke's: stress and tangent match the small-strain elastic density tightly."""
    plastic = _density()
    elastic = SmallStrain(E, NU)
    grad = 1e-5 * np.random.default_rng(2).standard_normal((4, d, d))

    ro, hooke = plastic.evaluate(grad), elastic.evaluate(grad)
    np.testing.assert_allclose(ro.P, hooke.P, rtol=0, atol=1e-8 * np.abs(hooke.P).max())
    np.testing.assert_allclose(ro.A, hooke.A, rtol=0, atol=1e-8 * E)
    np.testing.assert_allclose(ro.W, hooke.W, rtol=1e-7)


def test_pure_volumetric_strain_stays_elastic():
    """Plastic flow is deviatoric, so a purely volumetric strain (e_eq = 0) carries the
    exact elastic stress 3K c and the exact Hooke tangent, however large the trace."""
    density = _density()
    c = 0.4   # sigma_m far above sigma_y: only the deviator can yield
    grad = np.repeat(c * np.eye(3)[None], 2, axis=0)
    result = density.evaluate(grad)
    expected = np.repeat(3.0 * density.bulk * c * np.eye(3)[None], 2, axis=0)
    np.testing.assert_allclose(result.P, expected, rtol=1e-12)
    np.testing.assert_allclose(result.A, SmallStrain(E, NU).evaluate(grad).A, rtol=1e-12)


def test_out_of_plane_stress_reads_the_3d_law():
    """Plane strain holds eps_zz = 0 but the deviator e_zz = -tr/3 still carries
    stress: sigma_zz from the 2D reduction equals the 3D law's at the lifted strain."""
    density = _density()
    rng = np.random.default_rng(3)
    grad2 = 0.02 * rng.standard_normal((5, 2, 2))
    strain2 = density.strain(grad2)

    grad3 = np.zeros((5, 3, 3))
    grad3[:, :2, :2] = grad2
    sigma3 = density.evaluate(grad3).P
    np.testing.assert_allclose(density.out_of_plane_stress(strain2), sigma3[:, 2, 2],
                               rtol=1e-12)


def test_constructor_rejects_bad_parameters():
    for kwargs in ({'yield_stress': 0.0}, {'hardening_exponent': 0.5}, {'offset': 0.0}):
        full = {'yield_stress': SIGMA_Y, 'hardening_exponent': HARDENING} | kwargs
        with pytest.raises(ValueError):
            RambergOsgood(E, NU, **full)


# -- through the solver -------------------------------------------------------


def test_homogeneous_shear_solve_reproduces_the_curve(make_unit_square):
    """The analytic anchor: u = (gamma*y, 0) prescribed on the whole boundary makes the
    exact solution homogeneous simple shear (constant stress satisfies equilibrium on
    any mesh), so the Newton solve must return the interpolant exactly and recover the
    scalar curve's stress in every element, well past yield."""
    mesh = make_unit_square(6)
    gamma = 0.02   # e_eq = gamma/sqrt(3) ~ 5x the yield strain

    def sheared(points):
        return np.column_stack([gamma * points[:, 1], np.zeros(len(points))])

    bc = Conditions(Dirichlet(everywhere(), sheared))
    equation = DeformationPlasticity(E, NU, SIGMA_Y, HARDENING)
    # A tight tolerance so the recovered stress can be compared at ~1e-6: Newton's
    # convergence test is on the step norm, so the returned iterate is one unapplied
    # step away from exact.
    strategy = NewtonSolve(tol=1e-12, line_search=BacktrackingLineSearch())
    solution = equation.problem(mesh, bc).solve(strategy=strategy)
    assert isinstance(solution, ElasticSolution)

    exact = sheared(mesh.vertices).ravel()
    np.testing.assert_allclose(solution.dofs, exact, rtol=0, atol=1e-9)

    density = equation.energy_density()
    sigma_eq = float(density.equivalent_stress(np.array([gamma / np.sqrt(3.0)]))[0])
    assert sigma_eq > SIGMA_Y, 'the state must actually be plastic'
    np.testing.assert_allclose(solution.stress[:, 0, 1], sigma_eq / np.sqrt(3.0), rtol=1e-6)
    # The reported stress is Cauchy (J^-1 P F^T, the EnergyForm convention for every
    # density), which at simple shear adds O(gamma) normal terms to the pure-shear P;
    # their von Mises contribution is O(gamma^2), a few 1e-5 here.
    np.testing.assert_allclose(solution.von_mises, sigma_eq, rtol=1e-3)


def test_yielding_softens_the_response(make_unit_square):
    """Pulled past yield, the plate stretches by more than Hooke's law says and its
    stress stays near the flow stress: the redistribution flow-theory plasticity would
    also show under this monotonic load."""
    mesh = make_unit_square(8)
    traction = 1.6 * SIGMA_Y
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, None]),
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0]),
        Neumann(on_plane(0, 1.0), [traction, 0]),
    )
    plastic = DeformationPlasticity(E, NU, SIGMA_Y, HARDENING).problem(mesh, bc).solve()
    elastic = LinearElastic(E, NU).problem(mesh, bc).solve()

    end = np.isclose(mesh.vertices[:, 0], 1.0)
    stretch = plastic.nodal_values[end, 0].mean() / elastic.nodal_values[end, 0].mean()
    assert stretch > 2.0, f'past yield the block must be far softer than Hooke, got {stretch:.2f}x'
    # The section still carries the applied traction; yielding reshuffles strain, not
    # equilibrium. Loosely banded: under plane strain the out-of-plane restraint puts
    # the von Mises stress at ~sqrt(3)/2 of the axial stress, not equal to it.
    assert 0.7 * traction < np.median(plastic.von_mises) < 1.1 * traction


def test_plastic_front_tracks_hills_cylinder():
    """The classical benchmark: a thick-walled cylinder pressurized past first yield
    develops a plastic front at the radius Hill's elastic-plastic solution gives for
    that pressure. Run through the demo physics (quarter annulus, curved quadratic
    elements, near-perfectly-plastic hardening) at a coarse setting: the front must
    track Hill within a few percent of the wall thickness, sit at the bore below
    first yield, and march monotonically outward with the pressure."""
    from demos.pressurized_cylinder.physics import run

    s = run(n_pressures=4, max_area_fraction=0.004, resolution=0.04)
    wall = s.outer - s.inner
    assert s.pressures[0] < s.first_yield and s.fronts[0] == s.inner
    assert all(a < b for a, b in zip(s.fronts, s.fronts[1:])), 'the front must advance'
    np.testing.assert_allclose(s.fronts, s.hill_fronts, rtol=0, atol=0.06 * wall)


def test_equation_is_steady_only(make_unit_square):
    """Deformation theory has no meaning along a time-dependent path (its unloading is
    elastic reversal), so the equation carries time order 0 only and the integrators
    refuse it."""
    mesh = make_unit_square(2)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]))
    problem = DeformationPlasticity(E, NU, SIGMA_Y, HARDENING).problem(mesh, bc)
    assert problem.time_orders == frozenset({0})
    with pytest.raises(TypeError, match='first-order'):
        ThetaMethod(dt=0.1, steps=2).solve(problem)
