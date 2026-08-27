"""Modal analysis reproduces Euler-Bernoulli beam vibration.

The free-vibration pencil `K φ = ω² M φ` against a cantilever's natural frequencies,

    f_n = (β_n L)² / (2π) · sqrt(E* I / (ρ A L⁴)),

with β_n L = 1.875, 4.694, 7.855, … for a fixed-free beam, I = h³/12, A = h (unit
depth), and E* = E/(1-ν²). P2 elements throughout, since the constant-strain triangle
locks in bending. A 2D continuum carries a little shear flexibility and rotary inertia
that Euler-Bernoulli omits, so the frequencies sit a hair below the beam-theory value.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.elements import QuadraticTriangleElement
from fem.equations import Elasticity, StrainMeasure
from fem.mesh.structured import create_rect_mesh
from fem.regions import on_plane
from fem.solution import ModalSolution, Solution
from fem.space import FunctionSpace
from fem.modal import ModalAnalysis

E, NU, DENSITY = 200.0, 0.3, 1.0
E_STAR = E / (1 - NU**2)                                   # plane-strain modulus for bending
BETA_L = np.array([1.875104, 4.694091, 7.854757, 10.995541])   # fixed-free beam roots


def cantilever(length, height=1.0, n_length=48, n_across=6):
    """A slender rectangular beam, with `n_across` elements through the thickness so bending
    is resolved."""
    return create_rect_mesh(corners=[[0, 0], [length, height]],
                            resolution=(n_length, n_across))


def clamped_bc():
    """Fixed-free: one end clamped, the rest free, no load."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    return bc


def solve_modes(mesh, n_modes=6, density=DENSITY, E=E):
    equation = Elasticity(E, NU, density=density)
    problem = equation.problem(mesh, clamped_bc(), element_type=QuadraticTriangleElement)
    return ModalAnalysis(n_modes=n_modes).solve(problem)


def euler_bernoulli_hz(length, height=1.0, n=4):
    """The first `n` cantilever natural frequencies (Hz) from beam theory."""
    moment = height**3 / 12
    area = height
    omega = BETA_L[:n]**2 * np.sqrt(E_STAR * moment / (DENSITY * area * length**4))
    return omega / (2 * np.pi)


def test_first_bending_frequency_matches_euler_bernoulli():
    """The fundamental frequency lands on beam theory to within P2's few-percent headroom."""
    length = 24.0
    sol = solve_modes(cantilever(length))
    np.testing.assert_allclose(sol.frequencies[0], euler_bernoulli_hz(length, n=1)[0], rtol=0.02)


def test_bending_mode_ratios_follow_beam_theory():
    """The low modes are all bending, in the ratio (β_n/β_1)²: ~1 : 6.27 : 17.5."""
    length = 24.0
    sol = solve_modes(cantilever(length, n_length=64), n_modes=3)
    ratios = sol.frequencies[:3] / sol.frequencies[0]
    expected = (BETA_L[:3] / BETA_L[0]) ** 2
    np.testing.assert_allclose(ratios[1], expected[1], rtol=0.03)
    np.testing.assert_allclose(ratios[2], expected[2], rtol=0.05)


def test_frequency_scales_as_sqrt_stiffness_over_density():
    """f ∝ sqrt(E/ρ): 4x stiffness gives 2x frequency and 4x density half, to round-off,
    with the mode shapes unchanged."""
    mesh = cantilever(24.0)
    base = solve_modes(mesh, n_modes=1)
    stiffer = solve_modes(mesh, n_modes=1, E=4 * E)
    heavier = solve_modes(mesh, n_modes=1, density=4 * DENSITY)
    np.testing.assert_allclose(stiffer.frequencies[0], 2 * base.frequencies[0], rtol=1e-6)
    np.testing.assert_allclose(heavier.frequencies[0], 0.5 * base.frequencies[0], rtol=1e-6)


def test_modes_are_mass_orthonormal():
    """The mode shapes are M-orthonormal: φ_iᵀ M φ_j = δ_ij on the lifted vectors."""
    mesh = cantilever(24.0)
    sol = solve_modes(mesh, n_modes=4)
    mass = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2).mass_matrix
    gram = sol.modes @ (mass @ sol.modes.T)
    assert np.allclose(gram, np.eye(len(sol.frequencies)), atol=1e-8)


def test_frequencies_are_positive_ascending_and_units_consistent():
    """Every frequency is real and positive, ordered low-to-high, with f = ω/2π = 1/T."""
    sol = solve_modes(cantilever(24.0), n_modes=4)
    assert np.all(sol.frequencies > 0)
    assert np.all(np.diff(sol.frequencies) >= 0)
    np.testing.assert_allclose(sol.frequencies, sol.angular_frequencies / (2 * np.pi), rtol=1e-12)
    np.testing.assert_allclose(sol.periods, 1 / sol.frequencies, rtol=1e-12)


def test_solution_round_trips_through_io(tmp_path):
    """A ModalSolution saves and loads like any other."""
    sol = solve_modes(cantilever(16.0, n_length=24), n_modes=3)
    path = str(tmp_path / 'modal.npz')
    sol.save(path)
    loaded = Solution.load(path)
    assert isinstance(loaded, ModalSolution)
    np.testing.assert_allclose(loaded.angular_frequencies, sol.angular_frequencies)
    np.testing.assert_allclose(loaded.modes, sol.modes)


def test_green_lagrange_equation_is_rejected():
    """Modal analysis linearises about the unstressed state; a finite-strain law has no
    constant stiffness, so its problem is refused."""
    mesh = cantilever(12.0, n_length=12, n_across=3)
    equation = Elasticity(E, NU, kinematics=StrainMeasure.GREEN_LAGRANGE)
    with pytest.raises(TypeError, match='constant tangent'):
        ModalAnalysis(n_modes=2).solve(equation.problem(mesh))


def test_degenerate_parameters_are_rejected():
    """n_modes and the equation's density must be physical, caught at construction."""
    with pytest.raises(ValueError, match='n_modes'):
        ModalAnalysis(n_modes=0)
    with pytest.raises(ValueError, match='density'):
        Elasticity(E, NU, density=0.0)
