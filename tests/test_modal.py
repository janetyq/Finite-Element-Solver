"""Modal analysis reproduces Euler-Bernoulli beam vibration.

The companion to `test_buckling.py`: where buckling checks the eigenproblem `K φ = -λ K_g φ`
against Euler's column loads, this checks the free-vibration pencil `K φ = ω² M φ` against
a cantilever's natural frequencies -- the analytic answer beam theory gives,

    f_n = (β_n L)² / (2π) · sqrt(E* I / (ρ A L⁴)),

with the roots β_n L = 1.875, 4.694, 7.855, … for a fixed-free beam, I = h³/12 the second
moment, A = h the area (unit depth), and E* = E/(1-ν²) the plane-strain modulus (the same
effective modulus the buckling test uses for bending).

Quadratic (P2) elements throughout, for the same reason as buckling: the constant-strain
triangle locks in bending and reaches the analytic frequencies only on a mesh refined hard
through the thickness, where P2 matches them on a coarse one. A 2D continuum also carries a
little shear flexibility and rotary inertia that Euler-Bernoulli omits, so the frequencies
sit a hair below the beam-theory value -- the tolerances are honest headroom over that.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.elements import QuadraticTriangleElement
from fem.equations import LinearElastic, Poisson, StrainMeasure
from fem.mesh.structured import create_rect_mesh
from fem.regions import on_plane
from fem.solution import ModalSolution, Solution
from fem.space import FunctionSpace
from fem.modal import ModalSolver

E, NU, DENSITY = 200.0, 0.3, 1.0
E_STAR = E / (1 - NU**2)                                   # plane-strain modulus for bending
BETA_L = np.array([1.875104, 4.694091, 7.854757, 10.995541])   # fixed-free beam roots


def cantilever(length, height=1.0, n_length=48, n_across=6):
    """A slender rectangular beam, meshed like the buckling column.

    `n_across` is set independently of the aspect ratio so bending is resolved through
    the thickness rather than left to a near-isotropic triangle's two or three elements.
    """
    return create_rect_mesh(corners=[[0, 0], [length, height]],
                            resolution=(n_length, n_across))


def clamped_bc():
    """Fixed-free: one end clamped, the rest free -- no load (modal analysis reads none)."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    return bc


def solve_modes(mesh, n_modes=6, density=DENSITY, E=E):
    solver = ModalSolver(mesh, LinearElastic(E, NU), clamped_bc(), n_modes=n_modes,
                         element_type=QuadraticTriangleElement, density=density)
    return solver.solve()


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
    """The low modes are all bending, in the ratio (β_n/β_1)²: ~1 : 6.27 : 17.5.

    Axial modes sit far above these on a slender beam, so the lowest three eigenvalues are
    the first three bending tones, whose spacing beam theory fixes independent of scale.
    """
    length = 24.0
    sol = solve_modes(cantilever(length, n_length=64), n_modes=3)
    ratios = sol.frequencies[:3] / sol.frequencies[0]
    expected = (BETA_L[:3] / BETA_L[0]) ** 2
    np.testing.assert_allclose(ratios[1], expected[1], rtol=0.03)
    np.testing.assert_allclose(ratios[2], expected[2], rtol=0.05)


def test_frequency_scales_as_sqrt_stiffness_over_density():
    """f ∝ sqrt(E/ρ): the material dependence, checked as exact scaling on one mesh.

    Scaling E or ρ uniformly multiplies K or M by a constant, which moves every ω by an
    exact factor without changing the mode shapes -- so the same mesh gives 2x frequency
    for 4x stiffness and half for 4x density, to round-off.
    """
    mesh = cantilever(24.0)
    base = solve_modes(mesh, n_modes=1)
    stiffer = solve_modes(mesh, n_modes=1, E=4 * E)
    heavier = solve_modes(mesh, n_modes=1, density=4 * DENSITY)
    np.testing.assert_allclose(stiffer.frequencies[0], 2 * base.frequencies[0], rtol=1e-6)
    np.testing.assert_allclose(heavier.frequencies[0], 0.5 * base.frequencies[0], rtol=1e-6)


def test_modes_are_mass_orthonormal():
    """The mode shapes are M-orthonormal: φ_iᵀ M φ_j = δ_ij, the normalisation eigsh imposes.

    Checked on the lifted full-DOF vectors against the same unit-density mass matrix the
    solver assembled (density defaults to 1); the fixed DOFs are zero, so they drop out.
    """
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
    """A ModalSolution saves and loads like any other -- reflected over its dataclass fields."""
    sol = solve_modes(cantilever(16.0, n_length=24), n_modes=3)
    path = str(tmp_path / 'modal.npz')
    sol.save(path)
    loaded = Solution.load(path)
    assert isinstance(loaded, ModalSolution)
    np.testing.assert_allclose(loaded.angular_frequencies, sol.angular_frequencies)
    np.testing.assert_allclose(loaded.modes, sol.modes)


def test_green_lagrange_equation_is_rejected():
    """Modal analysis linearises about the unstressed state; a finite-strain law has no
    constant stiffness, so it is refused rather than silently linearised."""
    mesh = cantilever(12.0, n_length=12, n_across=3)
    equation = LinearElastic(E, NU, kinematics=StrainMeasure.GREEN_LAGRANGE)
    with pytest.raises(NotImplementedError, match='constant'):
        ModalSolver(mesh, equation)


def test_non_elastic_equation_is_rejected():
    """The mass/stiffness pencil is an elastic one; a scalar PDE has no vibration modes here."""
    mesh = cantilever(12.0, n_length=12, n_across=3)
    with pytest.raises(ValueError, match='elastic vibration'):
        ModalSolver(mesh, Poisson())


def test_degenerate_parameters_are_rejected():
    """n_modes and density must be physical -- caught at construction, not at solve."""
    mesh = cantilever(12.0, n_length=12, n_across=3)
    with pytest.raises(ValueError, match='n_modes'):
        ModalSolver(mesh, LinearElastic(E, NU), n_modes=0)
    with pytest.raises(ValueError, match='density'):
        ModalSolver(mesh, LinearElastic(E, NU), density=0.0)
