"""Linearised buckling reproduces Euler's column theory.

The geometric-stiffness form is pinned analytically in `test_geometric_stiffness.py`;
this checks that assembling it against a prestress and solving `K φ = -λ K_g φ`
recovers the buckling loads and modes of a slender column:

    P_cr = π² E* I / (K L)²,   I = h³/12,   E* = E/(1-ν²) (plane strain),

with K set by the end conditions (2 cantilever, 1 pinned-pinned, 0.5 fixed-fixed,
about 0.7 fixed-pinned). P2 elements throughout, since the constant-strain triangle
locks in bending. Tolerances are headroom over the observed 1-2% error.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.elements import QuadraticTriangleElement
from fem.equations import LinearElastic, StrainMeasure
from fem.mesh.structured import create_rect_mesh
from fem.regions import intersect, on_plane
from fem.buckling import BucklingAnalysis

E, NU = 200.0, 0.3
E_STAR = E / (1 - NU**2)   # plane-strain effective modulus for bending


def _problem(mesh, bc=None, equation=None):
    equation = equation if equation is not None else LinearElastic(E, NU)
    return equation.problem(mesh, bc, element_type=QuadraticTriangleElement)


def column(length, height=1.0, n_length=40, n_across=5):
    """A slender rectangular column, with `n_across` elements through the thickness so
    bending is resolved."""
    return create_rect_mesh(corners=[[0, 0], [length, height]],
                            resolution=(n_length, n_across))


def critical_load(mesh, bc, length, height=1.0, n_modes=2):
    """The lowest buckling load and the whole solution for `mesh` under `bc`.

    The load factor multiplies the axial force the reference solve carries, read at
    mid-span where it is uniform and free of the end disturbances."""
    solution = BucklingAnalysis(n_modes=n_modes).solve(_problem(mesh, bc))
    assert solution.reference is not None
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    dx = length / (len(np.unique(mesh.vertices[:, 0])) - 1)
    midspan = np.abs(centroids[:, 0] - length / 2) < dx
    axial_force = -np.mean(solution.reference.stress[midspan, 0, 0]) * height
    return solution.load_factors * axial_force, solution


def euler_load(length, factor_K, height=1.0):
    """Euler's critical load for effective-length factor `factor_K`."""
    moment = height**3 / 12
    return np.pi**2 * E_STAR * moment / (factor_K * length) ** 2


def cantilever_bc(length):
    """Fixed-free: the one end condition needing no support on the loaded edge."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])        # clamp
    bc.add(BCType.NEUMANN, on_plane(0, length), [-1.0, 0])    # axial compression
    return bc


def test_cantilever_matches_euler():
    """The fixed-free column buckles at π² E* I / (2L)², to within P2's ~1%."""
    length = 24.0
    mesh = column(length, n_length=48, n_across=6)
    loads, _ = critical_load(mesh, cantilever_bc(length), length)
    np.testing.assert_allclose(loads[0], euler_load(length, 2.0), rtol=0.03)


def test_load_factors_are_positive_and_ascending():
    """Every reported factor is a real (compressive) buckling load, lowest first."""
    length = 24.0
    mesh = column(length, n_length=48, n_across=6)
    _, solution = critical_load(mesh, cantilever_bc(length), length, n_modes=4)
    factors = solution.load_factors
    assert np.all(factors > 0)
    assert np.all(np.diff(factors) >= 0)


def test_cantilever_higher_modes_follow_the_odd_square_law():
    """A cantilever's buckling loads go as (2n-1)²: the second mode is ~9x the first."""
    length = 24.0
    mesh = column(length, n_length=64, n_across=6)
    loads, _ = critical_load(mesh, cantilever_bc(length), length, n_modes=3)
    np.testing.assert_allclose(loads[1] / loads[0], 9.0, rtol=0.05)


def test_critical_load_scales_as_inverse_length_squared():
    """P_cr ∝ 1/L²: the slenderness law, read off as a log-log slope of -2."""
    lengths = np.array([16.0, 24.0, 32.0, 48.0])
    loads = np.array([
        critical_load(column(L, n_length=max(32, int(2 * L))), cantilever_bc(L), L)[0][0]
        for L in lengths
    ])
    slope = np.polyfit(np.log(lengths), np.log(loads), 1)[0]
    np.testing.assert_allclose(slope, -2.0, atol=0.05)


def test_effective_length_factors_across_end_conditions():
    """The four classic end conditions recover their effective-length factors to within a
    few percent of 2, 1, 1/2, and ~0.7 (a 2D clamp adds a little Saint-Venant
    stiffening). What sets the factor is whether an end can rotate: a traction-loaded
    edge (u_x free) rotates, an imposed uniform displacement (u_x fixed) cannot."""
    length = 24.0
    mesh = column(length, n_length=48, n_across=6)
    delta = 0.02 * length
    mid_left = intersect(on_plane(0, 0.0), on_plane(1, 0.5))

    def clamp_left(bc):
        bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    # Cantilever (K=2): clamp one end, compress the free one.
    cantilever = cantilever_bc(length)

    # Pinned-pinned (K=1): both edges held transversely (u_y=0) with the axial DOF free
    # so each end rotates; one point anchors the rigid axial slide; traction compresses.
    pinned = BoundaryConditions()
    pinned.add(BCType.DIRICHLET, on_plane(0, 0.0), [None, 0])
    pinned.add(BCType.DIRICHLET, mid_left, [0, 0])
    pinned.add(BCType.DIRICHLET, on_plane(0, length), [None, 0])
    pinned.add(BCType.NEUMANN, on_plane(0, length), [-1.0, 0])

    # Fixed-fixed (K=0.5): clamp one end; drive the other by an imposed uniform axial
    # displacement with u_y=0 -- rotation clamped at both ends.
    fixed = BoundaryConditions()
    clamp_left(fixed)
    fixed.add(BCType.DIRICHLET, on_plane(0, length), [-delta, 0])

    # Fixed-pinned (K≈0.7): clamp one end, pin (u_y=0 edge, u_x free) and compress the other.
    fixed_pinned = BoundaryConditions()
    clamp_left(fixed_pinned)
    fixed_pinned.add(BCType.DIRICHLET, on_plane(0, length), [None, 0])
    fixed_pinned.add(BCType.NEUMANN, on_plane(0, length), [-1.0, 0])

    def measured_K(bc):
        load = critical_load(mesh, bc, length)[0][0]
        return np.pi / length * np.sqrt(E_STAR * (1 / 12) / load)

    factors = {name: measured_K(bc) for name, bc in (
        ('cantilever', cantilever), ('pinned', pinned),
        ('fixed', fixed), ('fixed_pinned', fixed_pinned))}

    assert factors['cantilever'] == pytest.approx(2.0, rel=0.05)
    assert factors['pinned'] == pytest.approx(1.0, rel=0.05)
    assert factors['fixed'] == pytest.approx(0.5, rel=0.05)
    assert factors['fixed_pinned'] == pytest.approx(0.699, rel=0.05)


def test_green_lagrange_equation_is_rejected():
    """Linearised buckling needs the constant small-strain stiffness; a finite-strain
    equation has none, so its problem is refused."""
    mesh = column(12.0, n_length=12, n_across=3)
    equation = LinearElastic(E, NU, kinematics=StrainMeasure.GREEN_LAGRANGE)
    with pytest.raises(TypeError, match='constant tangent'):
        BucklingAnalysis(n_modes=2).solve(_problem(mesh, equation=equation))


def test_scalar_problem_is_rejected():
    """Buckling reads a prestress, so a problem without recovered stress is refused."""
    from fem.equations import Poisson

    mesh = column(12.0, n_length=12, n_across=3)
    scalar = Poisson(source=1.0)
    with pytest.raises(TypeError, match='recovered stress'):
        BucklingAnalysis().solve(scalar.problem(mesh))


def test_degenerate_parameters_are_rejected():
    with pytest.raises(ValueError, match='n_modes'):
        BucklingAnalysis(n_modes=0)


def test_no_compression_means_no_buckling():
    """With no load there is no prestress, K_g vanishes, and the analysis reports no
    buckling mode rather than handing the eigensolver an all-zero K_g."""
    mesh = column(12.0, n_length=12, n_across=4)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])   # clamped, but nothing applied
    with pytest.raises(ValueError, match='compressive prestress'):
        BucklingAnalysis().solve(_problem(mesh, bc))
