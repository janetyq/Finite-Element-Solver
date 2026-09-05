"""Linearised buckling reproduces Euler's column theory.

The geometric-stiffness form is pinned analytically in `test_geometric_stiffness.py`;
this checks that assembling it against a prestress and solving `K φ = -λ K_g φ`
recovers the buckling loads and modes of a slender column:

    P_cr = π² E* I / (K L)²,   I = h³/12,   E* = E/(1-ν²) (plane strain),

with K set by the end conditions (2 cantilever, 1 pinned-pinned, 0.5 fixed-fixed,
about 0.7 fixed-pinned). P2 elements throughout, since the constant-strain triangle
locks in bending. Tolerances are headroom over the observed 1-2% error.

The second half checks `PostBucklingAnalysis`, which carries on where the eigenproblem
stops: the knee of the traced path against λ_cr, and the load carried past it against
the elastica's `P/P_cr = 1 + Θ²/8`.
"""
import logging

import numpy as np
import pytest

from fem.analysis.buckling import (
    BucklingAnalysis,
    PostBucklingAnalysis,
    _buckling_factors,
    _restated,
)
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.equations import FiniteStrainElastic, LinearElastic
from fem.regions import intersect, on_plane
from fem.space import FunctionSpace

E, NU = 200.0, 0.3
E_STAR = E / (1 - NU**2)   # plane-strain effective modulus for bending


def _problem(mesh, bc=None, equation=None):
    equation = equation if equation is not None else LinearElastic(E, NU)
    return equation.problem(mesh, bc, element_type=QuadraticTriangleElement)


def column(length, height=1.0, n_length=40, n_across=5):
    """A slender rectangular column, with `n_across` elements through the thickness so
    bending is resolved."""
    return box_mesh(corners=[[0, 0], [length, height]],
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
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, length), [-1.0, 0]),
    )
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

    clamp_left = Dirichlet(on_plane(0, 0.0), [0, 0])

    # Cantilever (K=2): clamp one end, compress the free one.
    cantilever = cantilever_bc(length)

    # Pinned-pinned (K=1): both edges held transversely (u_y=0) with the axial DOF free
    # so each end rotates; one point anchors the rigid axial slide; traction compresses.
    pinned = Conditions(
        Dirichlet(on_plane(0, 0.0), [None, 0]),
        Dirichlet(mid_left, [0, 0]),
        Dirichlet(on_plane(0, length), [None, 0]),
        Neumann(on_plane(0, length), [-1.0, 0]),
    )

    # Fixed-fixed (K=0.5): clamp one end; drive the other by an imposed uniform axial
    # displacement with u_y=0 -- rotation clamped at both ends.
    fixed = Conditions(
        clamp_left,
        Dirichlet(on_plane(0, length), [-delta, 0]),
    )

    # Fixed-pinned (K≈0.7): clamp one end, pin (u_y=0 edge, u_x free) and compress the other.
    fixed_pinned = Conditions(
        clamp_left,
        Dirichlet(on_plane(0, length), [None, 0]),
        Neumann(on_plane(0, length), [-1.0, 0]),
    )

    def measured_K(bc):
        load = critical_load(mesh, bc, length)[0][0]
        return np.pi / length * np.sqrt(E_STAR * (1 / 12) / load)

    factors = {name: measured_K(bc) for name, bc in (
        ('cantilever', cantilever), ('pinned', pinned),
        ('fixed', fixed), ('fixed_pinned', fixed_pinned))}

    assert factors['cantilever'] == pytest.approx(2.0, rel=0.05)
    assert factors['pinned'] == pytest.approx(1.0, rel=0.05)
    assert factors['fixed'] == pytest.approx(0.5, rel=0.05)
    # Fixed-pinned: K = pi / x for the first root of tan x = x, x = 4.4934, so 0.6992.
    assert factors['fixed_pinned'] == pytest.approx(0.699, rel=0.05)


def test_green_lagrange_equation_is_rejected():
    """Linearised buckling needs the constant small-strain stiffness; a finite-strain
    equation has none, so its problem is refused."""
    mesh = column(12.0, n_length=12, n_across=3)
    equation = FiniteStrainElastic(E, NU)
    with pytest.raises(TypeError, match='constant tangent'):
        BucklingAnalysis(n_modes=2).solve(_problem(mesh, equation=equation))


def test_scalar_problem_is_rejected():
    """Buckling reads a prestress, so a problem without recovered stress is refused."""
    from fem.physics.equations import Poisson

    mesh = column(12.0, n_length=12, n_across=3)
    scalar = Poisson()
    with pytest.raises(TypeError, match='recovered stress'):
        BucklingAnalysis().solve(scalar.problem(mesh, Conditions(Source(1.0))))


def test_degenerate_parameters_are_rejected():
    with pytest.raises(ValueError, match='n_modes'):
        BucklingAnalysis(n_modes=0)


def test_no_compression_means_no_buckling():
    """With no load there is no prestress, K_g vanishes, and the analysis reports no
    buckling mode rather than handing the eigensolver an all-zero K_g."""
    mesh = column(12.0, n_length=12, n_across=4)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]))
    with pytest.raises(ValueError, match='compressive prestress'):
        BucklingAnalysis().solve(_problem(mesh, bc))


# -- the post-buckling path -----------------------------------------------------------
#
# What the eigenproblem cannot say: how much load the column carries once it has bowed.
# `PostBucklingAnalysis` seeds an imperfection in the first mode and traces the path,
# and the yardsticks are the knee (at the critical load as the imperfection vanishes)
# and the elastica's initial post-buckling rise.


def knife_edge_bc(length, height=1.0):
    """Pinned-pinned with each end held at one point of the neutral axis.

    Holding a whole end edge transversely gives the same λ_cr (it does not resist
    rotation to first order) and a stiffer path past it, since the second-order sideways
    motion of a rotating section is held too. A knife edge is the elastica's end.
    """
    return Conditions(
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, height / 2)), [0, 0]),
        Dirichlet(intersect(on_plane(0, length), on_plane(1, height / 2)), [None, 0]),
        Neumann(on_plane(0, length), [-1.0, 0]),
    )


def _path_column(length=24.0, height=1.0):
    """A column coarse enough to trace a path in about a second."""
    return column(length, height, n_length=24, n_across=3), knife_edge_bc(length, height)


def _post_buckling(mesh, bc, amplitude, max_steps=40):
    """The traced path and its deflections: the largest transverse displacement at each
    state, which for the first mode is the mid-span bow."""
    result = PostBucklingAnalysis(imperfection=amplitude, max_steps=max_steps).solve(
        FiniteStrainElastic(E, NU), mesh, bc, element_type=QuadraticTriangleElement)
    deflections = np.array([float(np.abs(step.nodal_values[:, 1]).max())
                            for step in result.path])
    return result, deflections


def test_the_knee_of_the_path_approaches_the_critical_load():
    """The load carried at a fixed, visible deflection rises toward λ_cr as the
    imperfection shrinks: the path is a rounded-off corner and the corner is the
    linearised buckling load. At a tenth of the imperfection the knee sits within a
    percent of λ_cr, which is the whole claim of a linearised buckling analysis."""
    length = 24.0
    mesh, bc = _path_column(length)
    diagonal = np.hypot(length, 1.0)

    carried = []
    for amplitude in (1e-3 * diagonal, 1e-4 * diagonal):
        result, deflections = _post_buckling(mesh, bc, amplitude)
        assert deflections[-1] > 0.5, 'the path must reach a visibly bowed state'
        ratios = result.path.lambdas / result.critical_load_factor
        carried.append(float(np.interp(0.5, deflections, ratios)))

    coarse, fine = carried
    assert coarse < fine < 1.0
    assert fine == pytest.approx(1.0, rel=0.01)


def test_the_path_follows_the_elastica_past_the_critical_load():
    """Past the knee the column carries more than λ_cr, by the elastica's series

        P/P_cr = 1 + Θ²/8,   Θ the end rotation,

    which for a first-mode half sine of amplitude w over a span L has Θ = π w / L, so
    P/P_cr = 1 + (π w / L)²/8. The traced column is imperfect, and an imperfection of
    amplitude a costs it about a/w of λ_cr at deflection w (the classical
    w = a/(1 - λ/λ_cr)), so it is that deficit plus the rise that the series predicts."""
    length = 24.0
    amplitude = 1e-5 * np.hypot(length, 1.0)
    mesh, bc = _path_column(length)
    result, deflections = _post_buckling(mesh, bc, amplitude)
    ratios = result.path.lambdas / result.critical_load_factor

    for deflection in (1.0, 1.5):
        assert deflections[-1] > deflection
        rise = float(np.interp(deflection, deflections, ratios)) - 1.0
        elastica = (np.pi * deflection / length) ** 2 / 8
        assert rise + amplitude / deflection == pytest.approx(elastica, rel=0.1)
    assert np.all(result.path.stability == 1), 'the imperfect path never loses stability'


def test_the_conditions_are_restated_on_the_imperfect_mesh():
    """The rebinding risk, and the mitigation. The mode moves every vertex, the loaded
    end plane included, so a region written as `on_plane` selects different nodes on the
    imperfect mesh: here the traction finds no facet at all and the rebuild fails. The
    facade restates each region against the nodes it selected on the pristine mesh, so
    the same DOFs are held and the same load applied."""
    length = 24.0
    mesh, bc = _path_column(length)
    reference = _problem(mesh, bc)
    result, _ = _post_buckling(mesh, bc, 1e-2, max_steps=1)

    imperfect = result.mesh
    assert not np.allclose(imperfect.vertices, mesh.vertices)
    with pytest.raises(ValueError, match='no boundary facet'):
        _problem(imperfect, bc, equation=FiniteStrainElastic(E, NU))

    restated = _problem(imperfect, _restated(bc, reference.space),
                        equation=FiniteStrainElastic(E, NU))
    assert restated.partition == reference.partition
    # The same total load, up to the loaded facet having moved and rotated a little.
    np.testing.assert_allclose(restated.load.sum(), reference.load.sum(), rtol=1e-4)


def test_fewer_buckling_modes_than_asked_for_is_reported(caplog):
    """The positive-μ filter can leave fewer modes than were requested, since a
    stiffening direction has no buckling load. Handing back a short array silently is
    what the warning is for."""
    mu = np.array([2.0, -1.0, 0.5])
    modes = np.eye(3)
    with caplog.at_level(logging.WARNING, logger='fem.analysis.buckling'):
        factors, kept = _buckling_factors(mu, modes, 3)

    np.testing.assert_allclose(factors, [0.5, 2.0])
    assert len(kept) == 2
    assert '2 of the 3 requested modes buckle' in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fem.analysis.buckling'):
        _buckling_factors(mu, modes, 2)
    assert caplog.text == ''


def test_post_buckling_reports_a_mode_that_did_not_buckle(monkeypatch):
    """Asking for the third mode when only one of the eigenpairs buckles is refused by
    name, rather than by an IndexError out of the mode lookup."""
    mesh, bc = _path_column()
    solution = BucklingAnalysis(n_modes=1).solve(_problem(mesh, bc))
    monkeypatch.setattr(BucklingAnalysis, 'solve', lambda self, problem: solution)

    with pytest.raises(ValueError, match='only 1 of the 3 eigenpairs buckle'):
        PostBucklingAnalysis(n_modes=3, mode=2).solve(
            FiniteStrainElastic(E, NU), mesh, bc, element_type=QuadraticTriangleElement)


@pytest.mark.parametrize('kwargs, match', [
    ({'imperfection': 0.0}, 'positive amplitude'),
    ({'imperfection': -1.0}, 'positive amplitude'),
    ({'mode': -1}, 'index into the buckling modes'),
    ({'n_modes': 2, 'mode': 2}, 'n_modes must exceed mode'),
    ({'lambda_factor': 1.0}, 'lambda_factor'),
    ({'steps_to_critical': 0}, 'steps_to_critical'),
])
def test_post_buckling_rejects_bad_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PostBucklingAnalysis(**kwargs)


def test_post_buckling_needs_a_finite_strain_equation_and_a_mesh():
    """The path is the finite-strain problem's and the imperfection displaces a mesh, so
    a small-strain law or a ready-made space is refused with the alternative named."""
    mesh, bc = _path_column()
    with pytest.raises(TypeError, match='finite-strain law'):
        PostBucklingAnalysis().solve(LinearElastic(E, NU), mesh, bc)
    with pytest.raises(TypeError, match='takes a Mesh'):
        PostBucklingAnalysis().solve(FiniteStrainElastic(E, NU),
                                     FunctionSpace(mesh, n_components=2), bc)
