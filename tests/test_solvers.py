"""Assertion-based solver tests.

These are the demos from Tests.py rewritten as real, headless regression tests:
same problem setups, but asserting on physical/mathematical invariants instead
of producing a plot for a human to eyeball.
"""
import numpy as np
import pytest

from fem.numerics import bump_function
from fem.boundary import BoundaryConditions, BCType
from fem.regions import everywhere, on_plane
from fem.solver import Solver, Projection, Poisson, LinearElastic
from fem.problem import heat, wave
from fem.integrators import Newmark, ThetaMethod, wave_energy


def test_heat_conserves_mean_temperature(make_unit_square):
    """A theta-method with no-flux (natural) boundaries conserves total heat.

    Because K annihilates constants (K @ 1 = 0) and 1^T K = 0, the theta-method
    step leaves 1^T M u invariant for any theta, so the mean temperature is
    constant -- Crank-Nicolson (the default) included, not just backward Euler.
    """
    mesh = make_unit_square(20)
    corner = mesh.vertices.max(axis=0)
    u0 = bump_function(mesh.vertices, corner, mag=50, size=0.3) + 300

    problem = heat(mesh)  # no source, no BC -> natural (no-flux) boundaries
    solution = ThetaMethod(dt=0.01, steps=5).run(problem, u0.copy())

    means = [problem.space.mean_value(u) for u in solution.get_values("u_values")]
    assert np.allclose(means, means[0], rtol=1e-6), f"mean temperature drifted: {means}"


def test_l2_projection_reproduces_linear_field(make_unit_square):
    """Projecting a field that already lives in the P1 space returns it exactly.

    A linear function is representable exactly by linear elements, so its L2
    projection onto the FE space must equal it at every node (to solver
    tolerance). This is a patch test for the mass-matrix assembly and solve.
    """
    mesh = make_unit_square(20)

    def linear_field(p):
        return [2.0 * p[0] + 3.0 * p[1] - 1.0]

    solution = Solver(mesh, Projection(source=linear_field)).solve()

    u = solution.get_values("u")
    expected = np.array([linear_field(v)[0] for v in mesh.vertices])
    assert np.allclose(u, expected, atol=1e-8), "linear field not reproduced exactly"


def _pinned_square(make_unit_square, n=12):
    """Unit square with every boundary node pinned at u = 0."""
    mesh = make_unit_square(n)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    return mesh, bc


def test_wave_holds_static_equilibrium_under_load(make_unit_square):
    """Started *at* the static solution with zero velocity, the wave must sit still.

    Equilibrium of M u" + c^2 K u = b at rest means c^2 K u = b, which for c = 1
    is exactly the Poisson solution. Newmark starting there computes a zero initial
    acceleration and never moves, so a nonzero load is held exactly.
    """
    mesh, bc = _pinned_square(make_unit_square)
    source = 1.0

    u_static = Solver(mesh, Poisson(source=source), bc).solve().get_values("u")
    assert np.abs(u_static).max() > 0, "static solution is trivial; test proves nothing"

    problem = wave(mesh, c=1, bc=bc, source=source)
    v0 = np.zeros(len(u_static))
    u_values = Newmark(dt=0.01, steps=20).run(problem, u_static.copy(), v0).get_values("u_values")

    assert np.allclose(u_values[-1], u_static, atol=1e-8), "equilibrium drifted"


def test_wave_honors_dirichlet_bcs(make_unit_square):
    """Pinned boundary nodes stay pinned for the whole run, while the interior moves."""
    mesh, bc = _pinned_square(make_unit_square)
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    u0[mesh.boundary_idxs] = 0.0

    problem = wave(mesh, c=1, bc=bc)
    u_values = Newmark(dt=0.01, steps=20).run(problem, u0.copy(), np.zeros(len(u0))).get_values("u_values")

    for step, u in enumerate(u_values):
        assert np.allclose(u[mesh.boundary_idxs], 0.0, atol=1e-10), \
            f"pinned boundary moved at step {step}"
    assert not np.allclose(u_values[-1], u0), "solution never evolved"


def test_wave_conserves_energy(make_unit_square):
    """Undamped, unforced, pinned: total energy is conserved.

    Average-acceleration Newmark (beta=1/4, gamma=1/2) conserves
    1/2 (c^2 u^T K u + v^T M v) exactly for a linear system, so this holds to
    solver tolerance rather than discretization error.
    """
    mesh, bc = _pinned_square(make_unit_square)
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    u0[mesh.boundary_idxs] = 0.0

    problem = wave(mesh, c=2, bc=bc)
    solution = Newmark(dt=0.005, steps=40).run(problem, u0.copy(), np.zeros(len(u0)))

    energies = [
        wave_energy(problem, u, v)
        for u, v in zip(solution.get_values("u_values"), solution.get_values("dudt_values"))
    ]
    drift = max(abs(e - energies[0]) for e in energies) / energies[0]
    assert drift < 1e-9, f"energy drifted by {drift:.2e}: {energies}"


def test_wave_rejects_inconsistent_initial_state(make_unit_square):
    """u0 that disagrees with the Dirichlet data is a modelling error."""
    mesh, bc = _pinned_square(make_unit_square, n=8)
    n = len(mesh.vertices)
    u0 = np.ones(n)  # nonzero on the pinned boundary

    problem = wave(mesh, c=1, bc=bc)
    with pytest.raises(ValueError):
        Newmark(dt=0.01, steps=2).run(problem, u0, np.zeros(n))


def test_linear_elastic_stretches_under_tension(make_unit_square):
    """A bar fixed on the left and pulled right elongates in +x.

    Fixing the left edge (u = 0) and applying a +x traction on the right edge
    should leave the left edge unmoved and push the right edge to positive x
    displacement, with everything finite. Validates the elastic stiffness
    assembly, the Neumann load path, and Dirichlet handling together.
    """
    mesh = make_unit_square(20)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])  # +x traction

    eq = LinearElastic(E=200, nu=0.4)
    solution = Solver(mesh, eq, bc).solve()

    bidx = mesh.boundary_idxs
    bx = mesh.vertices[bidx, 0]
    left, right = bidx[np.isclose(bx, 0.0)], bidx[np.isclose(bx, 1.0)]

    u = solution.get_values("u").reshape(-1, 2)
    assert np.all(np.isfinite(u)), "displacement field has non-finite entries"
    assert np.allclose(u[left], 0.0, atol=1e-10), "fixed edge moved"
    assert u[right, 0].mean() > 0, "right edge did not elongate in +x"
