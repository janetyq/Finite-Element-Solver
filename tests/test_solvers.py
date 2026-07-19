"""Assertion-based solver tests.

These are the demos from Tests.py rewritten as real, headless regression tests:
same problem setups, but asserting on physical/mathematical invariants instead
of producing a plot for a human to eyeball.
"""
import numpy as np
import pytest

from fem.numerics import bump_function
from fem.boundary import BoundaryConditions
from fem.solver import Solver, Projection, Poisson, Heat, Wave, LinearElastic


def test_heat_conserves_mean_temperature(make_unit_square):
    """Backward Euler with no-flux (natural) boundaries conserves total heat.

    Because K annihilates constants (K @ 1 = 0), 1^T M u is invariant under the
    scheme (M + dt K) u_{n+1} = M u_n, so the mean temperature is constant.
    """
    femesh = make_unit_square(20)
    corner = femesh.vertices.max(axis=0)
    u0 = bump_function(femesh.vertices, corner, mag=50, size=0.3) + 300

    eq = Heat(u_initial=u0.copy(), iters=5, dt=0.01)
    solution = Solver(femesh, eq).solve()

    means = [femesh.calculate_mean_value(u) for u in solution.get_values("u_values")]
    assert np.allclose(means, means[0], rtol=1e-6), f"mean temperature drifted: {means}"


def test_l2_projection_reproduces_linear_field(make_unit_square):
    """Projecting a field that already lives in the P1 space returns it exactly.

    A linear function is representable exactly by linear elements, so its L2
    projection onto the FE space must equal it at every node (to solver
    tolerance). This is a patch test for the mass-matrix assembly and solve.
    """
    femesh = make_unit_square(20)

    def linear_field(p):
        return [2.0 * p[0] + 3.0 * p[1] - 1.0]

    bc = BoundaryConditions(femesh)
    bc.add_force(linear_field)
    solution = Solver(femesh, Projection(), bc).solve()

    u = solution.get_values("u")
    expected = np.array([linear_field(v)[0] for v in femesh.vertices])
    assert np.allclose(u, expected, atol=1e-8), "linear field not reproduced exactly"


def _pinned_square(make_unit_square, n=12):
    """Unit square with every boundary node pinned at u = 0."""
    femesh = make_unit_square(n)
    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", femesh.boundary_idxs, np.zeros(len(femesh.boundary_idxs)))
    return femesh, bc


def test_wave_holds_static_equilibrium_under_load(make_unit_square):
    """Started *at* the static solution with zero velocity, the wave must sit still.

    Equilibrium of M u" + c^2 K u = b at rest means c^2 K u = b, which for c = 1
    is exactly the Poisson solution. This pins down the Crank-Nicolson load term:
    the old `dt/2 * (b + np.roll(b, -1))` rolled the *spatial* load vector by one
    index, so the equilibrium would be violated the moment b is nonzero.
    """
    femesh, bc = _pinned_square(make_unit_square)
    bc.add_force(lambda p: [1.0])

    u_static = Solver(femesh, Poisson(), bc).solve().get_values("u")
    assert np.abs(u_static).max() > 0, "static solution is trivial; test proves nothing"

    eq = Wave(
        u_initial=u_static.copy(),
        dudt_initial=np.zeros(len(u_static)),
        c=1, dt=0.01, iters=20,
    )
    u_values = Solver(femesh, eq, bc).solve().get_values("u_values")

    assert np.allclose(u_values[-1], u_static, atol=1e-8), "equilibrium drifted"


def test_wave_honors_dirichlet_bcs(make_unit_square):
    """Pinned boundary nodes stay pinned for the whole run, while the interior moves."""
    femesh, bc = _pinned_square(make_unit_square)
    u0 = bump_function(femesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    u0[femesh.boundary_idxs] = 0.0

    eq = Wave(
        u_initial=u0.copy(),
        dudt_initial=np.zeros(len(u0)),
        c=1, dt=0.01, iters=20,
    )
    u_values = Solver(femesh, eq, bc).solve().get_values("u_values")

    for step, u in enumerate(u_values):
        assert np.allclose(u[femesh.boundary_idxs], 0.0, atol=1e-10), \
            f"pinned boundary moved at step {step}"
    assert not np.allclose(u_values[-1], u0), "solution never evolved"


def test_wave_conserves_energy(make_unit_square):
    """Undamped, unforced, pinned: total energy is conserved.

    Crank-Nicolson conserves 1/2 (c^2 u^T K u + v^T M v) exactly for a linear
    system, so this holds to solver tolerance rather than discretization error.
    """
    femesh, bc = _pinned_square(make_unit_square)
    u0 = bump_function(femesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    u0[femesh.boundary_idxs] = 0.0

    eq = Wave(
        u_initial=u0.copy(),
        dudt_initial=np.zeros(len(u0)),
        c=2, dt=0.005, iters=40,
    )
    solution = Solver(femesh, eq, bc).solve()

    energies = [
        femesh.calculate_energy(u, dudt, c=2)
        for u, dudt in zip(solution.get_values("u_values"), solution.get_values("dudt_values"))
    ]
    drift = max(abs(e - energies[0]) for e in energies) / energies[0]
    assert drift < 1e-9, f"energy drifted by {drift:.2e}: {energies}"


def test_wave_rejects_inconsistent_initial_state(make_unit_square):
    """u_initial that disagrees with the Dirichlet data is a modelling error."""
    femesh, bc = _pinned_square(make_unit_square, n=8)
    n = len(femesh.vertices)
    u0 = np.ones(n)  # nonzero on the pinned boundary

    eq = Wave(u_initial=u0, dudt_initial=np.zeros(n), c=1, dt=0.01, iters=2)
    with pytest.raises(ValueError):
        Solver(femesh, eq, bc).solve()


def test_linear_elastic_stretches_under_tension(make_unit_square):
    """A bar fixed on the left and pulled right elongates in +x.

    Fixing the left edge (u = 0) and applying a +x traction on the right edge
    should leave the left edge unmoved and push the right edge to positive x
    displacement, with everything finite. Validates the elastic stiffness
    assembly, the Neumann load path, and Dirichlet handling together.
    """
    femesh = make_unit_square(20)
    bidx = femesh.boundary_idxs
    bx = femesh.vertices[bidx, 0]
    left = bidx[np.isclose(bx, 0.0)]
    right = bidx[np.isclose(bx, 1.0)]

    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", left, [0, 0])
    bc.add("neumann", right, [50, 0])  # +x traction

    eq = LinearElastic(E=200, nu=0.4)
    solution = Solver(femesh, eq, bc).solve()

    u = solution.get_values("u").reshape(-1, 2)
    assert np.all(np.isfinite(u)), "displacement field has non-finite entries"
    assert np.allclose(u[left], 0.0, atol=1e-10), "fixed edge moved"
    assert u[right, 0].mean() > 0, "right edge did not elongate in +x"
