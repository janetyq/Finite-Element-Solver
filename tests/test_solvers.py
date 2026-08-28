"""Solve tests asserting on physical and mathematical invariants."""
import numpy as np
import pytest

from fem.numerics import bump_function
from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.regions import everywhere, on_plane
from fem.physics.equations import Heat, Projection, Poisson, LinearElastic, Wave
from fem.algebra.integrators import NewmarkMethod, ThetaMethod, wave_energy


def _on(equation, mesh, bc=None):
    return equation.problem(mesh, bc)


def test_heat_conserves_mean_temperature(make_unit_square):
    """A theta-method with no-flux boundaries conserves total heat: K annihilates constants,
    so 1^T M u is invariant for any theta."""
    mesh = make_unit_square(20)
    corner = mesh.vertices.max(axis=0)
    u0 = bump_function(mesh.vertices, corner, mag=50, size=0.3) + 300

    problem = _on(Heat(), mesh)  # no source, no BC -> natural (no-flux) boundaries
    solution = ThetaMethod(dt=0.01, steps=5).solve(problem, u0.copy())

    means = [problem.space.mean_value(u) for u in solution.u]
    assert np.allclose(means, means[0], rtol=1e-6), f"mean temperature drifted: {means}"


def test_l2_projection_reproduces_linear_field(make_unit_square):
    """A linear function is representable exactly by linear elements, so its L2 projection
    equals it at every node."""
    mesh = make_unit_square(20)

    def linear_field(p):
        return [2.0 * p[0] + 3.0 * p[1] - 1.0]

    solution = Projection(source=linear_field).problem(mesh).solve()

    u = solution.u
    expected = np.array([linear_field(v)[0] for v in mesh.vertices])
    assert np.allclose(u, expected, atol=1e-8), "linear field not reproduced exactly"


def _pinned_square(make_unit_square, n=12):
    """Unit square with every boundary node pinned at u = 0."""
    mesh = make_unit_square(n)
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    return mesh, bc


def test_wave_holds_static_equilibrium_under_load(make_unit_square):
    """Started at the static solution with zero velocity, the wave sits still: Newmark
    computes a zero initial acceleration and never moves."""
    mesh, bc = _pinned_square(make_unit_square)
    source = 1.0

    u_static = Poisson(source=source).problem(mesh, bc).solve().u
    assert np.abs(u_static).max() > 0, "static solution is trivial; test proves nothing"

    problem = _on(Wave(stiffness=1.0, source=source), mesh, bc)
    v0 = np.zeros(len(u_static))
    u_values = NewmarkMethod(dt=0.01, steps=20).solve(problem, u_static.copy(), v0).u

    assert np.allclose(u_values[-1], u_static, atol=1e-8), "equilibrium drifted"


def test_wave_honors_dirichlet_bcs(make_unit_square):
    """Pinned boundary nodes stay pinned for the whole run, while the interior moves."""
    mesh, bc = _pinned_square(make_unit_square)
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    u0[mesh.boundary_idxs] = 0.0

    problem = _on(Wave(stiffness=1.0), mesh, bc)
    u_values = NewmarkMethod(dt=0.01, steps=20).solve(problem, u0.copy(), np.zeros(len(u0))).u

    for step, u in enumerate(u_values):
        assert np.allclose(u[mesh.boundary_idxs], 0.0, atol=1e-10), \
            f"pinned boundary moved at step {step}"
    assert not np.allclose(u_values[-1], u0), "solution never evolved"


def test_wave_conserves_energy(make_unit_square):
    """Average-acceleration Newmark conserves 1/2 (c^2 u^T K u + v^T M v) for a linear
    system, to solver tolerance."""
    mesh, bc = _pinned_square(make_unit_square)
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    u0[mesh.boundary_idxs] = 0.0

    problem = _on(Wave(stiffness=4.0), mesh, bc)
    solution = NewmarkMethod(dt=0.005, steps=40).solve(problem, u0.copy(), np.zeros(len(u0)))

    energies = [
        wave_energy(problem, u, v)
        for u, v in zip(solution.u, solution.dudt)
    ]
    drift = max(abs(e - energies[0]) for e in energies) / energies[0]
    assert drift < 1e-9, f"energy drifted by {drift:.2e}: {energies}"


def test_wave_rejects_inconsistent_initial_state(make_unit_square):
    """u0 that disagrees with the Dirichlet data is a modelling error."""
    mesh, bc = _pinned_square(make_unit_square, n=8)
    n = len(mesh.vertices)
    u0 = np.ones(n)  # nonzero on the pinned boundary

    problem = _on(Wave(stiffness=1.0), mesh, bc)
    with pytest.raises(ValueError):
        NewmarkMethod(dt=0.01, steps=2).solve(problem, u0, np.zeros(n))


def test_linear_elastic_stretches_under_tension(make_unit_square):
    """A bar fixed on the left and pulled right elongates in +x, with the left edge unmoved."""
    mesh = make_unit_square(20)

    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )

    eq = LinearElastic(E=200, nu=0.4)
    solution = eq.problem(mesh, bc).solve()

    bidx = mesh.boundary_idxs
    bx = mesh.vertices[bidx, 0]
    left, right = bidx[np.isclose(bx, 0.0)], bidx[np.isclose(bx, 1.0)]

    u = solution.u.reshape(-1, 2)
    assert np.all(np.isfinite(u)), "displacement field has non-finite entries"
    assert np.allclose(u[left], 0.0, atol=1e-10), "fixed edge moved"
    assert u[right, 0].mean() > 0, "right edge did not elongate in +x"


def test_density_scales_the_mass_side_of_a_transient_problem(make_unit_square):
    """`Wave(T, density=4)` under Newmark is the same discrete system as `Wave(T/4)`; a
    heat problem with density 2 reaches at time t the state density 1 reaches at t/2."""
    mesh = make_unit_square(8)
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=1.0, size=0.2)
    v0 = np.zeros(len(u0))
    heavy = NewmarkMethod(dt=0.01, steps=10).solve(_on(Wave(stiffness=1.0, density=4.0), mesh), u0.copy(), v0)
    slow = NewmarkMethod(dt=0.01, steps=10).solve(_on(Wave(stiffness=0.25), mesh), u0.copy(), v0)
    np.testing.assert_allclose(heavy.u[-1], slow.u[-1], atol=1e-12)

    dense = ThetaMethod(dt=0.01, steps=10, theta=1.0).solve(_on(Heat(capacity=2.0), mesh), u0.copy())
    unit = ThetaMethod(dt=0.005, steps=10, theta=1.0).solve(_on(Heat(), mesh), u0.copy())
    np.testing.assert_allclose(dense.u[-1], unit.u[-1], atol=1e-12)
