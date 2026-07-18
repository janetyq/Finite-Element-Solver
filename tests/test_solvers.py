"""Assertion-based solver tests.

These are the demos from Tests.py rewritten as real, headless regression tests:
same problem setups, but asserting on physical/mathematical invariants instead
of producing a plot for a human to eyeball.
"""
import numpy as np

from fem.numerics import bump_function
from fem.boundary import BoundaryConditions
from fem.solver import Solver, Equation


def test_heat_conserves_mean_temperature(make_unit_square):
    """Backward Euler with no-flux (natural) boundaries conserves total heat.

    Because K annihilates constants (K @ 1 = 0), 1^T M u is invariant under the
    scheme (M + dt K) u_{n+1} = M u_n, so the mean temperature is constant.
    """
    femesh = make_unit_square(20)
    corner = femesh.vertices.max(axis=0)
    u0 = bump_function(femesh.vertices, corner, mag=50, size=0.3) + 300

    eq = Equation("heat", {"u_initial": u0.copy(), "iters": 5, "dt": 0.01}, dim=1)
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
    solution = Solver(femesh, Equation("projection", dim=1), bc).solve()

    u = solution.get_values("u")
    expected = np.array([linear_field(v)[0] for v in femesh.vertices])
    assert np.allclose(u, expected, atol=1e-8), "linear field not reproduced exactly"


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

    eq = Equation("linear_elastic", {"E": 200, "nu": 0.4}, dim=2)
    solution = Solver(femesh, eq, bc).solve()

    u = solution.get_values("u").reshape(-1, 2)
    assert np.all(np.isfinite(u)), "displacement field has non-finite entries"
    assert np.allclose(u[left], 0.0, atol=1e-10), "fixed edge moved"
    assert u[right, 0].mean() > 0, "right edge did not elongate in +x"
