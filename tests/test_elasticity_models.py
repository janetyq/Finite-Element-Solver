"""How the two elasticity paths relate to each other.

The package solves elasticity twice, along two independent axes:

    strain measure   Green-Lagrange  E = 1/2 (F^T F - I)     exact
                     small strain    eps = 1/2 (grad u + grad u^T)

    method           direct assembly  K u = b, one linear solve
                     energy minimization  Newton on grad(Pi) = 0

`Solver` occupies (small strain, direct); `EnergySolver` occupies
(Green-Lagrange, energy). Because they sit on the diagonal, switching solver
also switches physics, and neither difference can be observed alone. These tests
pin both relationships so the two comments that used to carry this knowledge --
"does not exactly match linear elastic solve for larger deformations" and "not
the linear approx ... so iterative solver does not converge in 1 iteration" --
are executable claims instead of prose.
"""
import numpy as np

from fem.boundary import BoundaryConditions, BCType
from fem.regions import on_plane
from fem.solver import LinearElastic
from fem.energy_solver import EnergySolver


def _stretched_square(make_unit_square, stretch=0.1, n=8):
    """Unit square, left edge pinned, right edge displaced by `stretch` in x.

    Displacement-driven with no body force and no traction, which is the only
    setup the two solvers can be compared on: EnergySolver builds no load
    vector, so `Solver` must see b = 0 for the problems to coincide.
    """
    mesh = make_unit_square(n)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [stretch, 0])
    return mesh, bc


def test_energy_solver_matches_recorded_solution(make_unit_square):
    """Regression pin on the St Venant-Kirchhoff answer.

    The relationship tests below fix how this model compares to small strain,
    but nothing else pins its absolute values, so a change in the energy
    density or its derivatives could shift both sides together and go unseen.
    Values recorded from the implementation, not derived independently -- this
    catches drift, it does not prove correctness.
    """
    mesh, bc = _stretched_square(make_unit_square)
    solver = EnergySolver(mesh, LinearElastic(E=200, nu=0.4), bc, verbose=False)
    u = solver.solve().get_values("u")

    np.testing.assert_allclose(np.linalg.norm(u), 0.503442620332, rtol=1e-9)
    np.testing.assert_allclose(u.max(), 0.1, rtol=1e-12)
    np.testing.assert_allclose(u.min(), -0.037995668257, rtol=1e-9)
    np.testing.assert_allclose(solver.energy(u.copy()), 1.590561321584, rtol=1e-9)
