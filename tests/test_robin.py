"""Robin boundary conditions, du/dn + kappa*u = g: one condition contributes to both
sides of the system, kappa*int u*v to the operator and int g*v to the load. These pin
the sign and magnitude of both.
"""
import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet, Robin
from fem.regions import everywhere, on_plane
from fem.equations import Poisson
from fem.solver import Solver


def test_constant_solution_is_reproduced_exactly(make_unit_square):
    """A patch test: -div grad u = 0 with du/dn + kappa*u = kappa*c on the whole boundary has
    the exact solution u == c. A wrong sign or coefficient on either side breaks it."""
    mesh = make_unit_square(10)
    c, kappa = 5.0, 2.0

    bc = BoundaryConditions()
    bc = bc + Robin(everywhere(), kappa=kappa, g=kappa * c)
    u = Solver(mesh, Poisson(source=0.0), bc).solve().u

    assert np.allclose(u, c, atol=1e-10), f"constant not reproduced: range {u.min()}..{u.max()}"


def test_large_kappa_approaches_the_dirichlet_limit(make_unit_square):
    """As kappa -> infinity the Robin solution converges to the u = 0 Dirichlet solution."""
    mesh = make_unit_square(12)
    source = 1.0

    bc_d = BoundaryConditions()
    bc_d = bc_d + Dirichlet(everywhere(), 0.0)
    u_dirichlet = Solver(mesh, Poisson(source=source), bc_d).solve().u

    gaps = []
    for kappa in (10.0, 100.0, 1000.0):
        bc_r = BoundaryConditions()
        bc_r = bc_r + Robin(everywhere(), kappa=kappa, g=0.0)
        u_robin = Solver(mesh, Poisson(source=source), bc_r).solve().u
        gaps.append(float(np.linalg.norm(u_robin - u_dirichlet)))

    assert gaps[0] > gaps[1] > gaps[2], f"gap did not shrink with kappa: {gaps}"
    assert gaps[-1] < 0.02 * gaps[0], f"kappa=1000 still far from Dirichlet: {gaps}"


def test_robin_on_one_edge_pins_only_that_edge(make_unit_square):
    """Region restriction: a Robin condition on the left edge alone (large kappa,
    g = 0) drives the left edge toward 0 while the rest of the boundary stays
    natural (insulated). If the facet mask leaked, other edges would be pinned too.
    """
    mesh = make_unit_square(12)

    bc = BoundaryConditions()
    bc = bc + Robin(on_plane(0, 0.0), kappa=1000.0, g=0.0)
    u = Solver(mesh, Poisson(source=1.0), bc).solve().u

    bidx = mesh.boundary_idxs
    left = bidx[np.isclose(mesh.vertices[bidx, 0], 0.0)]
    right = bidx[np.isclose(mesh.vertices[bidx, 0], 1.0)]

    assert np.abs(u[left]).max() < 0.02, "left edge not pinned toward 0"
    # The insulated right edge is free to heat up under the source.
    assert u[right].min() > 0.1, "right edge should be well above zero"
