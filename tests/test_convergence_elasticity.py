"""Numerical validation of linear elasticity via the Method of Manufactured
Solutions, in 2D and 3D.

The 3D case is the checkpoint for the `FieldShape` work: it is the first thing
the old architecture could not express, since `LinearElastic.dim` was a class
constant of 2. The 2D case is its control, and the regression net for the vector
mass matrix -- a body force is only correct if every component of it is.

The solver assembles K from B^T D B and solves  K u = M f, the weak form of

    -div(sigma(u)) = f,     sigma = 2 mu eps(u) + lambda tr(eps(u)) I

We manufacture a displacement that vanishes on the whole boundary, so
homogeneous Dirichlet conditions are exact:

    2D:  u = (sin(pi x) sin(pi y), 0)
    3D:  u = (sin(pi x) sin(pi y) sin(pi z), 0, 0)

Only the first component of u is non-zero, but the off-diagonal shear terms of
sigma make every component of f non-zero -- which is what makes this exercise
the coupled vector path rather than a scalar solve in disguise.

Both use P1 elements, so the L2 error is O(h^2).
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.materials import Enu_to_Lame
from fem.mesh.generation import create_box_mesh, create_rect_mesh
from fem.regions import everywhere
from fem.solver import Solver, LinearElastic

E, NU = 200.0, 0.3
MU, LAMB = Enu_to_Lame(E, NU)
PI = np.pi


def _observed_orders(data):
    """Convergence order from successive (h, error) pairs: error ~ C h^p."""
    return [
        np.log(data[i][1] / data[i + 1][1]) / np.log(data[i][0] / data[i + 1][0])
        for i in range(len(data) - 1)
    ]


def _l2_error(space, u_h, u_exact):
    # ||e||_L2^2 = e^T M e, with M the space's vector mass matrix -- the scalar
    # one repeated per component -- so this is the true vector L2 norm, not just
    # component 0.
    error = (u_h.reshape(u_exact.shape) - u_exact).flatten()
    return float(np.sqrt(error @ space.mass_matrix @ error))


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------

def _solve_2d(n):
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    def source(p):
        x, y = p
        return [
            PI**2 * (3 * MU + LAMB) * np.sin(PI * x) * np.sin(PI * y),
            -(MU + LAMB) * PI**2 * np.cos(PI * x) * np.cos(PI * y),
        ]

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0])
    solver = Solver(mesh, LinearElastic(E=E, nu=NU, source=source), bc)
    solution = solver.solve()

    exact = np.zeros((len(mesh.vertices), 2))
    exact[:, 0] = np.sin(PI * mesh.vertices[:, 0]) * np.sin(PI * mesh.vertices[:, 1])
    return 1.0 / (n - 1), _l2_error(solver.space, solution.get_values('u'), exact)


@pytest.fixture(scope='module')
def convergence_2d():
    return [_solve_2d(n) for n in (9, 17, 33)]


def test_2d_error_decreases(convergence_2d):
    errors = [e for _, e in convergence_2d]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f'error grew under refinement: {errors}'


def test_2d_second_order(convergence_2d):
    orders = _observed_orders(convergence_2d)
    for p in orders:
        assert 1.7 < p < 2.3, f'expected ~2nd order, got {orders}'


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------

def _solve_3d(n):
    # No element type to state: Solver reads it off the connectivity.
    # n vertices per side, so h = 1/(n-1) and the mesh has 6(n-1)^3 tets.
    mesh = create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(n, n, n))

    def source(p):
        x, y, z = p
        return [
            PI**2 * (4 * MU + LAMB) * np.sin(PI * x) * np.sin(PI * y) * np.sin(PI * z),
            -(MU + LAMB) * PI**2 * np.cos(PI * x) * np.cos(PI * y) * np.sin(PI * z),
            -(MU + LAMB) * PI**2 * np.cos(PI * x) * np.sin(PI * y) * np.cos(PI * z),
        ]

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0, 0.0])
    solver = Solver(mesh, LinearElastic(E=E, nu=NU, source=source), bc)
    solution = solver.solve()

    v = mesh.vertices
    exact = np.zeros((len(v), 3))
    exact[:, 0] = np.sin(PI * v[:, 0]) * np.sin(PI * v[:, 1]) * np.sin(PI * v[:, 2])
    return 1.0 / (n - 1), _l2_error(solver.space, solution.get_values('u'), exact)


@pytest.fixture(scope='module')
def convergence_3d():
    # h = 1/8, 1/12, 1/16, 1/20.
    #
    # The coarse end is deliberately dropped. Kuhn tets are distorted enough that
    # the error constant is large, so h = 1/4 and 1/6 are still pre-asymptotic
    # (they read 1.46 and 1.69) and including them would force a weaker assertion
    # on the whole sequence. Starting at h = 1/8 is not cherry-picking the answer:
    # it is declining to measure an asymptotic rate outside the asymptotic regime.
    #
    # Batched assembly is what makes this affordable -- assembling n=21 took ~28s
    # and now takes under a second. The cost is back on the sparse factorization,
    # which is most of the ~14s that n=21 spends.
    return [_solve_3d(n) for n in (9, 13, 17, 21)]


def test_3d_error_decreases(convergence_3d):
    errors = [e for _, e in convergence_3d]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f'error grew under refinement: {errors}'


def test_3d_second_order(convergence_3d):
    """The same O(h^2) band the 2D case asserts -- observed orders 1.82, 1.90, 1.94."""
    orders = _observed_orders(convergence_3d)
    for p in orders:
        assert 1.7 < p < 2.3, f'expected ~2nd order, got {orders}'


def test_3d_order_climbs_toward_two(convergence_3d):
    """Still approaching 2 from below, rather than sitting at a plateau.

    Being inside the band is necessary but not sufficient: a defect that degraded
    the rate to a constant 1.8 would also pass the band check on these meshes.
    Monotone improvement under refinement is what distinguishes a pre-asymptotic
    reading of a second-order method from a genuinely lower-order one.
    """
    orders = _observed_orders(convergence_3d)
    for coarse, fine in zip(orders, orders[1:]):
        assert fine > coarse, f'order should climb toward 2, got {orders}'
