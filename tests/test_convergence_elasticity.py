"""MMS validation of linear elasticity in 2D and 3D.

The solver assembles K from B^T D B and solves K u = M f, the weak form of

    -div(sigma(u)) = f,     sigma = 2 mu eps(u) + lambda tr(eps(u)) I

with a manufactured displacement that vanishes on the boundary:

    2D:  u = (sin(pi x) sin(pi y), 0)
    3D:  u = (sin(pi x) sin(pi y) sin(pi z), 0, 0)

Only the first component is nonzero, but the shear terms of sigma make every
component of f nonzero, so this exercises the coupled vector path. P1, so O(h^2).
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.backends import IterativeBackend
from fem.materials import Enu_to_Lame
from fem.mesh.structured import create_box_mesh, create_rect_mesh
from fem.regions import everywhere
from fem.equations import Elasticity
from fem.solver import Solver

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
    solver = Solver(mesh, Elasticity(E=E, nu=NU, source=source), bc)
    solution = solver.solve()

    exact = np.zeros((len(mesh.vertices), 2))
    exact[:, 0] = np.sin(PI * mesh.vertices[:, 0]) * np.sin(PI * mesh.vertices[:, 1])
    return 1.0 / (n - 1), _l2_error(solver.space, solution.u, exact)


@pytest.fixture(scope='module')
def convergence_2d():
    return [_solve_2d(n) for n in (9, 17, 33)]


def test_2d_second_order(convergence_2d):
    """The error falls monotonically under refinement, at O(h^2)."""
    errors = [e for _, e in convergence_2d]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:])), errors
    orders = _observed_orders(convergence_2d)
    assert all(1.7 < p < 2.3 for p in orders), f'expected ~2nd order, got {orders}'


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
    # AMG-preconditioned CG, not the direct factorization: it solves the same SPD
    # system (proven equivalent in test_linalg) but stays cheap on the fine meshes
    # this sequence needs, and it is what the convergence measures -- the assembly --
    # regardless of how the block is solved.
    solver = Solver(mesh, Elasticity(E=E, nu=NU, source=source), bc, backend=IterativeBackend())
    solution = solver.solve()

    v = mesh.vertices
    exact = np.zeros((len(v), 3))
    exact[:, 0] = np.sin(PI * v[:, 0]) * np.sin(PI * v[:, 1]) * np.sin(PI * v[:, 2])
    return 1.0 / (n - 1), _l2_error(solver.space, solution.u, exact)


@pytest.fixture(scope='module')
def convergence_3d():
    # h = 1/8, 1/12, 1/16, 1/20, 1/28.
    #
    # The coarse end is dropped. Kuhn tets are distorted enough that
    # the error constant is large, so h = 1/4 and 1/6 are still pre-asymptotic
    # (they read 1.46 and 1.69) and including them would force a weaker assertion
    # on the whole sequence. Starting at h = 1/8 is not cherry-picking the answer:
    # it is declining to measure an asymptotic rate outside the asymptotic regime.
    #
    # The fine end (n=29) is what AMG-CG buys: a direct n=29 solve is too slow to
    # keep here, but AMG makes it cheap, and it is where the observed order finally
    # arrives near 2 rather than merely climbing toward it.
    return [_solve_3d(n) for n in (9, 13, 17, 21, 29)]


def test_3d_second_order(convergence_3d):
    """The error falls monotonically under refinement, inside the O(h^2) band the 2D
    case asserts: observed orders 1.82, 1.90, 1.94, 1.96."""
    errors = [e for _, e in convergence_3d]
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:])), errors
    orders = _observed_orders(convergence_3d)
    assert all(1.7 < p < 2.3 for p in orders), f'expected ~2nd order, got {orders}'


def test_3d_order_climbs_to_two(convergence_3d):
    """Inside the band is necessary but not sufficient: a defect degrading the rate to
    a constant 1.8 would pass it. The order must climb monotonically under refinement
    (a pre-asymptotic reading of a second-order method, not a lower-order one) and the
    finest pair, which AMG-CG affords, must arrive near 2 rather than merely approach it."""
    orders = _observed_orders(convergence_3d)
    assert all(fine > coarse for coarse, fine in zip(orders, orders[1:])), orders
    assert orders[-1] > 1.95, f'finest order did not reach the asymptotic rate: {orders}'
