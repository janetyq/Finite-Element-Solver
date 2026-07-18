"""Numerical validation of the Poisson solver via the Method of Manufactured
Solutions (MMS).

The solver assembles the stiffness matrix K (the discrete -Laplacian) and the
mass matrix M, then solves  K u = M f  with homogeneous Dirichlet BCs -- i.e. the
weak form of  -div(grad u) = f.

We manufacture an exact solution and its forcing:

    u(x, y) = sin(pi x) sin(pi y)          (zero on the boundary of [0,1]^2)
    f(x, y) = -Laplacian(u) = 2 pi^2 sin(pi x) sin(pi y)

Then we solve on a sequence of uniformly refined meshes and check that the
discrete L2 error decreases at the theoretical rate for linear (P1) elements,
which is O(h^2).
"""
import numpy as np
import pytest

from utils.meshing import create_rect_mesh
from FEMesh import FEMesh
from BoundaryConditions import BoundaryConditions
from Solver import Solver, Equation


def _exact(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def _solve_poisson_mms(n):
    """Solve the manufactured Poisson problem on an n x n unit-square grid.

    Returns (h, l2_error) where h is the grid spacing and l2_error is the
    discrete L2 norm of (u_h - u_exact), computed with the mass matrix.
    """
    base = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    femesh = FEMesh(base.vertices, base.elements, base.boundary)

    equation = Equation("poisson", dim=1)
    bc = BoundaryConditions(femesh)
    bc.add("dirichlet", femesh.boundary_idxs, [0])
    bc.add_force(
        lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])]
    )

    solver = Solver(femesh, equation, bc)
    solution = solver.solve()
    u_h = solution.get_values("u")

    error = u_h - _exact(femesh.vertices)
    # ||e||_L2^2 = e^T M e  (M is the assembled dim=1 mass matrix)
    l2_error = np.sqrt(error @ femesh.M @ error)
    h = 1.0 / (n - 1)
    return h, l2_error


@pytest.fixture(scope="module")
def convergence_data():
    resolutions = [11, 21, 41]  # h = 0.1, 0.05, 0.025 (each halved)
    return [_solve_poisson_mms(n) for n in resolutions]


def test_error_decreases_monotonically(convergence_data):
    errors = [e for _, e in convergence_data]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f"error grew under refinement: {errors}"


def test_second_order_convergence(convergence_data):
    # Observed order p from successive (h, error) pairs:
    #   error ~ C h^p  =>  p = log(e1/e2) / log(h1/h2)
    hs = [h for h, _ in convergence_data]
    errors = [e for _, e in convergence_data]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    # P1 elements give order 2; allow a tolerance band for a structured mesh.
    for p in orders:
        assert 1.7 < p < 2.3, f"expected ~2nd order, got orders {orders}"


def test_absolute_accuracy_on_fine_mesh(convergence_data):
    # Sanity floor: the finest mesh should be reasonably accurate.
    _, finest_error = convergence_data[-1]
    assert finest_error < 1e-2
