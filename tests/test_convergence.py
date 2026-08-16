"""Numerical validation of the Poisson solver via the Method of Manufactured
Solutions (MMS).

The solver assembles the stiffness matrix K (the discrete -Laplacian) and the
mass matrix M, then solves  K u = M f  with homogeneous Dirichlet BCs -- i.e. the
weak form of  -div(grad u) = f.

The manufactured solution, its forcing, and the study that refines against them
live in `fem/convergence.py`, so that the `convergence` demo draws exactly what
these tests assert rather than a second implementation that could drift from it.
What is here is the assertions: that the error falls, that it falls at the O(h^2)
rate P1 elements promise, and that the finest mesh is accurate in absolute terms.
"""
import numpy as np
import pytest

from fem.convergence import ConvergenceStudy, solve_poisson_mms


@pytest.fixture(scope="module")
def convergence_data():
    resolutions = [11, 21, 41]  # h = 0.1, 0.05, 0.025 (each halved)
    return [solve_poisson_mms(n) for n in resolutions]


def test_error_decreases_monotonically(convergence_data):
    errors = [s.l2_error for s in convergence_data]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f"error grew under refinement: {errors}"


def test_second_order_convergence(convergence_data):
    # Observed order p from successive (h, error) pairs:
    #   error ~ C h^p  =>  p = log(e1/e2) / log(h1/h2)
    hs = [s.h for s in convergence_data]
    errors = [s.l2_error for s in convergence_data]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    # P1 elements give order 2; allow a tolerance band for a structured mesh.
    for p in orders:
        assert 1.7 < p < 2.3, f"expected ~2nd order, got orders {orders}"


def test_first_order_h1_convergence(convergence_data):
    """The H1 seminorm -- the gradient error, integrated against the analytic gradient
    rather than read off the assembled K -- converges at O(h) for P1, one order below
    L2. It is the sharper probe of the stiffness the operator is built from: a subtly
    wrong grad_phi degrades this rate while the L2 error can still look almost right."""
    hs = [s.h for s in convergence_data]
    errors = [s.h1_error for s in convergence_data]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    # P1 elements give order 1 in the H1 seminorm; band it like the L2 order-2 test.
    for p in orders:
        assert 0.7 < p < 1.3, f"expected ~1st order in H1, got orders {orders}"


def test_absolute_accuracy_on_fine_mesh(convergence_data):
    # Sanity floor: the finest mesh should be reasonably accurate.
    finest_error = convergence_data[-1].l2_error
    assert finest_error < 1e-2


def test_observed_orders_recover_a_known_rate():
    """The arithmetic every claim above rests on, checked against data whose rate is
    exact by construction."""
    h = np.array([0.4, 0.2, 0.1])
    study = ConvergenceStudy(h, 3.0 * h**2)
    assert np.allclose(study.orders, 2.0)
    assert study.fitted_order == pytest.approx(2.0)


def test_the_error_is_interior():
    """Homogeneous Dirichlet data is imposed exactly, so nothing is left to be wrong on
    the boundary -- which is what the `convergence` demo's error field shows."""
    solve = solve_poisson_mms(11)
    assert np.allclose(solve.pointwise_error[solve.mesh.boundary_idxs], 0.0)
    assert np.abs(solve.pointwise_error).max() > 0.0
