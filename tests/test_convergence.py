"""MMS validation of the Poisson solver.

The manufactured solution, its forcing, and the study live in `examples/mms.py`, so
the `convergence` demo draws what these tests assert: the error falls, at the O(h^2)
rate P1 promises, and the finest mesh is accurate in absolute terms.
"""
import numpy as np
import pytest

from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.physics.equations import Poisson
from fem.regions import everywhere
from mms import ConvergenceStudy, solve_poisson_mms


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
    """The H1 seminorm (the gradient error) converges at O(h) for P1, one order below L2.
    A subtly wrong grad_phi degrades this rate while the L2 error can still look right."""
    hs = [s.h for s in convergence_data]
    errors = [s.h1_error for s in convergence_data]
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    # P1 gives order 1 in the H1 seminorm. Banded tighter than the L2 order-2 test
    # (0.9-1.1, not the +/-0.3 the L2 rate needs): the gradient error is cleanly O(h)
    # here with essentially no pre-asymptotic drift -- observed 1.00 at both pairs,
    # where L2 still climbs 1.97 -> 1.99 -- so the slack buys nothing and a tight band
    # catches a subtly wrong grad_phi that degrades the rate.
    for p in orders:
        assert 0.9 < p < 1.1, f"expected ~1st order in H1, got orders {orders}"


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
    """Homogeneous Dirichlet data is imposed exactly, so the boundary error is zero."""
    solve = solve_poisson_mms(11)
    assert np.allclose(solve.pointwise_error[solve.mesh.boundary_idxs], 0.0)
    assert np.abs(solve.pointwise_error).max() > 0.0


def test_p1_reproduces_a_linear_solution_exactly(make_unit_square):
    """The patch test. A linear field lies in the P1 space, so with its trace as the
    Dirichlet data and no source the Galerkin solution is that field at every node, to
    round-off. The rates above say the error shrinks; this says the discretisation is
    consistent at the one place it can be exact."""
    mesh = make_unit_square(7)

    def exact(p):
        return 1.0 + 2.0 * p[:, 0] - 3.0 * p[:, 1]

    bc = Conditions(Dirichlet(everywhere(), exact))
    solution = Poisson().problem(mesh, bc).solve()
    np.testing.assert_allclose(solution.dofs, exact(mesh.vertices), atol=1e-12)
