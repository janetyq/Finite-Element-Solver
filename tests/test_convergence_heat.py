"""Temporal convergence of the heat time integrators.

The integrator is compared against the exact solution of the semi-discrete system,
u(t) = expm(-t M^{-1} K) u0 on the free DOFs, so the error is purely temporal. The
study lives in `examples/mms.py` as `theta_convergence`, so the `convergence` demo
draws what this asserts: theta = 1 (backward Euler) is first order; theta = 1/2
(Crank-Nicolson) is second.
"""
import pytest

from mms import theta_convergence


# Crank-Nicolson's steps are chosen to sit in the asymptotic band: coarser steps are
# pre-asymptotic (lambda dt not small), finer ones approach the roundoff floor of the expm.
@pytest.mark.parametrize('theta, step_counts, order, band', [
    (1.0, (2, 4, 8, 16), 1, (0.8, 1.3)),
    (0.5, (16, 32, 64), 2, (1.8, 2.3)),
], ids=['backward_euler', 'crank_nicolson'])
def test_theta_method_converges_at_its_order(theta, step_counts, order, band):
    study = theta_convergence(theta, step_counts)
    for coarse, fine in zip(study.error[:-1], study.error[1:]):
        assert fine < coarse, f'error grew under dt refinement: {study.error}'
    low, high = band
    for p in study.orders:
        assert low < p < high, f'expected order ~{order} in dt, got {study.orders}'
