"""MMS for variable-coefficient diffusion: -div(kappa(x) grad u) = f.

The Phase 2 capability -- a coefficient and a source that vary *within* an element,
sampled at the quadrature points -- checked the only honest way: by confirming the
solve still recovers a known field at the O(h^2) rate P1 promises. A
constant-coefficient assembly cannot represent this problem at all, so a passing
rate here is evidence the quadrature sampling on both the operator and the load is
correct, not merely present.
"""
import pytest

from fem.convergence import ConvergenceStudy, variable_coefficient_convergence


@pytest.fixture(scope="module")
def study():
    return ConvergenceStudy.from_solves(variable_coefficient_convergence((11, 21, 41)))


def test_error_decreases_monotonically(study):
    for coarse, fine in zip(study.error[:-1], study.error[1:]):
        assert fine < coarse, f"error grew under refinement: {study.error}"


def test_second_order_convergence(study):
    for p in study.orders:
        assert 1.7 < p < 2.3, f"expected ~2nd order, got orders {study.orders}"


def test_absolute_accuracy_on_fine_mesh(study):
    assert study.error[-1] < 1e-2
