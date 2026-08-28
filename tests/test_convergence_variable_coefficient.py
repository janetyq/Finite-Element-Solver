"""MMS for variable-coefficient diffusion: -div(kappa(x) grad u) = f.

A coefficient and a source that vary within an element, sampled at the quadrature
points, checked by recovering a known field at the O(h^2) rate P1 promises.
"""
import pytest

from mms import ConvergenceStudy, variable_coefficient_convergence


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
