"""Quadrature rules and the element sampling API.

The rules are the foundation the higher-order assembly stands on, so they are
certified directly against the closed form for a monomial integrated over a
simplex, `prod(a_i!) / (d + sum(a_i))!`. A rule of degree p must reproduce that
exactly for every monomial of total degree <= p; a mistyped point or weight fails
here rather than biasing every downstream integral.
"""
import math

import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.quadrature import _RULES, quadrature_rule

ELEMENTS = [LinearLineElement, LinearTriangleElement, LinearTetrahedralElement]


def _monomials(dim: int, total_degree: int):
    '''Every exponent tuple of length `dim` whose entries sum to `total_degree`.'''
    if dim == 1:
        yield (total_degree,)
        return
    for first in range(total_degree + 1):
        for rest in _monomials(dim - 1, total_degree - first):
            yield (first, *rest)


def _exact_simplex_integral(powers: tuple[int, ...], dim: int) -> float:
    '''Closed form: integral of prod(x_i ** a_i) over the reference simplex.'''
    numerator = math.prod(math.factorial(p) for p in powers)
    return numerator / math.factorial(dim + sum(powers))


@pytest.mark.parametrize("dim", sorted(_RULES))
def test_weights_sum_to_the_reference_measure(dim):
    measure = 1.0 / math.factorial(dim)
    for rule in _RULES[dim]:
        assert rule.weights.sum() == pytest.approx(measure)


@pytest.mark.parametrize("dim", sorted(_RULES))
def test_rules_integrate_monomials_up_to_their_degree(dim):
    for rule in _RULES[dim]:
        points = rule.points
        for total_degree in range(rule.degree + 1):
            for powers in _monomials(dim, total_degree):
                sampled = np.prod(points ** np.array(powers), axis=1)
                approx = float(rule.weights @ sampled)
                exact = _exact_simplex_integral(powers, dim)
                assert approx == pytest.approx(exact, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("dim", sorted(_RULES))
def test_a_rule_fails_one_degree_above_its_own(dim):
    """Sharpness: the degree claim is tight, not merely a lower bound. A rule should
    miss at least one monomial one degree past what it advertises -- otherwise the
    `degree` field understates it and rule selection wastes points."""
    for rule in _RULES[dim]:
        beyond = list(_monomials(dim, rule.degree + 1))
        errors = [
            abs(float(rule.weights @ np.prod(rule.points ** np.array(p), axis=1))
                - _exact_simplex_integral(p, dim))
            for p in beyond
        ]
        assert max(errors) > 1e-12


def test_rule_selection_returns_the_cheapest_adequate_rule():
    assert quadrature_rule(2, 1).n_points == 1
    assert quadrature_rule(2, 2).n_points == 3
    with pytest.raises(NotImplementedError):
        quadrature_rule(2, 99)


@pytest.mark.parametrize("element", ELEMENTS)
def test_linear_shape_functions_are_nodal(element):
    """1 at their own node, 0 at the others -- the property that makes a DOF the
    value at its node. Reference nodes are the origin and the unit basis vectors."""
    d = element.reference_dim()
    reference_nodes = np.vstack([np.zeros(d), np.eye(d)])
    np.testing.assert_allclose(element.shape_values(reference_nodes), np.eye(element.N),
                               atol=1e-15)


@pytest.mark.parametrize("element", ELEMENTS)
def test_linear_shape_functions_partition_unity(element):
    """The hats sum to 1 everywhere, so a constant field is represented exactly."""
    rule = element.quadrature(1)
    values = element.shape_values(rule.points)
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
