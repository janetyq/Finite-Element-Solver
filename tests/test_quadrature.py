"""Quadrature rules and the element sampling API, certified against the closed form for
a monomial over a simplex, `prod(a_i!) / (d + sum(a_i))!`.
"""
import math

import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
    QuadraticLineElement,
    QuadraticTetrahedralElement,
    QuadraticTriangleElement,
)
from fem.quadrature import _RULES, quadrature_rule

ELEMENTS = [LinearLineElement, LinearTriangleElement, LinearTetrahedralElement]

# Reference-node coordinates per element, in the element's own node ordering. The
# nodal test checks each hat is 1 at its own node and 0 at the others.
REFERENCE_NODES = {
    LinearLineElement: [[0.0], [1.0]],
    LinearTriangleElement: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    LinearTetrahedralElement: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    QuadraticLineElement: [[0.0], [1.0], [0.5]],
    QuadraticTriangleElement: [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],   # corners
        [0.5, 0.5], [0.0, 0.5], [0.5, 0.0],   # m12, m02, m01
    ],
    # Spelled out rather than read off `EDGE_NODES`, so the ten-node ordering is stated
    # independently of the element that claims it.
    QuadraticTetrahedralElement: [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],                 # corners
        [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],          # m01, m12, m02
        [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],          # m03, m13, m23
    ],
}


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
    """The degree claim is tight: a rule misses at least one monomial one degree past it."""
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


@pytest.mark.parametrize("element", list(REFERENCE_NODES))
def test_shape_functions_are_nodal(element):
    """1 at their own node, 0 at the others, for linear and quadratic elements alike."""
    nodes = np.asarray(REFERENCE_NODES[element], dtype=float)
    np.testing.assert_allclose(element.shape_values(nodes), np.eye(element.N), atol=1e-14)


@pytest.mark.parametrize("element", list(REFERENCE_NODES))
def test_shape_functions_partition_unity(element):
    """The hats sum to 1 everywhere (a constant field is represented exactly), so
    their gradients sum to zero (a constant field has no gradient)."""
    rule = element.quadrature(2)
    values = element.shape_values(rule.points)
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    gradients = element.shape_gradients(rule.points)
    np.testing.assert_allclose(gradients.sum(axis=1), 0.0, atol=1e-14)
