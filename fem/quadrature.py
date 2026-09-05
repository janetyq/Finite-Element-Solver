"""Quadrature rules on the reference simplex: points, weights, exactness degree.

A rule estimates an integral over an element by sampling the integrand at a few
reference points and summing weighted values. `degree` is the highest total
polynomial degree the rule integrates exactly: the property assembly selects a
rule by, and the property `tests/test_quadrature.py` certifies against the closed
form for a monomial over a simplex.

Weights sum to the reference simplex measure `1/d!`, so a physical integral is
`sum_q weight_q * f(x_q) * |det J|`. The points are in the element's own reference
coordinates: node 0 at the origin and the remaining nodes at the unit basis vectors,
the frame `LinearElement._dshape` is written in.
"""
from dataclasses import dataclass

import numpy as np

from fem.typing import FloatArray


@dataclass(frozen=True)
class QuadratureRule:
    '''Reference-simplex sample points, their weights, and the degree they integrate.

    `points` is `(n_points, reference_dim)` in the reference simplex's own
    coordinates: each coordinate is the barycentric weight on one corner, with
    corner 0 taking the remaining `1 - sum`, so `(1/3, 1/3)` is the triangle's
    centroid. `weights` is `(n_points,)` and sums to the reference measure `1/d!`,
    the measure the constant 1 integrates to. A rule of degree `p` integrates every
    polynomial of total degree <= p over the reference simplex exactly.
    '''
    points: FloatArray
    weights: FloatArray
    degree: int

    @property
    def n_points(self) -> int:
        return len(self.weights)


# Key is the simplex dimension (1 line, 2 triangle, 3 tet); value is that simplex's
# rules ordered cheapest (lowest degree, fewest points) first, so `quadrature_rule`
# returns the first entry meeting the requested degree. Each constant is certified by
# tests/test_quadrature.py (monomial exactness up to `degree`, and weights summing
# to the reference measure), so a mistyped entry fails there. Higher-degree rules are added when a form or element
# first needs one; the linear-simplex stiffness needs only degree 1.
_RULES: dict[int, list[QuadratureRule]] = {
    1: [  # reference line [0, 1], measure 1
        QuadratureRule(np.array([[0.5]]), np.array([1.0]), degree=1),
        # 2-point Gauss-Legendre mapped to [0, 1]: exact to degree 3.
        QuadratureRule(
            np.array([[0.5 - 0.5 / np.sqrt(3.0)], [0.5 + 0.5 / np.sqrt(3.0)]]),
            np.array([0.5, 0.5]), degree=3),
        # 3-point Gauss-Legendre mapped to [0, 1]: exact to degree 5, for the P2
        # boundary line's mass and traction integrals (phi_i phi_j is degree 4).
        QuadratureRule(
            np.array([[0.5 - 0.5 * np.sqrt(3 / 5)], [0.5], [0.5 + 0.5 * np.sqrt(3 / 5)]]),
            np.array([5 / 18, 8 / 18, 5 / 18]), degree=5),
    ],
    2: [  # reference triangle, measure 1/2
        QuadratureRule(np.array([[1 / 3, 1 / 3]]), np.array([0.5]), degree=1),
        # Strang three-point interior rule, exact to degree 2: the P1 mass and P2
        # stiffness rule.
        QuadratureRule(
            np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]),
            np.full(3, 1 / 6), degree=2),
        # Dunavant six-point rule, exact to degree 4: the P2 mass rule (phi_i phi_j
        # is degree 4) and the higher-degree load. Two three-point orbits; the
        # barycentric weights sum to 1 and are halved to the reference measure 1/2.
        QuadratureRule(
            np.array([
                [0.44594849091596489, 0.44594849091596489],
                [0.10810301816807022, 0.44594849091596489],
                [0.44594849091596489, 0.10810301816807022],
                [0.09157621350977074, 0.09157621350977074],
                [0.81684757298045851, 0.09157621350977074],
                [0.09157621350977074, 0.81684757298045851],
            ]),
            np.array([
                0.11169079483900573, 0.11169079483900573, 0.11169079483900573,
                0.05497587182766094, 0.05497587182766094, 0.05497587182766094,
            ]),
            degree=4),
    ],
    3: [  # reference tetrahedron, measure 1/6
        QuadratureRule(np.array([[1 / 4, 1 / 4, 1 / 4]]), np.array([1 / 6]), degree=1),
        # Keast four-point rule, exact to degree 2.
        QuadratureRule(
            np.array([
                [0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
                [0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
                [0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
                [0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
            ]),
            np.full(4, 1 / 24), degree=2),
        # 14-point rule, exact to degree 5: what a P2 tet needs, since `phi_i phi_j` in
        # its mass matrix is degree 4 and no cheaper tabulated rule reaches it with
        # positive weights (the 11-point Keast degree-4 rule has a negative one, which
        # can cost the mass matrix its definiteness). Three symmetric orbits in
        # barycentric coordinates: two of four points, permutations of (b, a, a, a),
        # and one of six, permutations of (a, a, b, b). Corner 0 takes the remaining
        # barycentric weight, so only the last three coordinates are listed. The
        # barycentric weights sum to 1 and are scaled to the reference measure 1/6.
        QuadratureRule(
            np.array([
                [0.3108859192633005, 0.3108859192633005, 0.3108859192633005],
                [0.0673422422100983, 0.3108859192633005, 0.3108859192633005],
                [0.3108859192633005, 0.0673422422100983, 0.3108859192633005],
                [0.3108859192633005, 0.3108859192633005, 0.0673422422100983],
                [0.0927352503108912, 0.0927352503108912, 0.0927352503108912],
                [0.7217942490673264, 0.0927352503108912, 0.0927352503108912],
                [0.0927352503108912, 0.7217942490673264, 0.0927352503108912],
                [0.0927352503108912, 0.0927352503108912, 0.7217942490673264],
                [0.0455037041256497, 0.4544962958743503, 0.4544962958743503],
                [0.4544962958743503, 0.0455037041256497, 0.4544962958743503],
                [0.4544962958743503, 0.4544962958743503, 0.0455037041256497],
                [0.0455037041256497, 0.0455037041256497, 0.4544962958743503],
                [0.0455037041256497, 0.4544962958743503, 0.0455037041256497],
                [0.4544962958743503, 0.0455037041256497, 0.0455037041256497],
            ]),
            np.array([
                0.1126879257180162, 0.1126879257180162, 0.1126879257180162,
                0.1126879257180162, 0.0734930431163619, 0.0734930431163619,
                0.0734930431163619, 0.0734930431163619, 0.0425460207770812,
                0.0425460207770812, 0.0425460207770812, 0.0425460207770812,
                0.0425460207770812, 0.0425460207770812,
            ]) / 6.0,
            degree=5),
    ],
}


def quadrature_rule(reference_dim: int, min_degree: int) -> QuadratureRule:
    '''The cheapest tabulated rule on `reference_dim` exact to at least `min_degree`.'''
    for rule in _RULES.get(reference_dim, []):
        if rule.degree >= min_degree:
            return rule
    raise NotImplementedError(
        f'no quadrature rule of degree >= {min_degree} for reference_dim={reference_dim}'
    )
