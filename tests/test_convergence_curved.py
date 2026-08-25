"""Curved (isoparametric) boundary elements: the geometry fidelity they deliver.

A straight P2 element approximates a curved boundary by a chord, so the meshed area
is too small by O(h^2). An isoparametric element places its boundary edge node on the
true curve, so the boundary nodes land on the rim to machine precision, the meshed
area converges at the element's order (~h^3), the isoparametric Poisson solve keeps
the P2 L2 rate, and the curved mass matrix integrates the true area.
"""
import numpy as np
import pytest

from fem.convergence import (
    ANNULUS_INNER,
    ANNULUS_OUTER,
    ConvergenceStudy,
    annulus_convergence,
    create_annulus_mesh,
)
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.space import FunctionSpace

RESOLUTIONS = (5, 9, 17, 33)
TRUE_AREA = np.pi * (ANNULUS_OUTER**2 - ANNULUS_INNER**2)


def _observed_orders(hs, errors):
    hs, errors = np.asarray(hs), np.asarray(errors)
    return np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])


def _area_errors(element_type):
    hs, errors = [], []
    for n in RESOLUTIONS:
        mesh = create_annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, n, 4 * n)
        space = FunctionSpace(mesh, element_type, n_components=1)
        errors.append(abs(space.geometry.total_volume - TRUE_AREA))
        hs.append(1.0 / (n - 1))
    return np.array(hs), np.array(errors)


def test_curved_boundary_nodes_lie_on_the_true_rim():
    """The defining property: an isoparametric element's boundary edge nodes sit on the
    true circle, where a straight element's chord midpoints fall short by the sagitta."""
    mesh = create_annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, 9, 36)

    def max_rim_distance(element_type):
        space = FunctionSpace(mesh, element_type, n_components=1)
        boundary_idx = np.unique(space.boundary_nodes)
        coords = space.node_coords[boundary_idx]
        radius = np.hypot(coords[:, 0], coords[:, 1])
        return float(np.minimum(np.abs(radius - ANNULUS_INNER),
                                np.abs(radius - ANNULUS_OUTER)).max())

    assert max_rim_distance(IsoparametricTriangleElement) < 1e-12
    # The straight element is off by a resolvable amount: the chord sagitta.
    assert max_rim_distance(QuadraticTriangleElement) > 1e-3


def test_curved_area_is_higher_order_than_straight():
    """Domain area is pure geometry. The polygonal straight boundary gives an O(h^2)
    area error; the curved boundary reaches the element's own order, orders of magnitude
    smaller at every mesh."""
    _, straight = _area_errors(QuadraticTriangleElement)
    hs, curved = _area_errors(IsoparametricTriangleElement)

    straight_orders = _observed_orders(hs, straight)
    curved_orders = _observed_orders(hs, curved)

    for p in straight_orders:
        assert 1.5 < p < 2.3, f"expected ~2nd order area for straight, got {straight_orders}"
    assert curved_orders.min() > 2.7, f"expected >2 for curved, got {curved_orders}"
    assert np.all(curved < straight / 20), (
        f"curved area error not far below straight: {curved} vs {straight}")


def test_isoparametric_solve_keeps_the_p2_rate():
    """The curved geometry map must not degrade the solver: the isoparametric Poisson
    solve on the annulus still converges at the P2 rate in L2."""
    study = ConvergenceStudy.from_solves(
        annulus_convergence(RESOLUTIONS, IsoparametricTriangleElement))
    for coarse, fine in zip(study.error[:-1], study.error[1:]):
        assert fine < coarse, f"error grew under refinement: {study.error}"
    assert study.fitted_order > 2.7, f"expected ~3rd order, got orders {study.orders}"


def test_curved_mass_matrix_integrates_the_curved_area():
    """The consistent mass matrix sums to the domain measure (the P2 hats are a
    partition of unity), a direct check that the curved MassForm integrates over the
    true curved area rather than the inscribed polygon's."""
    mesh = create_annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, 17, 68)
    space = FunctionSpace(mesh, IsoparametricTriangleElement, n_components=1)
    assert space.mass_matrix.sum() == pytest.approx(TRUE_AREA, rel=1e-3)
