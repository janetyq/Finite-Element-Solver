"""Curved (isoparametric) boundary elements: the geometry fidelity they deliver.

A straight P2 element approximates a curved boundary by a chord, so the meshed area
is too small by O(h^2). An isoparametric element places its boundary edge node on the
true curve, so the boundary nodes land on the rim to machine precision, the meshed
area converges at the element's order (~h^3), the isoparametric Poisson solve keeps
the P2 L2 rate, and the curved mass matrix integrates the true area.
"""
import numpy as np
import pytest

from mms import (
    ANNULUS_INNER,
    ANNULUS_OUTER,
    ConvergenceStudy,
    annulus_convergence,
    annulus_mesh,
)
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.space import FunctionSpace

RESOLUTIONS = (5, 9, 17, 33)
TRUE_AREA = np.pi * (ANNULUS_OUTER**2 - ANNULUS_INNER**2)


def test_annulus_rims_carry_their_circles():
    mesh = annulus_mesh(1.0, 2.0, n_radial=4, n_theta=12)
    assert mesh.boundary_curves is not None
    radii = {round(float(np.hypot(*mesh.vertices[f[0]])), 6) for f in mesh.boundary}
    assert radii == {1.0, 2.0}
    assert all(curve is not None for curve in mesh.boundary_curves)


def _area_study(element_type) -> ConvergenceStudy:
    """The meshed area's error against the true annulus, per resolution."""
    hs, errors = [], []
    for n in RESOLUTIONS:
        mesh = annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, n, 4 * n)
        space = FunctionSpace(mesh, element_type, n_components=1)
        errors.append(abs(space.geometry.total_volume - TRUE_AREA))
        hs.append(1.0 / (n - 1))
    return ConvergenceStudy(np.array(hs), np.array(errors))


def test_curved_boundary_nodes_lie_on_the_true_rim():
    """The defining property: an isoparametric element's boundary edge nodes sit on the
    true circle, where a straight element's chord midpoints fall short by the sagitta."""
    mesh = annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, 9, 36)

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
    straight = _area_study(QuadraticTriangleElement)
    curved = _area_study(IsoparametricTriangleElement)

    for p in straight.orders:
        assert 1.5 < p < 2.3, f"expected ~2nd order area for straight, got {straight.orders}"
    assert curved.orders.min() > 2.7, f"expected >2 for curved, got {curved.orders}"
    assert np.all(curved.error < straight.error / 20), (
        f"curved area error not far below straight: {curved.error} vs {straight.error}")


def test_isoparametric_solve_keeps_the_p2_rate():
    """The curved geometry map must not degrade the solver: the isoparametric Poisson
    solve on the annulus still converges at the P2 rate in L2."""
    study = ConvergenceStudy.from_solves(
        annulus_convergence(RESOLUTIONS, IsoparametricTriangleElement))
    for coarse, fine in zip(study.error[:-1], study.error[1:]):
        assert fine < coarse, f"error grew under refinement: {study.error}"
    assert study.fitted_order > 2.7, f"expected ~3rd order, got orders {study.orders}"


def test_isoparametric_solve_keeps_the_h1_rate():
    """The gradient error is the sharper probe of the isoparametric geometry map: a wrong
    curved-element Jacobian degrades the O(h^2) H1 seminorm rate before it shows in the L2
    error above. Measured against the closed-form gradient (never the assembled K), and held
    to a two-sided per-pair band. Observed orders 1.78, 1.88, 1.94, a pre-asymptotic climb
    toward two that the band's lower edge allows."""
    solves = annulus_convergence(RESOLUTIONS, IsoparametricTriangleElement)
    study = ConvergenceStudy(np.array([s.h for s in solves]),
                             np.array([s.h1_error for s in solves]))
    for coarse, fine in zip(study.error[:-1], study.error[1:]):
        assert fine < coarse, f"gradient error grew under refinement: {study.error}"
    for p in study.orders:
        assert 1.6 < p < 2.2, f"expected ~O(h^2) H1 seminorm, got orders {study.orders}"


def test_curved_mass_matrix_integrates_the_curved_area():
    """The consistent mass matrix sums to the domain measure (the P2 hats are a
    partition of unity), a direct check that the curved MassForm integrates over the
    true curved area rather than the inscribed polygon's."""
    mesh = annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, 17, 68)
    space = FunctionSpace(mesh, IsoparametricTriangleElement, n_components=1)
    assert space.mass_matrix.sum() == pytest.approx(TRUE_AREA, rel=1e-3)
