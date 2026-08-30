"""MMS validation: the error falls under refinement at the rate each discretisation
promises.

The manufactured solutions, their forcings, and the studies live in `examples/mms.py`,
so the `convergence` demo draws what these tests assert. Each spatial study is one row
of `STUDIES`: what it solves, the order it promises, the band the observed order is
held to, and where one is worth stating, an absolute error the finest mesh must beat.
The three rate tests run over every row. Then the 3D elasticity sequence, the patch
tests (exact where the space can be exact), and the load comparison.
"""
from dataclasses import dataclass
from functools import cache
from typing import Callable

import numpy as np
import pytest

from fem.algebra.backends import IterativeBackend
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.physics.equations import LinearElastic, Poisson
from fem.physics.materials import Enu_to_Lame
from fem.regions import everywhere
from mms import (
    ELASTIC_E,
    ELASTIC_NU,
    ConvergenceStudy,
    elastic_convergence,
    elastic_p2_convergence,
    load_comparison_convergence,
    mixed_bc_convergence,
    poisson_convergence,
    poisson_p2_convergence,
    solve_poisson_mms,
    variable_coefficient_convergence,
)
from helpers import solved


# -- the spatial studies ------------------------------------------------------


@dataclass(frozen=True)
class Study:
    name: str
    build: Callable[[], ConvergenceStudy]   # cached, so the rate tests share one run
    order: int                              # the rate the discretisation promises
    band: tuple[float, float]               # what each observed per-pair order is held to
    floor: float | None = None              # an absolute error the finest mesh must beat


@cache
def _poisson_p1_solves():
    return poisson_convergence((11, 21, 41))    # h = 0.1, 0.05, 0.025


@cache
def poisson_p1_l2():
    return ConvergenceStudy.from_solves(_poisson_p1_solves())


@cache
def poisson_p1_h1():
    solves = _poisson_p1_solves()
    return ConvergenceStudy(np.array([s.h for s in solves]), np.array([s.h1_error for s in solves]))


@cache
def poisson_p2():
    return ConvergenceStudy.from_solves(poisson_p2_convergence((5, 9, 17, 33)))


@cache
def variable_coefficient():
    return ConvergenceStudy.from_solves(variable_coefficient_convergence((11, 21, 41)))


@cache
def elastic_p1():
    return ConvergenceStudy.from_solves(elastic_convergence((9, 17, 33)))


@cache
def elastic_p2():
    return ConvergenceStudy.from_solves(elastic_p2_convergence((5, 9, 17)))


@cache
def poisson_3d():
    # h = 1/8, 1/12, 1/16. The n=5 (h=1/4) coarse level is dropped as pre-asymptotic
    # on Kuhn tets (it reads 1.74), the same reasoning the 3D elasticity fixture states;
    # the three kept here are cleanly in band.
    return ConvergenceStudy.from_solves(poisson_convergence((9, 13, 17), dim=3))


@cache
def _mixed_bc_solves():
    return mixed_bc_convergence((11, 21, 41))    # h = 0.1, 0.05, 0.025


@cache
def mixed_bc_l2():
    return ConvergenceStudy.from_solves(_mixed_bc_solves())


@cache
def mixed_bc_h1():
    solves = _mixed_bc_solves()
    return ConvergenceStudy(np.array([s.h for s in solves]), np.array([s.h1_error for s in solves]))


STUDIES = [
    # P1 gives order 2 in L2; the band allows for a structured mesh's coarse end.
    Study('poisson_p1_l2', poisson_p1_l2, 2, (1.7, 2.3), floor=1e-2),
    # The H1 seminorm (the gradient error) is one order below L2, and banded tighter:
    # it is cleanly O(h) with no pre-asymptotic drift (observed 1.00 at both pairs,
    # where L2 still climbs 1.97 -> 1.99), so the slack buys nothing, and a subtly
    # wrong grad_phi degrades this rate while the L2 error can still look right.
    Study('poisson_p1_h1', poisson_p1_h1, 1, (0.9, 1.1)),
    # P2 is O(h^3). A wrong edge-node numbering, shape function, or unpinned boundary
    # edge node does not converge at the cubic rate. The floor is orders of magnitude
    # below what P1 reaches at the same spacing (~1e-3 by h = 1/40): faster, not
    # merely converging.
    Study('poisson_p2', poisson_p2, 3, (2.7, 3.3), floor=1e-5),
    # kappa(x) and f both vary within an element, so both sides of the solve go
    # through the quadrature layer a constant-coefficient assembly lacks.
    Study('variable_coefficient', variable_coefficient, 2, (1.7, 2.3), floor=1e-2),
    # The coupled vector path: only u_x is nonzero, but the shear terms of sigma make
    # both components of f nonzero.
    Study('elastic_p1', elastic_p1, 2, (1.7, 2.3)),
    # The vector P2 path: the node numbering under n_components = 2 and the coupled
    # operator, at the scalar P2 rate.
    Study('elastic_p2', elastic_p2, 3, (2.7, 3.3)),
    # The same scalar Poisson study in 3D, on a tetrahedral box: assembly and the P1
    # solve on tets, not only triangles. Observed orders 1.91, 1.95, climbing to two as
    # the 3D elasticity sequence does.
    Study('poisson_3d', poisson_3d, 2, (1.7, 2.3), floor=5e-3),
    # All-natural boundary: nonzero Neumann flux on three edges and a Robin condition on
    # the fourth, so the boundary-load quadrature and the Robin boundary-mass term enter
    # a rate for the first time. A boundary integral wrong by a factor breaks the O(h^2).
    Study('mixed_bc_l2', mixed_bc_l2, 2, (1.7, 2.3), floor=1e-3),
    # The gradient error of the same solve, O(h) and banded tight like the Poisson H1:
    # a wrong boundary flux shows here directly.
    Study('mixed_bc_h1', mixed_bc_h1, 1, (0.9, 1.1)),
]
FLOORED = [study for study in STUDIES if study.floor is not None]

each_study = pytest.mark.parametrize('study', STUDIES, ids=lambda s: s.name)


@each_study
def test_error_decreases_monotonically(study):
    error = study.build().error
    for coarse, fine in zip(error[:-1], error[1:]):
        assert fine < coarse, f'{study.name}: error grew under refinement: {error}'


@each_study
def test_converges_at_the_promised_order(study):
    # Observed order p from successive (h, error) pairs: error ~ C h^p, so
    # p = log(e1/e2) / log(h1/h2), the arithmetic `ConvergenceStudy.orders` carries.
    orders = study.build().orders
    low, high = study.band
    for p in orders:
        assert low < p < high, f'{study.name}: expected order ~{study.order}, got {orders}'


@pytest.mark.parametrize('study', FLOORED, ids=lambda s: s.name)
def test_finest_mesh_beats_the_absolute_floor(study):
    finest = study.build().error[-1]
    assert finest < study.floor, f'{study.name}: finest error {finest:.3e} is not below {study.floor}'


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


# -- 3D elasticity -------------------------------------------------------------


@pytest.fixture(scope='module')
def elastic_3d():
    # h = 1/8, 1/12, 1/16, 1/20, 1/28.
    #
    # The coarse end is dropped. Kuhn tets are distorted enough that the error constant
    # is large, so h = 1/4 and 1/6 are still pre-asymptotic (they read 1.46 and 1.69)
    # and including them would force a weaker assertion on the whole sequence. Starting
    # at h = 1/8 is not cherry-picking the answer: it is declining to measure an
    # asymptotic rate outside the asymptotic regime.
    #
    # AMG-preconditioned CG, not the direct factorization: it solves the same SPD system
    # (test_backends.py proves them equivalent) but stays cheap on the fine meshes this
    # sequence needs, and it is what the convergence measures, the assembly, regardless
    # of how the block is solved. The fine end (n=29) is what it buys: a direct n=29
    # solve is too slow to keep here, and it is where the observed order finally arrives
    # near 2 rather than merely climbing toward it.
    return ConvergenceStudy.from_solves(
        elastic_convergence((9, 13, 17, 21, 29), dim=3, backend=IterativeBackend()))


def test_3d_elasticity_is_second_order(elastic_3d):
    """The error falls monotonically under refinement, inside the O(h^2) band the 2D
    case asserts: observed orders 1.79, 1.88, 1.93, 1.96."""
    for coarse, fine in zip(elastic_3d.error[:-1], elastic_3d.error[1:]):
        assert fine < coarse, f'error grew under refinement: {elastic_3d.error}'
    assert all(1.7 < p < 2.3 for p in elastic_3d.orders), f'expected ~2nd order, got {elastic_3d.orders}'


def test_3d_order_climbs_to_two(elastic_3d):
    """Inside the band is necessary but not sufficient: a defect degrading the rate to
    a constant 1.8 would pass it. The order must climb monotonically under refinement
    (a pre-asymptotic reading of a second-order method, not a lower-order one) and the
    finest pair, which AMG-CG affords, must arrive near 2 rather than merely approach it."""
    orders = elastic_3d.orders
    assert all(fine > coarse for coarse, fine in zip(orders[:-1], orders[1:])), orders
    assert orders[-1] > 1.95, f'finest order did not reach the asymptotic rate: {orders}'


# -- patch tests: exact where the space can be exact --------------------------


def test_p1_reproduces_a_linear_solution_exactly(make_unit_square):
    """A linear field lies in the P1 space, so with its trace as the Dirichlet data and
    no source the Galerkin solution is that field at every node, to round-off. The rates
    above say the error shrinks; this says the discretisation is consistent at the one
    place it can be exact."""
    mesh = make_unit_square(7)

    def exact(p):
        return 1.0 + 2.0 * p[:, 0] - 3.0 * p[:, 1]

    bc = Conditions(Dirichlet(everywhere(), exact))
    _, solution = solved(Poisson(), mesh, bc)
    np.testing.assert_allclose(solution.dofs, exact(mesh.vertices), atol=1e-12)


def test_p1_reproduces_a_linear_displacement_exactly(make_unit_square):
    """u = A x has constant strain and lies in the P1 space, so it is reproduced to
    round-off from its Dirichlet trace alone, shear coupling included."""
    mesh = make_unit_square(7)
    A = np.array([[0.02, 0.01], [-0.03, 0.05]])

    def exact(p):
        return p @ A.T

    bc = Conditions(Dirichlet(everywhere(), exact))
    _, solution = solved(LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU), mesh, bc)
    np.testing.assert_allclose(solution.nodal_values, exact(mesh.vertices), atol=1e-12)


def test_p2_reproduces_a_linear_displacement_and_its_constant_stress(make_unit_square):
    """The constant-stress patch test on P2: a linear displacement imposed on the
    boundary of an unloaded block is reproduced in the interior, edge nodes included,
    and the recovered stress is the single constant plane-strain value that strain
    implies."""
    E, nu, a = 200.0, 0.3, 0.01
    bc = Conditions(Dirichlet(everywhere(), lambda p: [a * p[:, 0], 0.0]))
    problem, solution = solved(LinearElastic(E=E, nu=nu), make_unit_square(6), bc,
                               element_type=QuadraticTriangleElement)

    expected_u = np.zeros((problem.space.n_nodes, 2))
    expected_u[:, 0] = a * problem.space.node_coords[:, 0]
    np.testing.assert_allclose(solution.nodal_values, expected_u, atol=1e-10)

    mu, lamb = Enu_to_Lame(E, nu)
    expected_stress = np.diag([(2 * mu + lamb) * a, lamb * a, lamb * a])   # plane strain
    np.testing.assert_allclose(solution.stress, np.broadcast_to(expected_stress, solution.stress.shape), atol=1e-8)


# -- the load: sampled at the quadrature points, or read at the nodes ---------


def test_sampled_load_beats_nodal_at_every_resolution():
    for solve in load_comparison_convergence((11, 21, 41)):
        assert solve.sampled_error < solve.nodal_error


def test_both_loads_are_second_order():
    solves = load_comparison_convergence((11, 21, 41))
    steps = np.array([s.h for s in solves])
    nodal = ConvergenceStudy(steps, np.array([s.nodal_error for s in solves]))
    sampled = ConvergenceStudy(steps, np.array([s.sampled_error for s in solves]))
    # The under-resolved coarse mesh pulls the nodal rate slightly below 2; both are
    # comfortably second order by the fit.
    assert nodal.fitted_order > 1.7
    assert sampled.fitted_order > 1.8
