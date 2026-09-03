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
    thermoelastic_convergence,
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


def _h1_study(solves):
    """A `ConvergenceStudy` over the gradient error, so an H1 study reuses an L2 study's
    solves rather than re-running them; the `h1_error` is set on the same `MMSSolve`s."""
    return ConvergenceStudy(np.array([s.h for s in solves]), np.array([s.h1_error for s in solves]))


@cache
def poisson_p1_l2():
    return ConvergenceStudy.from_solves(_poisson_p1_solves())


@cache
def poisson_p1_h1():
    return _h1_study(_poisson_p1_solves())


@cache
def _poisson_p2_solves():
    return poisson_p2_convergence((5, 9, 17, 33))


@cache
def poisson_p2():
    return ConvergenceStudy.from_solves(_poisson_p2_solves())


@cache
def poisson_p2_h1():
    return _h1_study(_poisson_p2_solves())


@cache
def variable_coefficient():
    return ConvergenceStudy.from_solves(variable_coefficient_convergence((11, 21, 41)))


@cache
def _elastic_p1_solves():
    return elastic_convergence((9, 17, 33))


@cache
def elastic_p1():
    return ConvergenceStudy.from_solves(_elastic_p1_solves())


@cache
def elastic_p1_h1():
    return _h1_study(_elastic_p1_solves())


@cache
def _elastic_p2_solves():
    return elastic_p2_convergence((5, 9, 17))


@cache
def elastic_p2():
    return ConvergenceStudy.from_solves(_elastic_p2_solves())


@cache
def elastic_p2_h1():
    return _h1_study(_elastic_p2_solves())


@cache
def thermoelastic_p1():
    return ConvergenceStudy.from_solves(thermoelastic_convergence((9, 17, 33)))


@cache
def thermoelastic_nodal_p1():
    return ConvergenceStudy.from_solves(thermoelastic_convergence((9, 17, 33), nodal=True))


@cache
def poisson_3d():
    # h = 1/4, 1/8, 1/12 on the default regular box mesh. Its near-regular tets are in
    # band from the coarsest level, where the Kuhn mesh read 1.74; observed 1.93, 1.98.
    return ConvergenceStudy.from_solves(poisson_convergence((5, 9, 13), dim=3))


@cache
def _mixed_bc_solves():
    return mixed_bc_convergence((11, 21, 41))    # h = 0.1, 0.05, 0.025


@cache
def mixed_bc_l2():
    return ConvergenceStudy.from_solves(_mixed_bc_solves())


@cache
def mixed_bc_h1():
    return _h1_study(_mixed_bc_solves())


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
    # The gradient error of the same P2 solve, O(h^2), one order below its L2 rate. It
    # reads grad_phi directly (never the assembled K), so a wrong P2 shape gradient or
    # edge-node numbering degrades this while the L2 rate can still look right. Observed
    # 1.95, 1.99, 2.00 with a coarse-end pre-asymptotic climb, so the band allows it.
    Study('poisson_p2_h1', poisson_p2_h1, 2, (1.8, 2.2)),
    # kappa(x) and f both vary within an element, so both sides of the solve go
    # through the quadrature layer a constant-coefficient assembly lacks.
    Study('variable_coefficient', variable_coefficient, 2, (1.7, 2.3), floor=1e-2),
    # The coupled vector path: only u_x is nonzero, but the shear terms of sigma make
    # both components of f nonzero.
    Study('elastic_p1', elastic_p1, 2, (1.7, 2.3)),
    # The gradient (strain) error of the vector P1 solve, O(h): the sharp probe of the
    # elastic B-matrix and the coupled stiffness, measured against the closed-form
    # deformation gradient rather than through K. Observed 0.98, 0.99.
    Study('elastic_p1_h1', elastic_p1_h1, 1, (0.85, 1.15)),
    # The vector P2 path: the node numbering under n_components = 2 and the coupled
    # operator, at the scalar P2 rate.
    Study('elastic_p2', elastic_p2, 3, (2.7, 3.3)),
    # Its gradient error, O(h^2): the vector P2 shape gradients under n_components = 2.
    # Observed 1.94, 1.98.
    Study('elastic_p2_h1', elastic_p2_h1, 2, (1.8, 2.2)),
    # The elastic study under a manufactured temperature: the thermal load
    # (sampled at the quadrature points) and the corrected stress enter the rate.
    Study('thermoelastic_p1', thermoelastic_p1, 2, (1.7, 2.3)),
    # The same with the temperature handed over as its nodal interpolant, the
    # coupling path from a heat solve; the interpolation error is O(h^2) too.
    Study('thermoelastic_nodal_p1', thermoelastic_nodal_p1, 2, (1.7, 2.3)),
    # The same scalar Poisson study in 3D, on a tetrahedral box: assembly and the P1
    # solve on tets, not only triangles. On the regular mesh it is in a tighter band
    # from a coarse sequence; observed orders 1.93, 1.98.
    Study('poisson_3d', poisson_3d, 2, (1.8, 2.2), floor=5e-3),
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
    # h = 1/4, 1/6, 1/8, 1/12, on the default regular (five-tet) box mesh. Its
    # near-regular tets are asymptotic from the coarsest level, so the sequence stays
    # coarse and the direct factorization is cheap; the same study on Kuhn tets needed
    # h down to 1/28 to reach the same rate (see fem/mesh/structured.py). Observed
    # orders 1.99, 2.00, 2.00.
    return ConvergenceStudy.from_solves(elastic_convergence((5, 7, 9, 13), dim=3))


def test_3d_elasticity_is_second_order(elastic_3d):
    """The error falls monotonically under refinement, inside a tight O(h^2) band the
    Kuhn mesh could not hold at these coarse sizes."""
    for coarse, fine in zip(elastic_3d.error[:-1], elastic_3d.error[1:]):
        assert fine < coarse, f'error grew under refinement: {elastic_3d.error}'
    assert all(1.9 < p < 2.1 for p in elastic_3d.orders), f'expected ~2nd order, got {elastic_3d.orders}'


def test_3d_is_asymptotic_from_the_coarsest_mesh(elastic_3d):
    """The regular mesh's payoff: there is no pre-asymptotic drift to climb out of, so
    the very first pair already reads ~2. That makes this a sharp defect test at coarse
    sizes, where the old Kuhn sequence could only reach 1.9 by refining to h = 1/28: a
    defect degrading the rate to a constant 1.8 shows here immediately, on tiny meshes."""
    orders = elastic_3d.orders
    assert orders[0] > 1.95, f'coarsest pair is not yet asymptotic: {orders}'
    assert 1.95 < elastic_3d.fitted_order < 2.05, f'fitted rate off two: {elastic_3d.fitted_order}'


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
