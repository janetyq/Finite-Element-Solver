"""MMS for P2 (quadratic) elements.

The same manufactured Poisson problem as `test_convergence.py`, one degree up. The
claim is the rate: O(h^3) in L2 where P1 gives O(h^2). A wrong edge-node numbering,
shape function, or unpinned boundary edge node does not converge at the cubic rate,
so a passing rate is strong evidence the whole P2 path is correct.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet
from mms import (
    ConvergenceStudy,
    elastic_p2_convergence,
    poisson_p2_convergence,
)
from fem.elements import QuadraticTriangleElement
from fem.physics.forms import LinearElasticForm
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.mesh.structured import box_mesh
from fem.problem import LinearProblem
from fem.regions import everywhere
from fem.post.solution import ElasticSolution
from fem.algebra.solve import LinearSolve
from fem.space import FunctionSpace


@pytest.fixture(scope="module")
def study():
    return ConvergenceStudy.from_solves(poisson_p2_convergence((5, 9, 17, 33)))


def test_error_decreases_monotonically(study):
    for coarse, fine in zip(study.error[:-1], study.error[1:]):
        assert fine < coarse, f"error grew under refinement: {study.error}"


def test_third_order_convergence(study):
    # P2 elements give order 3 in L2; allow a band for a structured mesh.
    for p in study.orders:
        assert 2.7 < p < 3.3, f"expected ~3rd order, got orders {study.orders}"


def test_far_below_the_p1_error_floor(study):
    # Not merely "converges" but "converges faster": the finest P2 error is orders of
    # magnitude below what P1 reaches at the same spacing (test_convergence.py's ~1e-2
    # floor, and ~1e-3 by h=1/40).
    assert study.error[-1] < 1e-5


@pytest.fixture(scope="module")
def elastic_study():
    return ConvergenceStudy.from_solves(elastic_p2_convergence((5, 9, 17)))


def test_p2_elasticity_is_third_order(elastic_study):
    """The vector P2 path converges at O(h^3) like the scalar one, so the node numbering is
    right under n_components = 2 and the coupled operator."""
    for p in elastic_study.orders:
        assert 2.7 < p < 3.3, f"expected ~3rd order, got orders {elastic_study.orders}"


def test_p2_reproduces_a_linear_displacement_and_its_constant_stress():
    """The constant-stress patch test: a linear displacement imposed on the boundary of an
    unloaded block is reproduced in the interior to machine precision, and the recovered
    stress is the single constant plane-strain value that strain implies."""
    E, nu, a = 200.0, 0.3, 0.01
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)

    bc = BoundaryConditions(Dirichlet(everywhere(), lambda p: [a * p[0], 0.0]))
    problem = LinearProblem(space, LinearElasticForm(LinearElasticMaterial(E, nu)), None, bc)
    u = LinearSolve().solve(problem)

    # The interior reproduces u = (a x, 0) exactly, edge nodes included.
    expected_u = np.zeros((space.n_nodes, 2))
    expected_u[:, 0] = a * space.node_coords[:, 0]
    np.testing.assert_allclose(u.reshape(-1, 2), expected_u, atol=1e-10)

    solution = ElasticSolution.from_solve(space, u, problem.operator)
    mu, lamb = Enu_to_Lame(E, nu)
    expected_stress = np.diag([(2 * mu + lamb) * a, lamb * a, lamb * a])   # plane strain
    for element_stress in solution.stress:
        np.testing.assert_allclose(element_stress, expected_stress, atol=1e-8)


def test_p2_is_reachable_through_the_solver_facade():
    """P2 is a first-class option on the documented entry point, not only by
    hand-building a LinearProblem: `element_type` flows through `Equation.problem` to the space,
    and the solve is the accurate quadratic one."""
    from fem.physics.equations import Poisson

    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(9, 9))
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    source = lambda p: [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])]  # noqa: E731
    problem = Poisson(source=source).problem(mesh, bc, element_type=QuadraticTriangleElement)
    solution = problem.solve()

    exact = np.sin(np.pi * problem.space.node_coords[:, 0]) * np.sin(np.pi * problem.space.node_coords[:, 1])
    # P2 on this mesh is already far past the P1 error floor at the same spacing.
    assert len(solution.u) == problem.space.n_dofs
    assert np.abs(solution.u - exact).max() < 5e-3
