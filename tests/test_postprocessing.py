"""Derived fields: the quantities recovered from a solved displacement.

Recovery is shared -- `Solver` and `TopologyOptimizer` both turn a solved `u`
into strain, stress, and compliance -- so these pin the contract both go through
rather than either one's implementation of it.

The compliance identity below is the anchor. Element compliance is the volume
integral of the double contraction sigma:epsilon, and the assembled stiffness is
K = int B^T D B, so summing the per-element compliance must reproduce u^T K u
exactly. That ties the recovered fields to the operator that produced them
without a single hard-coded magic number, which is what makes it survive a
change of representation.
"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.equations import LinearElastic
from fem.regions import on_plane
from fem.solver import Solver
from fem.topology import TopologyOptimizer


def _cantilever_bc() -> BoundaryConditions:
    """Left edge pinned, right edge pulled down -- a loaded 2D cantilever."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])
    return bc


def _solved(mesh):
    solver = Solver(mesh, LinearElastic(E=1.0, nu=0.3), _cantilever_bc())
    return solver, solver.solve()


def test_element_compliance_sums_to_the_strain_energy_form(make_unit_square):
    """sum_e int sigma:epsilon == u^T K u.

    The per-element compliance is a contraction of the recovered stress against
    the recovered strain; K is assembled from the same B and D. If the two ever
    disagree -- a dropped factor of two on the shear terms, a mismatched Voigt
    ordering -- the identity breaks, which is exactly the class of error a
    representation change can introduce.
    """
    mesh = make_unit_square(8)
    solver, solution = _solved(mesh)

    K = solver._steady_problem().tangent(None)
    np.testing.assert_allclose(
        solution.compliance.sum(), solution.u @ (K @ solution.u), rtol=1e-10
    )


def test_compliance_is_positive_and_finite(make_unit_square):
    """Strain energy density is non-negative for a positive-definite D, so no
    element may report negative compliance."""
    _, solution = _solved(make_unit_square(8))
    assert np.all(np.isfinite(solution.compliance))
    assert np.all(solution.compliance >= 0.0)


def test_solver_and_optimizer_recover_the_same_fields(make_unit_square):
    """The two call sites that recover derived fields must agree.

    `Solver` and `TopologyOptimizer` run the same numerical path -- LinearProblem
    -> LinearSolve -> derived fields -- so at a uniform unit density the
    optimizer's first iterate is the plain elastic solve. They recover the fields
    through separate code today; this pins that they agree, so collapsing them
    onto one owner is a refactor rather than a change.
    """
    mesh = make_unit_square(6)
    _, solution = _solved(mesh)

    optimizer = TopologyOptimizer(
        mesh, LinearElastic(E=1.0, nu=0.3), _cantilever_bc(),
        iters=1, volume_frac=1.0, penalty=1.0,
    )
    # rho = 1 and p = 1 leaves E(rho) = E_0, so this is the same elastic problem.
    optimizer_solution = optimizer._solve()

    np.testing.assert_allclose(optimizer_solution.u, solution.u, rtol=1e-10)
    np.testing.assert_allclose(
        optimizer_solution.compliance, solution.compliance, rtol=1e-10
    )


@pytest.mark.parametrize('n', [4, 6])
def test_compliance_is_mesh_convergent(n, make_unit_square):
    """Total compliance is a physical quantity, so refining the mesh must not
    move it wildly -- a coarse guard against a recovery that scales with element
    count rather than with volume."""
    _, coarse = _solved(make_unit_square(n))
    _, fine = _solved(make_unit_square(n + 4))
    assert coarse.compliance.sum() == pytest.approx(fine.compliance.sum(), rel=0.5)
