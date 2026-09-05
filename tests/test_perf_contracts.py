"""Performance contracts: the counts that fix a solve path's cost, asserted exactly.

Wall-clock time is not testable in CI (runners differ, one machine drifts), but what
makes a path fast is countable and deterministic: how many factorizations a solve
performs, how many back-substitutions, how much fill the direct ordering produces, how
many CG iterations AMG needs as the mesh refines. Each test here names the regression it
guards. `benchmarks/bench.py` is the timing side of the same question, run by hand.

The set also includes `tests/test_loads.py::test_a_constant_source_integrates_element_wise_without_the_mass_matrix`,
which pins that a constant load never assembles the mass matrix.
"""
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import LinearOperator, splu

from fem.algebra.backends import DirectBackend, IterativeBackend
from fem.algebra.integrators import NewmarkMethod, ThetaMethod
from fem.algebra.solve import LinearSolve
from fem.algebra.system import DiscreteSystem
from fem.analysis.sensitivity import Compliance, DensityParameterization, PointValue, SensitivityAnalysis
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.equations import Heat, LinearElastic, Poisson, Wave
from fem.regions import on_plane
from helpers import CountingBackend, pinned


def _poisson(n: int = 12):
    return Poisson().problem(box_mesh([[0, 0], [1, 1]], (n, n)), pinned() + Source(1.0))


def _cantilever(n: int = 12):
    mesh = box_mesh([[0, 0], [2, 1]], (2 * n, n))
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]), Neumann(on_plane(0, 2.0), [0.0, -1.0]))
    return LinearElastic(E=200.0, nu=0.3).problem(mesh, bc)


def test_linear_solve_factors_once_and_back_substitutes_once():
    """Guards: a steady linear solve is one factorization and one back-substitution."""
    backend = CountingBackend()
    _poisson().solve(LinearSolve(), backend=backend)
    assert (backend.factorizations, backend.solves) == (1, 1)


def test_theta_method_factors_once_over_all_steps():
    """Guards: the theta method's constant step operator is factored once and reused,
    one back-substitution per step."""
    steps = 25
    problem = Heat(conductivity=1.0).problem(box_mesh([[0, 0], [1, 1]], (12, 12)), pinned() + Source(1.0))
    backend = CountingBackend()
    ThetaMethod(dt=1e-3, steps=steps).solve(problem, backend=backend)
    assert (backend.factorizations, backend.solves) == (1, steps)


def test_newmark_factors_twice_over_all_steps():
    """Guards: Newmark factors the mass once for the initial acceleration and the
    effective operator once for every step, one back-substitution per step plus the
    initial one."""
    steps = 25
    problem = Wave().problem(box_mesh([[0, 0], [1, 1]], (12, 12)), pinned() + Source(1.0))
    backend = CountingBackend()
    NewmarkMethod(dt=1e-3, steps=steps).solve(problem, backend=backend)
    assert (backend.factorizations, backend.solves) == (2, steps + 1)


def test_sensitivity_shares_one_factorization_and_skips_the_self_adjoint_solve():
    """Guards: the forward and adjoint solves share one factorization, and a self-adjoint
    quantity of interest (compliance under homogeneous supports) takes lambda = u with no
    adjoint back-substitution; a point value costs exactly one."""
    problem = _cantilever()
    rho = np.full(len(problem.space.element_nodes), 0.5)
    material = problem.physics.material
    parameterization = DensityParameterization.create(problem.space, rho, material)

    backend = CountingBackend()
    analysis = SensitivityAnalysis(problem, backend=backend)
    u = analysis.solve_forward()
    analysis.gradient(Compliance(), parameterization, u)
    assert (backend.factorizations, backend.solves) == (1, 1)

    analysis.gradient(PointValue([2.0, 0.5], component=1), parameterization, u)
    assert (backend.factorizations, backend.solves) == (1, 2)


def test_direct_ordering_keeps_the_fill_low():
    """Guards: `DirectBackend`'s minimum-degree ordering. On this 2D P1 Poisson block the
    L + U fill is 5.3x nnz(A); scipy's default COLAMD ordering gives 12.2x and grows
    faster with the mesh, so the bound separates the two with room for a scipy update."""
    problem = _poisson(100)
    free = problem.constraints[0]
    block = csc_array(problem.tangent()[np.ix_(free, free)])
    system = DiscreteSystem(problem.tangent(), problem.constraints, DirectBackend())
    lu = system._factorization   # the SuperLU object DirectBackend returns as its Factorization
    fill = (lu.L.nnz + lu.U.nnz) / block.nnz
    assert fill < 8.0, f'L + U fill is {fill:.1f}x nnz(A); the ordering has regressed'
    # The bound is tight enough to fail on the default ordering.
    colamd = splu(block, permc_spec='COLAMD')
    assert (colamd.L.nnz + colamd.U.nnz) / block.nnz > 8.0


class _CountingOperator(LinearOperator):
    """A sparse matrix that counts its matvecs, standing in for CG's operator."""

    def __init__(self, A) -> None:
        super().__init__(dtype=A.dtype, shape=A.shape)
        self.A = A
        self.matvecs = 0

    def _matvec(self, x):
        self.matvecs += 1
        return self.A @ x


def _cg_matvecs(n: int) -> int:
    problem = _poisson(n)
    system = DiscreteSystem(problem.tangent(), problem.constraints, IterativeBackend())
    # The CG solver holds the free block as `_A`; swap in a counting view of it. A private
    # attribute, read here because the count is the contract and nothing public exposes it.
    solver = system._factorization
    counting = _CountingOperator(solver._A)
    solver._A = counting
    system.solve(problem.load)
    return counting.matvecs


def test_amg_cg_iterations_stay_bounded_under_refinement():
    """Guards: the AMG preconditioner keeps CG's iteration count flat as the mesh
    refines; a broken hierarchy or near-kernel shows as a count that grows with the
    mesh. The bound is loose (1.5x on a doubling) because the exact count depends on
    pyamg's aggregation, which is deterministic but not part of this contract."""
    coarse, fine = _cg_matvecs(60), _cg_matvecs(120)
    assert fine <= 1.5 * coarse, f'CG matvecs grew from {coarse} to {fine} on a mesh doubling'
    assert fine < 40, f'{fine} CG matvecs at 120x120; the preconditioner is not doing its job'


def test_counting_backend_matches_the_direct_result():
    """The counting wrapper changes nothing but the counts."""
    problem = _poisson()
    direct = problem.solve(backend=DirectBackend()).dofs
    counted = problem.solve(backend=CountingBackend()).dofs
    np.testing.assert_allclose(counted, direct)
