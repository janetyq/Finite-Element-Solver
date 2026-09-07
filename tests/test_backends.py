"""The iterative backends agree with the direct one and fail loudly.

AMG-CG and MINRES solve the same free-free block `DirectBackend` factors, so their
answers must match a direct solve to problem tolerance: on isolated systems, through the
full Poisson/elasticity solves, and through the MMS convergence net.
"""
import numpy as np
import pytest
from helpers import cantilever_bc, pinned
from mms import ConvergenceStudy, solve_poisson_mms

from fem.algebra.backends import DirectBackend, IterativeBackend, MinresBackend, det_sign
from fem.algebra.system import DiscreteSystem, Partition
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions, Initial
from fem.field import NodalField
from fem.mesh.structured import box_mesh
from fem.physics.equations import Heat, LinearElastic, Poisson
from fem.physics.forms import LinearElasticForm, rigid_body_modes
from fem.physics.materials import LinearElasticMaterial
from fem.regions import on_plane
from fem.space import FunctionSpace


def _spd(n, seed=0):
    """A random symmetric positive-definite matrix (well posed for CG)."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _symmetric_indefinite(n, seed=0):
    """A symmetric, nonsingular, indefinite matrix with a known spectrum."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.linspace(-2.0, 3.0, n)
    eigenvalues[np.abs(eigenvalues) < 0.4] = 0.7  # keep clear of zero
    return (Q * eigenvalues) @ Q.T


def test_iterative_matches_direct_on_poisson():
    """AMG-CG reproduces the direct Poisson solution to problem tolerance."""
    direct = solve_poisson_mms(41, backend=DirectBackend()).dofs
    iterative = solve_poisson_mms(41, backend=IterativeBackend()).dofs
    np.testing.assert_allclose(iterative, direct, atol=1e-8)


def test_iterative_matches_direct_on_elasticity():
    """AMG-CG reproduces the direct elasticity solution (vector field, SPD K)."""
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(31, 31))
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )
    eq = LinearElastic(E=200, nu=0.3)

    direct = eq.problem(mesh, bc).with_backend(DirectBackend()).solve().dofs
    iterative = eq.problem(mesh, bc).with_backend(IterativeBackend()).solve().dofs
    # Scale the tolerance to the field magnitude: the displacements are O(1).
    np.testing.assert_allclose(iterative, direct, atol=1e-8 * np.abs(direct).max())


def test_iterative_backend_preserves_second_order_convergence():
    """The MMS O(h^2) rate holds through the iterative backend, not just the direct one."""
    study = ConvergenceStudy.from_solves(
        [solve_poisson_mms(n, backend=IterativeBackend()) for n in (11, 21, 41)])
    for p in study.orders:
        assert 1.7 < p < 2.3, f"expected ~2nd order under CG, got {study.orders}"


def test_iterative_matches_direct_on_3d_elasticity():
    """The direct/iterative equivalence holds for 3D vector elasticity, where the rigid-body
    near-kernel and the tet assembly both differ from 2D."""
    mesh = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(7, 7, 7))
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0, 0]),
        Neumann(on_plane(0, 1.0), [0, -5, 0]),
    )
    eq = LinearElastic(E=200, nu=0.3)

    direct = eq.problem(mesh, bc).with_backend(DirectBackend()).solve().dofs
    iterative = eq.problem(mesh, bc).with_backend(IterativeBackend()).solve().dofs
    np.testing.assert_allclose(iterative, direct, atol=1e-7 * np.abs(direct).max())


def test_backends_agree_on_a_constrained_dense_system():
    """Both backends reproduce the same Dirichlet-eliminated solve of one SPD matrix."""
    A = _spd(12, seed=3)
    b = np.linspace(-1, 1, 12)
    free = np.arange(2, 12)
    fixed = np.array([0, 1])
    values = np.array([0.3, -0.4])

    direct = DiscreteSystem(A, Partition(free, fixed, A.shape[0]), DirectBackend()).solve(b, values)
    iterative = DiscreteSystem(A, Partition(free, fixed, A.shape[0]), IterativeBackend()).solve(b, values)
    np.testing.assert_allclose(iterative, direct, atol=1e-8)


def test_iterative_solver_reuses_its_setup_across_right_hand_sides():
    """One DiscreteSystem, many b's: the AMG hierarchy is built once and reused."""
    A = _spd(15, seed=4)
    free = np.arange(15)
    system = DiscreteSystem(A, Partition(free, np.array([], dtype=int), A.shape[0]), IterativeBackend())
    for seed in range(3):
        b = np.random.default_rng(seed).normal(size=15)
        np.testing.assert_allclose(system.solve_homogeneous(b), np.linalg.solve(A, b), atol=1e-8)


@pytest.mark.parametrize('dim', [2, 3])
def test_rigid_body_modes_are_in_the_stiffness_kernel(dim):
    """Translations and rotations produce no strain: K @ mode == 0 unconstrained, for
    the 3 modes of a plane body and the 6 of a solid."""
    mesh = box_mesh(corners=[[0] * dim, [1] * dim], resolution=(9, 9) if dim == 2 else (4, 4, 4))
    space = FunctionSpace(mesh, n_components=dim)
    K = space.assemble(LinearElasticForm(LinearElasticMaterial(E=200, nu=0.3)))
    modes = rigid_body_modes(mesh.vertices, dim)
    assert modes.shape == (space.n_dofs, 3 if dim == 2 else 6)
    residual = K @ modes
    assert np.abs(residual).max() < 1e-8, "a rigid-body mode strained the body"


def _cantilever():
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(25, 25))
    return mesh, cantilever_bc(traction=(0.0, -20.0))


def test_an_elastic_problem_gives_its_iterative_backend_the_rigid_body_modes():
    """An elastic problem resolves an iterative backend to one carrying its rigid-body
    modes, restricted to the free DOFs: the near-kernel AMG needs."""
    mesh, bc = _cantilever()
    elastic = LinearElastic(E=200, nu=0.3)
    problem = elastic.problem(mesh, bc).with_backend(IterativeBackend())
    backend = problem.backend
    assert isinstance(backend, IterativeBackend)
    free = problem.partition.free
    assert backend.near_null_space is not None
    assert backend.near_null_space.shape == (len(free), 3)
    np.testing.assert_array_equal(backend.near_null_space, problem.near_null_space()[free])
    assert problem.backend is backend, 'resolved once and held'

    # A near-kernel the caller set is kept; a scalar problem and a direct backend get none.
    preset = IterativeBackend(near_null_space=np.ones((len(free), 1)))
    assert problem.with_backend(preset).backend is preset
    scalar = Poisson(1.0).problem(mesh, pinned()).with_backend(IterativeBackend()).backend
    assert isinstance(scalar, IterativeBackend) and scalar.near_null_space is None
    direct = DirectBackend()
    assert problem.with_backend(direct).backend is direct
    assert isinstance(elastic.problem(mesh, bc).backend, DirectBackend), 'the default'


def test_iterative_elastic_solve_matches_direct_through_facade_and_composition():
    from fem.algebra.solve import LinearSolve

    mesh, bc = _cantilever()
    eq = LinearElastic(E=200, nu=0.3)
    direct = eq.problem(mesh, bc).with_backend(DirectBackend()).solve().dofs
    tol = 1e-7 * np.abs(direct).max()

    iterative = eq.problem(mesh, bc).with_backend(IterativeBackend()).solve().dofs
    np.testing.assert_allclose(iterative, direct, atol=tol)
    problem = eq.problem(mesh, bc).with_backend(IterativeBackend())
    composed = LinearSolve().solve(problem)
    np.testing.assert_allclose(composed, direct, atol=tol)


def test_iterative_backend_matches_direct_through_a_time_step():
    """A heat step's effective operator M + θdtK is SPD, so AMG-CG matches direct."""
    from fem.algebra.integrators import ThetaMethod
    from fem.numerics import bump_function

    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(21, 21))
    u0 = bump_function(mesh.vertices, np.array([0.5, 0.5]), mag=10, size=0.2) + 300
    problem = Heat().problem(mesh)

    start = Initial(NodalField(problem.space, u0))
    direct = ThetaMethod(dt=0.01, steps=5).solve(problem, initial=start).dofs[-1]
    iterative = ThetaMethod(dt=0.01, steps=5).solve(
        problem.with_backend(IterativeBackend()), initial=start).dofs[-1]
    np.testing.assert_allclose(iterative, direct, atol=1e-7)


def test_non_convergence_raises():
    """A CG that cannot reach tolerance in its iteration budget raises rather than returning
    the unconverged iterate."""
    A = _spd(40, seed=5)
    free = np.arange(40)
    backend = IterativeBackend(rtol=1e-14, maxiter=1)
    system = DiscreteSystem(A, Partition(free, np.array([], dtype=int), A.shape[0]), backend)
    with pytest.raises(RuntimeError, match="CG failed"):
        system.solve_homogeneous(np.ones(40))


# -- MINRES: the iterative path for symmetric indefinite systems ---------------


def test_minres_matches_direct_on_an_indefinite_system():
    """MINRES solves a symmetric indefinite block that CG cannot, matching a direct solve."""
    A = _symmetric_indefinite(30, seed=1)
    b = np.linspace(-1, 1, 30)
    free = np.arange(30)
    none = np.array([], dtype=int)

    minres = DiscreteSystem(A, Partition(free, none, A.shape[0]), MinresBackend()).solve_homogeneous(b)
    direct = DiscreteSystem(A, Partition(free, none, A.shape[0]), DirectBackend()).solve_homogeneous(b)
    np.testing.assert_allclose(minres, direct, atol=1e-8)


def test_minres_matches_direct_through_dirichlet_elimination():
    """MINRES solves the free-free block of a constrained indefinite system, matching direct."""
    A = _symmetric_indefinite(24, seed=2)
    b = np.ones(24)
    free = np.arange(2, 24)
    fixed = np.array([0, 1])
    values = np.array([0.2, -0.3])

    minres = DiscreteSystem(A, Partition(free, fixed, A.shape[0]), MinresBackend()).solve(b, values)
    direct = DiscreteSystem(A, Partition(free, fixed, A.shape[0]), DirectBackend()).solve(b, values)
    np.testing.assert_allclose(minres, direct, atol=1e-8)


def test_minres_matches_direct_on_an_spd_system():
    """MINRES also solves SPD systems (a superset of CG's domain), matching direct."""
    A = _spd(20, seed=6)
    b = np.linspace(2, -2, 20)
    free = np.arange(20)

    minres = DiscreteSystem(A, Partition(free, np.array([], dtype=int), A.shape[0]), MinresBackend()).solve_homogeneous(b)
    np.testing.assert_allclose(minres, np.linalg.solve(A, b), atol=1e-8)


def test_minres_non_convergence_raises():
    """MINRES that cannot reach tolerance in its iteration budget fails loudly, like CG."""
    A = _symmetric_indefinite(40, seed=7)
    free = np.arange(40)
    backend = MinresBackend(rtol=1e-14, maxiter=1)
    system = DiscreteSystem(A, Partition(free, np.array([], dtype=int), A.shape[0]), backend)
    with pytest.raises(RuntimeError, match="MINRES failed"):
        system.solve_homogeneous(np.ones(40))


# -- det_sign: the determinant's sign, read off a sparse LU ----------------------


def _sign_of(A, backend=None):
    """`det_sign` of the whole of `A`, factored through `backend` (direct by default)."""
    free = np.arange(A.shape[0])
    system = DiscreteSystem(A, Partition(free, np.array([], dtype=int), A.shape[0]),
                            backend if backend is not None else DirectBackend())
    return det_sign(system.factorization)


def test_det_sign_is_positive_for_a_positive_definite_matrix():
    """Every eigenvalue positive, so the determinant is: the stable branch of a path."""
    assert _sign_of(_spd(15, seed=11)) == 1


@pytest.mark.parametrize('n_negative', [1, 2, 3])
def test_det_sign_follows_the_parity_of_the_negative_eigenvalues(n_negative):
    """The sign is (-1)^(number of negative eigenvalues), which is what makes a flip
    between two states a bracket around an odd number of crossings."""
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    eigenvalues = np.linspace(1.0, 4.0, 12)
    eigenvalues[:n_negative] *= -1.0
    A = (Q * eigenvalues) @ Q.T
    assert _sign_of(A) == (-1) ** n_negative
    assert _sign_of(A) == int(np.sign(np.linalg.det(A)))


def test_a_singular_block_never_reaches_det_sign():
    """Singularity surfaces at factorization time: `splu` refuses the matrix, so the
    determinant is never reported as a sign it does not have."""
    with pytest.raises(RuntimeError, match='singular'):
        _sign_of(np.diag([1.0, 2.0, 0.0, 3.0]))


def test_det_sign_is_unknown_for_an_iterative_factorization():
    """AMG-CG forms no LU, so there is no determinant to read: None, not a guess."""
    A = _spd(20, seed=12)
    assert _sign_of(A, IterativeBackend()) is None
