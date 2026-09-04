"""`Partition` splits the DOFs and eliminates; `DiscreteSystem` factors the free block once."""
import numpy as np
import pytest
import scipy.sparse as sp

from fem.algebra.system import DiscreteSystem, Partition


def _spd(n, seed=0):
    """A random symmetric positive-definite matrix, so the solves are well posed."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def test_matches_dense_elimination():
    """solve() reproduces the hand-written free/fixed elimination."""
    A = _spd(6)
    b = np.arange(6, dtype=float)
    free = np.array([0, 2, 3, 5])
    fixed = np.array([1, 4])
    fixed_values = np.array([0.7, -0.3])

    x = DiscreteSystem(A, Partition(free, fixed, len(A))).solve(b, fixed_values)

    # Reference: fixed DOFs held, free block solved directly.
    expected = np.zeros(6)
    expected[fixed] = fixed_values
    b_free = b[free] - A[np.ix_(free, fixed)] @ fixed_values
    expected[free] = np.linalg.solve(A[np.ix_(free, free)], b_free)
    np.testing.assert_allclose(x, expected)
    np.testing.assert_allclose(x[fixed], fixed_values)


def test_residual_is_zero_on_free_rows():
    """The solution satisfies (A x - b) = 0 on the free DOFs."""
    A = _spd(8, seed=1)
    b = np.linspace(-1, 1, 8)
    free = np.array([0, 1, 4, 5, 6])
    fixed = np.array([2, 3, 7])
    x = DiscreteSystem(A, Partition(free, fixed, len(A))).solve(b, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose((A @ x - b)[free], 0, atol=1e-10)


def test_factorization_is_reused_across_right_hand_sides():
    """One factorization, many b's: each solve matches an independent dense solve."""
    A = _spd(10, seed=2)
    free = np.arange(2, 10)
    fixed = np.array([0, 1])
    fixed_values = np.array([0.5, -0.5])
    system = DiscreteSystem(A, Partition(free, fixed, len(A)))

    for seed in range(3):
        b = np.random.default_rng(seed).normal(size=10)
        x = system.solve(b, fixed_values)
        b_free = b[free] - A[np.ix_(free, fixed)] @ fixed_values
        expected_free = np.linalg.solve(A[np.ix_(free, free)], b_free)
        np.testing.assert_allclose(x[free], expected_free)


def test_elimination_preserves_symmetry():
    """Eliminating the Dirichlet DOFs keeps the factored free-free block symmetric, so
    the reduced system is still SPD. Symmetry shows as reciprocity through the
    homogeneous solve: b1 . A_ff^-1 b2 == b2 . A_ff^-1 b1. A free/fixed indexing slip or
    a one-sided elimination would break it while a plain solve could still look right."""
    A = _spd(9, seed=4)
    free = np.array([0, 1, 3, 5, 6, 8])
    fixed = np.array([2, 4, 7])
    system = DiscreteSystem(A, Partition(free, fixed, len(A)))

    rng = np.random.default_rng(5)
    b1, b2 = rng.normal(size=9), rng.normal(size=9)
    x1, x2 = system.solve_homogeneous(b1), system.solve_homogeneous(b2)
    assert float(b1 @ x2) == pytest.approx(float(b2 @ x1))


def test_no_fixed_dofs_is_a_plain_solve():
    """With an empty fixed set the system is just A x = b."""
    A = _spd(5, seed=3)
    b = np.ones(5)
    x = DiscreteSystem(A, Partition(np.arange(5), np.array([], dtype=int), 5)).solve_homogeneous(b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b))


def test_partition_eliminates_into_the_free_free_and_free_fixed_blocks():
    """`eliminate` returns exactly the two blocks a solve reads, in the partition's own
    index order, and works on a sparse operator as on a dense one."""
    A = _spd(7, seed=6)
    free = np.array([5, 0, 3, 6])
    fixed = np.array([1, 2, 4])
    partition = Partition(free, fixed, 7)
    assert partition.n_dofs == 7 and partition.n_free == 4

    for operator in (A, sp.csr_array(A)):
        free_free, free_fixed = partition.eliminate(operator)
        np.testing.assert_allclose(np.asarray(free_free.todense() if sp.issparse(free_free) else free_free),
                                   A[np.ix_(free, free)])
        np.testing.assert_allclose(np.asarray(free_fixed.todense() if sp.issparse(free_fixed) else free_fixed),
                                   A[np.ix_(free, fixed)])


def test_partition_with_no_fixed_dofs_eliminates_nothing():
    A = _spd(4, seed=7)
    free_free, free_fixed = Partition(np.arange(4), np.array([], dtype=int), 4).eliminate(A)
    np.testing.assert_allclose(free_free, A)
    assert free_fixed.shape == (4, 0)


def test_partition_must_cover_the_dofs():
    with pytest.raises(ValueError, match='do not partition'):
        Partition(np.array([0, 1]), np.array([2]), 4)
    with pytest.raises(ValueError, match='against a partition'):
        DiscreteSystem(_spd(5), Partition(np.arange(3), np.array([3]), 4))


def test_partition_compares_by_content():
    same = Partition(np.array([0, 2]), np.array([1]), 3)
    assert same == Partition(np.array([0, 2]), np.array([1]), 3)
    assert same != Partition(np.array([2, 0]), np.array([1]), 3)
    assert same != Partition(np.array([0, 1]), np.array([2]), 3)
