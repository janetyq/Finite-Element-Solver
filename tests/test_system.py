"""DiscreteSystem eliminates the Dirichlet DOFs and reuses its factorization."""
import numpy as np

from fem.system import DiscreteSystem


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

    x = DiscreteSystem(A, (free, fixed, fixed_values)).solve(b)

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
    x = DiscreteSystem(A, (free, fixed, np.array([1.0, 2.0, 3.0]))).solve(b)
    np.testing.assert_allclose((A @ x - b)[free], 0, atol=1e-10)


def test_factorization_is_reused_across_right_hand_sides():
    """One factorization, many b's: each solve matches an independent dense solve."""
    A = _spd(10, seed=2)
    free = np.arange(2, 10)
    fixed = np.array([0, 1])
    fixed_values = np.array([0.5, -0.5])
    system = DiscreteSystem(A, (free, fixed, fixed_values))

    for seed in range(3):
        b = np.random.default_rng(seed).normal(size=10)
        x = system.solve(b)
        b_free = b[free] - A[np.ix_(free, fixed)] @ fixed_values
        expected_free = np.linalg.solve(A[np.ix_(free, free)], b_free)
        np.testing.assert_allclose(x[free], expected_free)


def test_no_fixed_dofs_is_a_plain_solve():
    """With an empty fixed set the system is just A x = b."""
    A = _spd(5, seed=3)
    b = np.ones(5)
    x = DiscreteSystem(A, (np.arange(5), np.array([], dtype=int), np.array([]))).solve(b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b))
