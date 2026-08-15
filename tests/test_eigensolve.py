"""`EigenSolve` on hand-checkable pencils: reduction, both modes, lift, and guard.

The physics tests live with the facades (`test_buckling.py`, `test_modal.py`); this
pins the reusable algebra atom itself on diagonal systems whose eigenpairs are read off
by inspection. A fixed DOF is carried through every case so the free-block reduction and
the lift back to full DOFs are exercised, not just the eigensolve.
"""
import numpy as np
import pytest
from scipy.sparse import csr_array

from fem.solve import EigenSolve


def diagonal(values):
    """A diagonal sparse operator -- the assembled shape EigenSolve indexes and factors."""
    return csr_array(np.diag(np.asarray(values, dtype=float)))


def test_regular_mode_returns_largest_eigenvalues_lifted():
    """`which='LA'` (the buckling mode) picks the largest algebraic eigenvalues.

    The operator is block-diagonal, DOF 4 fixed at an unrelated value, so the free block
    is diag(1, 2, 3, 4) and the top two eigenvalues are 3 and 4 -- with the fixed DOF
    zero in every lifted mode, and each mode supported on its own free coordinate.
    """
    A = diagonal([1.0, 2.0, 3.0, 4.0, 99.0])
    B = diagonal([1.0, 1.0, 1.0, 1.0, 1.0])   # standard problem: mu = eigenvalues of A
    free = np.array([0, 1, 2, 3])

    mu, modes = EigenSolve(n_modes=2, which='LA').solve(A, B, free, n_dofs=5)

    assert np.allclose(np.sort(mu), [3.0, 4.0])
    assert modes.shape == (2, 5)
    assert np.allclose(modes[:, 4], 0.0)                       # fixed DOF zero in every mode
    support = {int(np.argmax(np.abs(row))) for row in modes}
    assert support == {2, 3}                                   # eigenvalue 3 -> DOF 2, 4 -> DOF 3


def test_shift_invert_returns_smallest_eigenvalues():
    """`sigma=0, which='LM'` (the modal mode) picks the eigenvalues nearest zero.

    Same free block diag(1, 2, 3, 4); shift-invert about zero returns the smallest two,
    1 and 2 -- the lowest-frequency selection a modal solve relies on.
    """
    A = diagonal([1.0, 2.0, 3.0, 4.0, 99.0])
    B = diagonal([1.0, 1.0, 1.0, 1.0, 1.0])
    free = np.array([0, 1, 2, 3])

    mu, modes = EigenSolve(n_modes=2, sigma=0.0, which='LM').solve(A, B, free, n_dofs=5)

    assert np.allclose(np.sort(mu), [1.0, 2.0])
    support = {int(np.argmax(np.abs(row))) for row in modes}
    assert support == {0, 1}


def test_generalized_pencil_uses_the_mass_side():
    """A B != I is honoured: mu solves A phi = mu B phi, not the standard problem.

    A_ff = diag(2, 8, 18, 32), B_ff = diag(1, 2, 3, 4) give mu = A/B = 2, 4, 6, 8, so the
    two smallest are 2 and 4 -- values that only come out if B is actually on the pencil.
    """
    A = diagonal([2.0, 8.0, 18.0, 32.0, 1.0])
    B = diagonal([1.0, 2.0, 3.0, 4.0, 1.0])
    free = np.array([0, 1, 2, 3])

    mu, _ = EigenSolve(n_modes=2, sigma=0.0, which='LM').solve(A, B, free, n_dofs=5)

    assert np.allclose(np.sort(mu), [2.0, 4.0])


def test_too_few_free_dofs_is_rejected():
    """The Lanczos subspace needs headroom (k <= n_free - 2): a two-DOF free block
    yields no mode, so the solver raises rather than hand eigsh an impossible request."""
    A = diagonal([1.0, 2.0])
    B = diagonal([1.0, 1.0])
    with pytest.raises(ValueError, match='too few free DOFs'):
        EigenSolve(n_modes=1).solve(A, B, np.array([0, 1]), n_dofs=2)


def test_modes_are_b_orthonormal():
    """eigsh returns B-orthonormal modes: phi_i^T B phi_j = delta_ij on the free block.

    The mass-normalisation a modal post-process assumes, checked on the lifted vectors
    (the fixed DOFs are zero, so they do not disturb the inner product).
    """
    A = diagonal([1.0, 2.0, 3.0, 4.0, 99.0])
    B = diagonal([1.0, 1.0, 1.0, 1.0, 1.0])
    free = np.array([0, 1, 2, 3])

    _, modes = EigenSolve(n_modes=2, sigma=0.0, which='LM').solve(A, B, free, n_dofs=5)

    gram = modes @ (B @ modes.T)
    assert np.allclose(gram, np.eye(2), atol=1e-10)
