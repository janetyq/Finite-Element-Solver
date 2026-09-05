"""The solve strategies' failure paths: a line search that cannot descend and a
regularization that the backend rejects at every shift.

Both refuse rather than return a bad answer, the pattern `NewtonDivergence` sets for an
unconverged Newton state: a step that fails its test is never handed back as if it had
passed. The success paths of these strategies are exercised throughout the elasticity
and problem tests; this file pins the two error branches.
"""
import numpy as np
import pytest
from helpers import pinned

from fem.algebra.solve import (
    BacktrackingLineSearch,
    LineSearchFailure,
    NewtonSolve,
    TangentRegularization,
)
from fem.loads import Source
from fem.physics.equations import Poisson


class _AlwaysBreaks:
    """A backend whose factorization rejects every right-hand side, standing in for one
    that breaks down on an indefinite shift however far the regularization escalates."""

    def prepare(self, A):
        return self

    def solve(self, b):
        raise RuntimeError('backend breakdown')


def test_line_search_refuses_a_step_that_never_descends():
    """Handed a descent slope but a merit that rises for every step length, backtracking
    cannot satisfy Armijo; it raises rather than return the smallest, non-descending
    step. The exception carries the smallest alpha it tried and the merit either side."""
    line_search = BacktrackingLineSearch(max_backtracks=5)
    u, step = np.zeros(3), np.ones(3)

    def rising_merit(w):
        return 1.0 + float(np.linalg.norm(w))

    with pytest.raises(LineSearchFailure, match='sufficient decrease') as caught:
        line_search.search(rising_merit, u, step, slope=-1.0)

    smallest = 0.5**4   # alpha = rho^(max_backtracks - 1) at the last attempt
    assert caught.value.alpha == pytest.approx(smallest)
    assert caught.value.phi0 == pytest.approx(1.0)
    assert caught.value.phi_alpha == pytest.approx(1.0 + smallest * np.sqrt(3.0))


def test_regularization_gives_up_when_the_backend_rejects_every_shift(make_unit_square):
    """When no shift the schedule tries yields a solvable system (here because the
    backend breaks down on all of them), the regularized step is unreachable and the
    solve raises rather than return an unstepped state."""
    problem = Poisson().problem(make_unit_square(4), pinned() + Source(1.0)).with_backend(_AlwaysBreaks())
    solver = NewtonSolve(regularization=TangentRegularization(max_shifts=3))

    with pytest.raises(RuntimeError, match='rejected every shift'):
        solver.solve(problem)
