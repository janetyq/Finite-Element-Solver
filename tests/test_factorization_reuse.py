"""A linear problem factors its operator once and every solve of it reuses that.

The problem holds its backend and its factored `system`; its snapshots share both.
These tests pin the counts with `CountingBackend`: how many times each path factors
and back-substitutes is exact where wall-clock time is not.
"""
import numpy as np
import pytest
from helpers import CountingBackend, pinned

from fem.algebra.backends import DirectBackend
from fem.algebra.solve import LinearSolve, NewtonSolve
from fem.analysis.estimators import GoalOrientedEstimator
from fem.analysis.sensitivity import PointValue
from fem.boundary import Dirichlet
from fem.conditions import Conditions, TimeDependent
from fem.loads import Source
from fem.physics.equations import FiniteStrainElastic, LinearElastic, Poisson
from fem.physics.forms import DiffusionForm
from fem.regions import on_plane


def _poisson(mesh, backend):
    return Poisson().problem(mesh, pinned() + Source(1.0)).with_backend(backend)


def test_two_solves_of_one_problem_factor_once(make_unit_square):
    backend = CountingBackend()
    problem = _poisson(make_unit_square(6), backend)
    first = problem.solve().dofs
    second = LinearSolve().solve(problem)
    assert (backend.factorizations, backend.solves) == (1, 2)
    np.testing.assert_array_equal(first, second)
    assert problem.system is problem.system, 'held, not rebuilt'


def test_snapshots_in_time_share_one_assembly_and_one_factorization(make_unit_square):
    """`solve(t=...)` solves the snapshot `at(t)`, a copy: the matrix and the system it
    builds fill the parent's holders, so the next time costs a back-substitution, and
    each snapshot solves with its own prescribed values."""
    backend = CountingBackend()
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
                    Dirichlet(on_plane(0, 1.0), TimeDependent(lambda p, t: [0.01 * t, 0.0])))
    mesh = make_unit_square(6)
    problem = LinearElastic(E=10.0, nu=0.3).problem(mesh, bc).with_backend(backend)

    solutions = [problem.solve(t=t).dofs for t in (0.5, 1.0, 1.5)]
    assert (backend.factorizations, backend.solves) == (1, 3)
    assert problem.at(2.0).tangent() is problem.tangent(), 'one matrix, shared'

    # A fresh problem per time, stated with the values fixed, gives the same answers.
    for t, dofs in zip((0.5, 1.0, 1.5), solutions, strict=True):
        fixed = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
                           Dirichlet(on_plane(0, 1.0), [0.01 * t, 0.0]))
        expected = LinearElastic(E=10.0, nu=0.3).problem(mesh, fixed).solve().dofs
        np.testing.assert_allclose(dofs, expected, atol=1e-12)


def test_a_load_factor_snapshot_solves_with_its_own_values_on_the_shared_system(make_unit_square):
    backend = CountingBackend()
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]), Dirichlet(on_plane(0, 1.0), [0.1, 0.0]))
    mesh = make_unit_square(6)
    problem = LinearElastic(E=10.0, nu=0.3).problem(mesh, bc).with_backend(backend)
    full = problem.solve().dofs
    half = problem.with_load_factor(0.5).solve().dofs
    assert (backend.factorizations, backend.solves) == (1, 2)
    np.testing.assert_allclose(half, 0.5 * full, atol=1e-12)


def test_newton_on_a_linear_problem_factors_once(make_unit_square):
    """Newton reads the held system: the exact first step and the zero second step
    are two back-substitutions of one factorization."""
    backend = CountingBackend()
    problem = _poisson(make_unit_square(6), backend)
    u = NewtonSolve().solve(problem)
    assert (backend.factorizations, backend.solves) == (1, 2)
    np.testing.assert_allclose(u, LinearSolve().solve(problem), atol=1e-12)


def test_a_goal_oriented_round_factors_once(make_unit_square):
    """The dual solve reuses the forward solve's factorization."""
    backend = CountingBackend()
    problem = _poisson(make_unit_square(8), backend)
    solution = problem.solve()
    GoalOrientedEstimator(PointValue(np.array([0.5, 0.5]))).estimate(problem, solution)
    assert backend.factorizations == 1
    assert backend.solves == 2


def test_a_restated_operator_refactors(make_unit_square):
    backend = CountingBackend()
    problem = _poisson(make_unit_square(6), backend)
    problem.solve()
    mass = problem.mass
    doubled = problem.with_operator(DiffusionForm(2.0))
    np.testing.assert_allclose(doubled.solve().dofs, 0.5 * problem.solve().dofs, atol=1e-12)
    assert backend.factorizations == 2
    assert doubled.tangent() is not problem.tangent()
    assert doubled.mass is mass, 'the mass does not depend on the operator'


def test_with_backend_shares_the_matrix_and_factors_its_own(make_unit_square):
    first, second = CountingBackend(), CountingBackend()
    problem = _poisson(make_unit_square(6), first)
    problem.solve()
    other = problem.with_backend(second)
    assert other.tangent() is problem.tangent()
    other.solve()
    assert (first.factorizations, second.factorizations) == (1, 1)
    assert other.backend is second
    assert isinstance(problem.with_backend(None).backend, DirectBackend)


def test_a_state_dependent_problem_holds_no_system(make_unit_square):
    problem = FiniteStrainElastic(E=200, nu=0.4).problem(make_unit_square(4))
    with pytest.raises(ValueError, match='no one system'):
        _ = problem.system
