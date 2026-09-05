"""Quasi-static continuation: `QuasiStaticStepping` and `Problem.with_load_factor`.

The anchors: on a linear problem every level is exactly the load fraction of the full
solution, so the whole path is analytic; on a nonlinear one the walk must land on the
same equilibrium a direct Newton solve finds. Bisection and failure are pinned with a
brittle strategy that refuses increments above a threshold, so the retry logic is
exercised deterministically rather than by hunting for a physically divergent case.
"""
from dataclasses import dataclass, field

import numpy as np
import pytest

from fem.algebra.solve import BacktrackingLineSearch, NewtonDivergence, NewtonSolve
from fem.algebra.stepping import QuasiStaticStepping, SteppingDivergence
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.loads import Source
from fem.physics.equations import FiniteStrainElastic, Heat, LinearElastic, Poisson
from fem.post.solution import ElasticSolution
from fem.regions import TimeDependent, on_plane


def _pulled_block(mesh, pull=0.3):
    """A block held at x = 0 and pulled to `pull` at x = 1: proportional Dirichlet
    loading, the case with_load_factor scales."""
    return Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [pull, 0]),
    )


# -- with_load_factor ----------------------------------------------------------


def test_with_load_factor_scales_load_and_dirichlet(make_unit_square):
    mesh = make_unit_square(4)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [0.2, 0]),
        Source([0, -1.0]),
    )
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, bc)
    half = problem.with_load_factor(0.5)

    np.testing.assert_allclose(half.load, 0.5 * problem.load)
    values0 = problem.fixed_values
    assert half.partition == problem.partition, 'the same DOFs are fixed; only the values scale'
    np.testing.assert_allclose(half.fixed_values, 0.5 * values0)
    # A snapshot, not a mutation: the original still answers with its own loading.
    np.testing.assert_allclose(problem.fixed_values, values0)


def test_with_load_factor_refuses_a_time_dependent_problem(make_unit_square):
    mesh = make_unit_square(3)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), 0.0),
                    Source(TimeDependent(lambda p, t: t * np.ones(len(p)))))
    problem = Poisson().problem(mesh, bc)
    with pytest.raises(ValueError, match='at\\(t\\)'):
        problem.with_load_factor(0.5)
    # Its snapshot has one fixed loading, so that scales.
    assert problem.at(2.0).with_load_factor(0.5).load == pytest.approx(problem.at(1.0).load)


# -- the walk -------------------------------------------------------------------


def test_linear_path_is_the_load_fraction_of_the_full_solution(make_unit_square):
    """On a linear problem the equilibrium at load fraction t is exactly t times the
    full solution, Dirichlet-driven and traction-driven parts alike, so the whole
    history is analytic."""
    mesh = make_unit_square(4)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [0.1, 0]),
        Source([0, -0.5]),
    )
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, bc)
    full = problem.solve()

    history = QuasiStaticStepping(steps=4).solve(problem)
    np.testing.assert_allclose(history.t, [0.0, 0.25, 0.5, 0.75, 1.0])
    for t, step in zip(history.t, history, strict=True):
        assert isinstance(step, ElasticSolution)
        np.testing.assert_allclose(step.dofs, t * full.dofs, atol=1e-9)


def test_nonlinear_walk_lands_on_the_direct_equilibrium(make_unit_square):
    """Warm-started stepping and a direct Newton solve must agree on the final
    equilibrium of a finite-strain stretch; the walk just gets there in easy steps."""
    mesh = make_unit_square(5)
    problem = FiniteStrainElastic(E=200, nu=0.3).problem(mesh, _pulled_block(mesh, 0.3))
    newton = NewtonSolve(line_search=BacktrackingLineSearch())

    history = QuasiStaticStepping(steps=5, newton=newton).solve(problem)
    direct = newton.solve(problem)
    assert len(history) == 6
    np.testing.assert_allclose(history[-1].dofs, direct, atol=1e-5)
    # The path is monotone loading: the pulled edge moves further at every level.
    tips = [float(step.nodal_values[:, 0].max()) for step in history]
    assert all(a < b + 1e-12 for a, b in zip(tips, tips[1:], strict=False))


def test_time_dependent_values_set_the_path(make_unit_square):
    """A problem with TimeDependent values walks their own path: each level is the
    snapshot at t, and the final level matches the steady solve at t_end."""
    mesh = make_unit_square(4)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), 0.0),
                    Source(TimeDependent(lambda p, t: t * p[:, 0])))
    problem = Poisson().problem(mesh, bc)

    history = QuasiStaticStepping(steps=2).solve(problem)
    np.testing.assert_allclose(history[-1].dofs, problem.solve(t=1.0).dofs, atol=1e-10)
    np.testing.assert_allclose(history[1].dofs, problem.solve(t=0.5).dofs, atol=1e-10)


def test_refuses_a_problem_with_no_steady_meaning(make_unit_square):
    mesh = make_unit_square(3)
    problem = Heat().problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), 0.0)))
    with pytest.raises(TypeError, match='integrator'):
        QuasiStaticStepping().solve(problem)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        QuasiStaticStepping(steps=0)
    with pytest.raises(ValueError):
        QuasiStaticStepping(t_end=0.0)


# -- bisection and failure --------------------------------------------------------


@dataclass
class _Brittle:
    """A strategy that refuses any increment larger than `max_step` beyond the last
    level it converged, then delegates: deterministic divergence for the retry logic."""
    inner: NewtonSolve
    max_step: float
    reached: float = 0.0
    attempts: list = field(default_factory=list)

    def solve(self, problem, *, initial=None):
        # The pulled edge is prescribed to t * pull with pull = 1, so the level is
        # the largest fixed value.
        t = float(problem.fixed_values.max())
        self.attempts.append(t)
        if t - self.reached > self.max_step + 1e-12:
            raise NewtonDivergence('the step was refused', np.zeros(1), 1, np.inf)
        u = self.inner.solve(problem, initial=initial)
        self.reached = t
        return u


def test_a_refused_step_is_bisected_and_the_walk_completes(make_unit_square):
    mesh = make_unit_square(3)
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, _pulled_block(mesh, pull=1.0))
    brittle = _Brittle(NewtonSolve(), max_step=0.3)

    history = QuasiStaticStepping(steps=1, newton=brittle).solve(problem)
    # 1.0 refused, 0.5 refused, 0.25 taken; then 0.5, 0.75, 1.0 in 0.25 increments.
    assert brittle.attempts == [1.0, 0.5, 0.25, 0.5, 1.0, 0.75, 1.0]
    np.testing.assert_allclose(history.t, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(history[-1].dofs, problem.solve().dofs, atol=1e-9)


def test_exhausted_bisections_carry_the_partial_path(make_unit_square):
    mesh = make_unit_square(3)
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, _pulled_block(mesh, pull=1.0))
    never = _Brittle(NewtonSolve(), max_step=0.0)

    with pytest.raises(SteppingDivergence) as failure:
        QuasiStaticStepping(steps=2, max_bisections=3, newton=never).solve(problem)
    assert failure.value.t == 0.0
    np.testing.assert_allclose(failure.value.history.t, [0.0])
    # 0.5, then 3 bisections toward 0: the budget, no more.
    assert len(never.attempts) == 4
