"""Continuation: `QuasiStaticStepping`, `ArcLengthStepping`, `Problem.with_load_factor`.

The anchors: on a linear problem every level is exactly the load fraction of the full
solution, so the whole path is analytic; on a nonlinear one the walk must land on the
same equilibrium a direct Newton solve finds. Bisection and failure are pinned with a
brittle strategy that refuses increments above a threshold, so the retry logic is
exercised deterministically rather than by hunting for a physically divergent case.

Arc-length is anchored the same way and then taken where load control cannot go: the
straight path of a linear problem to round-off, the imperfect column's knee at the load
`BucklingAnalysis` predicts, and a shallow arch's fold, traversed down the unstable
branch and out the other side while load control jumps across it.
"""
from dataclasses import dataclass, field

import numpy as np
import pytest

from fem.algebra.solve import BacktrackingLineSearch, NewtonDivergence, NewtonSolve
from fem.algebra.stepping import ArcLengthStepping, QuasiStaticStepping, SteppingDivergence
from fem.analysis.buckling import BucklingAnalysis
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.equations import FiniteStrainElastic, Heat, LinearElastic, Poisson
from fem.post.solution import ElasticSolution, PathSolution
from fem.regions import TimeDependent, intersect, on_plane


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


# -- arc-length control -----------------------------------------------------------


def _traction_block(mesh, traction=2.0):
    """A block held at x = 0 and pulled by a traction at x = 1: force control, the
    loading arc-length scales."""
    return Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [traction, 0]),
    )


def _column(length=24.0, height=1.0, n_length=24, n_across=3):
    """A slender column pinned at both ends and compressed by an end traction: the Euler
    case, meshed coarsely enough to trace a path in a second."""
    mesh = box_mesh([[0, 0], [length, height]], (n_length, n_across))
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [None, 0]),
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, height / 2)), [0, 0]),
        Dirichlet(on_plane(0, length), [None, 0]),
        Neumann(on_plane(0, length), [-1.0, 0]),
    )
    return mesh, bc


def _shallow_arch(span=10.0, thickness=0.25, rise=1.0, n_span=30, n_through=2):
    """A thin shallow arch under a downward pressure: a box strip lifted onto a sine
    hump, clamped at both ends. Shallow enough to snap through, which is the fold no
    load path can walk over."""
    strip = box_mesh([[0, 0], [span, thickness]], (n_span, n_through))
    hump = np.zeros((strip.n_vertices, 2))
    hump[:, 1] = rise * np.sin(np.pi * strip.vertices[:, 0] / span)
    mesh = strip.displaced(hump)
    # The loaded surface is the top of the arch, minus the facets at the clamped ends:
    # a traction on a pinned component is refused rather than silently dropped.
    margin = 0.5 * span / n_span

    def crown_surface(points):
        x = points[:, 0]
        above = points[:, 1] - rise * np.sin(np.pi * x / span) > 0.5 * thickness
        return above & (x > margin) & (x < span - margin)

    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, span), [0, 0]),
        Neumann(crown_surface, [0.0, -1.0]),
    )
    return FiniteStrainElastic(E=200.0, nu=0.3).problem(
        mesh, bc, element_type=QuadraticTriangleElement)


def _deflection(step):
    """The largest transverse displacement of a state on the path."""
    return float(np.abs(step.nodal_values[:, 1]).max())


@pytest.mark.parametrize('initial_step', [0.05, 0.25, 0.9])
def test_arc_length_traces_a_linear_problem_exactly(make_unit_square, initial_step):
    """The equilibrium path of a linear problem is the straight line u = lambda K^-1 f,
    and arc-length must trace it to round-off at any step size: the predictor is already
    exact there and the corrector's first constraint root is zero. This pins the bordered
    algebra on its own, with no physics to hide in."""
    mesh = make_unit_square(4)
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, _traction_block(mesh))
    unit = problem.solve().dofs

    path = ArcLengthStepping(initial_step=initial_step, max_steps=5).solve(problem)

    assert isinstance(path, PathSolution)
    assert len(path) == 6
    np.testing.assert_allclose(path.lambdas[:2], [0.0, initial_step], rtol=1e-12)
    for lam, dofs in zip(path.lambdas, path.dofs, strict=True):
        np.testing.assert_allclose(dofs, lam * unit, atol=1e-10 * np.abs(unit).max())
    # Every state on a linear path is stable, so no bracket holds a limit point.
    assert np.all(path.stability == 1)
    assert path.limit_points.size == 0


def test_arc_length_stops_at_a_load_factor_target(make_unit_square):
    mesh = make_unit_square(4)
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, _traction_block(mesh))

    path = ArcLengthStepping(initial_step=0.2, lambda_max=0.5, max_steps=50).solve(problem)

    assert path.lambdas[-1] >= 0.5
    assert path.lambdas[-2] < 0.5


def test_arc_length_stops_at_a_displacement_target(make_unit_square):
    mesh = make_unit_square(4)
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, _traction_block(mesh))
    unit = problem.solve().dofs
    bound = 0.5 * np.abs(unit).max()

    path = ArcLengthStepping(initial_step=0.2, displacement_max=bound,
                             max_steps=50).solve(problem)

    free = problem.partition.free
    assert np.abs(path.dofs[-1][free]).max() >= bound
    assert np.abs(path.dofs[-2][free]).max() < bound


def test_column_path_knees_at_the_critical_load():
    """The post-buckling story: seed the column with a small geometric imperfection in
    its first buckling mode and trace the path. The load-deflection curve rises stiffly,
    knees at the load linearised buckling predicts, then flattens while the column keeps
    bowing, which is the behaviour the critical load alone cannot describe."""
    mesh, bc = _column()
    length, height = 24.0, 1.0
    linear = LinearElastic(E=200.0, nu=0.3).problem(
        mesh, bc, element_type=QuadraticTriangleElement)
    buckling = BucklingAnalysis(n_modes=1).solve(linear)
    critical = buckling.critical_load_factor

    # Only the transverse component of the mode is seeded, so the imperfect mesh keeps
    # every x coordinate and the conditions resolve on it to the same partition.
    mode = buckling.mode(0).nodal_values[:mesh.n_vertices, 1]
    warp = np.zeros((mesh.n_vertices, 2))
    warp[:, 1] = 1e-3 * np.hypot(length, height) * mode / np.abs(mode).max()
    imperfect = mesh.displaced(warp)
    problem = FiniteStrainElastic(E=200.0, nu=0.3).problem(
        imperfect, bc, element_type=QuadraticTriangleElement)
    assert problem.partition == linear.partition

    path = ArcLengthStepping(initial_step=0.3, max_steps=25,
                             displacement_max=1.5).solve(problem)

    deflections = [_deflection(step) for step in path]
    assert all(a < b for a, b in zip(deflections, deflections[1:], strict=False))
    assert deflections[-1] > 1.0, 'the path must reach a visibly bowed state'
    # The knee: the load carried once the column has bowed by a couple of percent of its
    # length sits at the critical load, rounded off by the imperfection.
    knee = next(lam for lam, w in zip(path.lambdas, deflections, strict=True) if w > 0.4)
    assert knee == pytest.approx(critical, rel=0.1)
    # Stiff before the knee, flat after it: the load-deflection slope collapses.
    early = (path.lambdas[1] - path.lambdas[0]) / (deflections[1] - deflections[0])
    late = (path.lambdas[-1] - path.lambdas[-2]) / (deflections[-1] - deflections[-2])
    assert late < 0.1 * early
    # The imperfect path never loses stability: it shadows the buckled branch rather
    # than crossing the bifurcation.
    assert np.all(path.stability == 1)


def test_arc_length_turns_back_through_a_snap_through_fold():
    """The shallow arch: the load rises to a limit point, the path turns back in lambda
    down the unstable branch, and rises again on the snapped-through one. Turning back is
    what arc-length control buys; the stability flag flips at each fold."""
    problem = _shallow_arch()

    path = ArcLengthStepping(initial_step=0.02, max_steps=32,
                             displacement_max=1.5).solve(problem)

    deflections = [_deflection(step) for step in path]
    assert all(a < b for a, b in zip(deflections, deflections[1:], strict=False))
    # The path descends in lambda while the arch keeps deflecting: no load path does that.
    assert any(b < a for a, b in zip(path.lambdas, path.lambdas[1:], strict=False))
    assert path.limit_points.size >= 2
    assert set(np.unique(path.stability)) == {-1, 1}
    # The unstable branch is exactly the stretch between the folds.
    first, last = int(path.limit_points[0]), int(path.limit_points[-1])
    assert np.all(path.stability[first + 1:last + 1] == -1)


def test_load_control_cannot_visit_the_unstable_branch():
    """The contrast that motivates arc-length. Load control prescribes lambda, so at the
    limit load the arch snaps: the walk jumps straight to the far branch (or gives up),
    and no level it converges lands on the descending branch the path traverses."""
    problem = _shallow_arch()
    try:
        history = QuasiStaticStepping(steps=5, t_end=0.05, max_bisections=4).solve(problem)
    except SteppingDivergence as failure:
        history = failure.history
    visited = [_deflection(step) for step in history]
    assert not any(0.4 < w < 1.3 for w in visited)


def test_arc_length_divergence_carries_the_partial_path(make_unit_square):
    """A tolerance no corrector can meet fails every increment; the budget of halvings
    runs out and the partial path comes back on the exception, as a PathSolution."""
    mesh = make_unit_square(3)
    problem = FiniteStrainElastic(E=200, nu=0.3).problem(mesh, _traction_block(mesh, 50.0))
    stepping = ArcLengthStepping(initial_step=0.2, tol=1e-18, max_iters=3, max_retries=1)

    with pytest.raises(SteppingDivergence, match='arc length') as failure:
        stepping.solve(problem)

    assert failure.value.t == 0.0
    assert isinstance(failure.value.history, PathSolution)
    np.testing.assert_allclose(failure.value.history.lambdas, [0.0])


def test_arc_length_refuses_a_problem_with_no_steady_meaning(make_unit_square):
    mesh = make_unit_square(3)
    problem = Heat().problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), 0.0), Source(1.0)))
    with pytest.raises(TypeError, match='integrator'):
        ArcLengthStepping().solve(problem)


def test_arc_length_refuses_a_time_dependent_problem(make_unit_square):
    mesh = make_unit_square(3)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), 0.0),
                    Source(TimeDependent(lambda p, t: t * np.ones(len(p)))))
    problem = Poisson().problem(mesh, bc)
    with pytest.raises(ValueError, match='at\\(t\\)'):
        ArcLengthStepping().solve(problem)


def test_arc_length_refuses_prescribed_displacements(make_unit_square):
    """Force control only: lambda scales the load, so a nonzero Dirichlet value would sit
    at full size along a path that starts from rest."""
    mesh = make_unit_square(3)
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, _pulled_block(mesh, 0.3))
    with pytest.raises(ValueError, match='force control'):
        ArcLengthStepping().solve(problem)


def test_arc_length_refuses_a_problem_with_no_load(make_unit_square):
    mesh = make_unit_square(3)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0, 0]))
    problem = LinearElastic(E=10, nu=0.3).problem(mesh, bc)
    with pytest.raises(ValueError, match='no load'):
        ArcLengthStepping().solve(problem)


@pytest.mark.parametrize('kwargs', [
    {'initial_step': 0.0}, {'max_steps': 0}, {'max_iters': 0},
    {'n_target': 0}, {'max_retries': -1},
])
def test_arc_length_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        ArcLengthStepping(**kwargs)


def test_an_increment_whose_constraint_has_no_real_root_is_retried_shorter(monkeypatch):
    """Where the path curves sharply, the corrected increment can leave the sphere of
    radius ds entirely and the constraint quadratic loses its real roots. The increment
    is then restarted from the last converged state at half the arc length, and the walk
    carries on rather than failing."""
    problem = _shallow_arch()
    rootless = []
    constraint = ArcLengthStepping._constraint_root

    def counted(du, du_r, du_t, step):
        root = constraint(du, du_r, du_t, step)
        if root is None:
            rootless.append(step)
        return root

    monkeypatch.setattr(ArcLengthStepping, '_constraint_root', staticmethod(counted))
    path = ArcLengthStepping(initial_step=0.1, max_steps=20, max_iters=8,
                             displacement_max=1.5).solve(problem)

    assert rootless, 'expected at least one increment to overshoot the path'
    assert len(path) == 21
    assert path.limit_points.size >= 1
