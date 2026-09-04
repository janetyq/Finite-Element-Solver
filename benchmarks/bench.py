"""Phase timings of the solver's main workloads: the before/after table for a PR that
touches a hot path.

Each workload below builds one problem the library is used for and times its phases
(build, factor, solve, recover, estimate, refine), printing one aligned table. Run it on
the base commit and on the branch, on the same machine in the same sitting, and paste
both tables into the PR. Ratios within one run on one machine are meaningful; absolutes
are not: the same laptop drifted 3x over one day under thermal throttling, and CI runners
differ in CPU, so nothing here belongs in CI or in a threshold test. The count-based
contracts in `tests/test_perf_contracts.py` are what guards a regression automatically.

    uv run python benchmarks/bench.py --quick              # every workload, ~30 s
    uv run python benchmarks/bench.py                      # audit sizes, a few minutes
    uv run python benchmarks/bench.py --only poisson2d newton
    uv run python benchmarks/bench.py --quick --json benchmarks/history.jsonl

`--json` appends one JSON record per phase (commit, timestamp, library versions,
workload, phase, seconds) to the file, one record per line, so a history accumulates
without any of it being parsed here.

The workloads and the default sizes are those of `attic/performance-audit-2026-09-03.md`
section 1; `--quick` shrinks each so the whole set runs in about half a minute. The
sizes are fixed on purpose: a table is only comparable to another table of the same
sizes.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fem.algebra.backends import IterativeBackend
from fem.algebra.integrators import ThetaMethod
from fem.analysis.adaptivity import AdaptiveRefinement
from fem.analysis.design import DesignOptimizer, SIMPModel, calculate_smoothing_matrix
from fem.analysis.estimators import RecoveryEstimator, ResidualEstimator
from fem.analysis.modal import ModalAnalysis
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.elements import QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.curves import Circle
from fem.mesh.outline import Outline
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.structured import box_mesh
from fem.physics.equations import FiniteStrainElastic, Heat, LinearElastic, Poisson
from fem.regions import TimeDependent, everywhere, on_plane

Phases = list[tuple[str, float]]
Sizes = dict[str, Any]

# One entry per workload: the sizes the audit measured at, and the quick preset.
FULL: dict[str, Sizes] = {
    'elastic3d': dict(n=17),
    'poisson2d': dict(p1=300, p2=150),
    'newton': dict(p1=(80, 40), p2=(40, 20)),
    'heat': dict(n=150, steps=200),
    'adapt': dict(n=20, max_triangles=40_000),
    'simp': dict(resolution=(121, 41), iters=15),
    'ruppert': dict(max_areas=(0.005, 0.002)),
    'redgreen': dict(n=150),
    'modal': dict(resolution=(81, 21)),
}
QUICK: dict[str, Sizes] = {
    'elastic3d': dict(n=11),
    'poisson2d': dict(p1=150, p2=75),
    'newton': dict(p1=(60, 30), p2=(30, 15)),
    'heat': dict(n=100, steps=100),
    'adapt': dict(n=12, max_triangles=4_000),
    'simp': dict(resolution=(61, 21), iters=5),
    'ruppert': dict(max_areas=(0.05, 0.02)),
    'redgreen': dict(n=60),
    'modal': dict(resolution=(41, 11)),
}


class Stopwatch:
    '''Records the time between successive `lap` calls as named phases.'''

    def __init__(self) -> None:
        self.phases: Phases = []
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.phases.append((name, now - self._t))
        self._t = now

    def restart(self) -> None:
        self._t = time.perf_counter()


@contextlib.contextmanager
def quiet_logging() -> Iterator[None]:
    '''Silence the solvers' progress logging for the duration of a run.'''
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


# -- the workloads ---------------------------------------------------------------------


def elastic3d(sizes: Sizes) -> Phases:
    '''3D linear elasticity on a box: the build phases, both solve backends, the recoveries.'''
    n = int(sizes['n'])
    watch = Stopwatch()
    mesh = box_mesh([[0, 0, 0], [1, 1, 1]], (n, n, n))
    watch.lap(f'box_mesh ({mesh.n_elements} tets)')
    equation = LinearElastic(E=200.0, nu=0.3)
    bc = Conditions(Dirichlet(everywhere(), [0.0, 0.0, 0.0]), Source([1.0, 0.0, 0.0]))
    problem = equation.problem(mesh, bc)
    watch.lap(f'equation.problem ({problem.space.n_dofs} dofs)')
    solution = problem.solve()
    watch.lap('solve direct')
    problem.solve(backend=IterativeBackend())
    watch.lap('solve AMG-CG')
    solution.nodal_stress('average')
    watch.lap("nodal_stress('average')")
    solution.nodal_stress('l2')
    watch.lap("nodal_stress('l2')")
    return watch.phases


def poisson2d(sizes: Sizes) -> Phases:
    '''2D Poisson on the unit square, P1 then P2: build, both backends, recoveries, estimators.'''
    phases: Phases = []
    bc = Conditions(Dirichlet(on_plane(0, 0.0), 0.0), Neumann(on_plane(0, 1.0), 1.0),
                    Source(lambda p: [np.sin(np.pi * p[:, 0])]))
    for label, n, element_type in (('P1', sizes['p1'], None), ('P2', sizes['p2'], QuadraticTriangleElement)):
        n = int(n)
        watch = Stopwatch()
        mesh = box_mesh([[0, 0], [1, 1]], (n, n))
        problem = Poisson().problem(mesh, bc, element_type=element_type)
        watch.lap(f'{label} equation.problem ({mesh.n_elements} tris, {problem.space.n_dofs} dofs)')
        solution = problem.solve()
        watch.lap(f'{label} solve direct')
        problem.solve(backend=IterativeBackend())
        watch.lap(f'{label} solve AMG-CG')
        solution.nodal_gradient('average')
        watch.lap(f"{label} nodal_gradient('average')")
        solution.nodal_gradient('l2')
        watch.lap(f"{label} nodal_gradient('l2')")
        RecoveryEstimator().estimate(problem, solution)
        watch.lap(f'{label} RecoveryEstimator')
        ResidualEstimator().estimate(problem, solution)
        watch.lap(f'{label} ResidualEstimator')
        phases += watch.phases
    return phases


def newton(sizes: Sizes) -> Phases:
    '''Line-searched Newton on a St-Venant-Kirchhoff cantilever, P1 then P2.'''
    phases: Phases = []
    equation = FiniteStrainElastic(E=10.0, nu=0.3)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]), Neumann(on_plane(0, 2.0), [0.0, -0.02]))
    for label, resolution, element_type in (('P1', sizes['p1'], None), ('P2', sizes['p2'], QuadraticTriangleElement)):
        mesh = box_mesh([[0, 0], [2, 1]], tuple(resolution))
        problem = equation.problem(mesh, bc, element_type=element_type)
        watch = Stopwatch()
        problem.solve()
        watch.lap(f'{label} Newton solve ({problem.space.n_dofs} dofs)')
        phases += watch.phases
    return phases


def heat(sizes: Sizes) -> Phases:
    '''ThetaMethod stepping of the heat equation, with constant and with time-dependent data.'''
    n, steps = int(sizes['n']), int(sizes['steps'])
    mesh = box_mesh([[0, 0], [1, 1]], (n, n))
    equation = Heat(conductivity=1.0)
    cases = (
        ('constant data', Conditions(Dirichlet(everywhere(), 0.0), Source(1.0))),
        ('TimeDependent source and Dirichlet', Conditions(
            Dirichlet(everywhere(), TimeDependent(lambda p, t: np.sin(t) * np.ones(len(p)))),
            Source(TimeDependent(lambda p, t: [np.sin(t) * p[:, 0]])))),
    )
    phases: Phases = []
    for label, bc in cases:
        problem = equation.problem(mesh, bc)
        watch = Stopwatch()
        ThetaMethod(dt=1e-3, steps=steps).solve(problem)
        watch.lap(f'{steps} steps, {label} ({problem.space.n_dofs} dofs)')
        phases += watch.phases
    return phases


def adapt(sizes: Sizes) -> Phases:
    '''AdaptiveRefinement of a Poisson problem with a peaked source, each estimator in turn.'''
    n, max_triangles = int(sizes['n']), int(sizes['max_triangles'])
    mesh = box_mesh([[0, 0], [1, 1]], (n, n))
    bc = Conditions(Dirichlet(everywhere(), 0.0), Source(
        lambda p: [np.exp(-200 * ((p[:, 0] - 0.5) ** 2 + (p[:, 1] - 0.5) ** 2))]))
    phases: Phases = []
    for estimator in (ResidualEstimator(), RecoveryEstimator()):
        driver = AdaptiveRefinement(mesh, lambda m: Poisson().problem(m, bc), estimator,
                                    max_triangles=max_triangles, max_iters=40, refine_fraction=0.5)
        watch = Stopwatch()
        driver.run()
        watch.lap(f'{type(estimator).__name__} to {driver.mesh.n_elements} tris')
        phases += watch.phases
    return phases


def simp(sizes: Sizes) -> Phases:
    '''SIMP compliance minimisation of a cantilever, a fixed number of iterations.'''
    resolution, iters = tuple(sizes['resolution']), int(sizes['iters'])
    mesh = box_mesh([[0, 0], [3, 1]], resolution)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0]), Neumann(on_plane(0, 3.0), [0.0, -1.0]))
    template = LinearElastic(E=1.0, nu=0.3).problem(mesh, bc)
    watch = Stopwatch()
    model = SIMPModel(template, sensitivity_filter=calculate_smoothing_matrix(mesh, 0.06))
    watch.lap(f'SIMPModel + filter ({mesh.n_elements} tris)')
    DesignOptimizer(model, iters=iters).run()
    watch.lap(f'{iters} iterations ({template.space.n_dofs} dofs)')
    return watch.phases


def ruppert(sizes: Sizes) -> Phases:
    '''Ruppert refinement of a plate with a circular hole, at two area caps.'''
    plate = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 3.0], [0.0, 3.0]])
    outline = Outline([Outline.from_polygons([plate]).loops[0], Circle([3.0, 1.5], 0.3)])
    watch = Stopwatch()
    for max_area in sizes['max_areas']:
        mesh = outline.mesh(max_area=max_area)
        watch.lap(f'max_area={max_area}: {mesh.n_elements} tris')
    return watch.phases


def redgreen(sizes: Sizes) -> Phases:
    '''RedGreenRefiner: build over a box mesh, refine everything, then refine a seventh.'''
    n = int(sizes['n'])
    mesh = box_mesh([[0, 0], [1, 1]], (n, n))
    watch = Stopwatch()
    refiner = RedGreenRefiner(mesh)
    watch.lap(f'RedGreenRefiner.__init__ ({mesh.n_elements} tris)')
    refined = refiner.refine(range(mesh.n_elements))
    watch.lap(f'refine all -> {refined.n_elements}')
    again = refiner.refine(range(0, refined.n_elements, 7))
    watch.lap(f'refine 1/7 -> {again.n_elements}')
    return watch.phases


def modal(sizes: Sizes) -> Phases:
    '''ModalAnalysis of a clamped P2 elastic beam: shift-invert eigsh on the free block.'''
    resolution = tuple(sizes['resolution'])
    mesh = box_mesh([[0, 0], [8, 1]], resolution)
    equation = LinearElastic(E=200e3, nu=0.3, density=1.0)
    problem = equation.problem(mesh, Conditions(Dirichlet(on_plane(0, 0.0), [0.0, 0.0])),
                               element_type=QuadraticTriangleElement)
    watch = Stopwatch()
    ModalAnalysis(n_modes=6).solve(problem)
    watch.lap(f'6 modes ({problem.space.n_dofs} dofs)')
    return watch.phases


WORKLOADS: dict[str, Callable[[Sizes], Phases]] = {
    'elastic3d': elastic3d,
    'poisson2d': poisson2d,
    'newton': newton,
    'heat': heat,
    'adapt': adapt,
    'simp': simp,
    'ruppert': ruppert,
    'redgreen': redgreen,
    'modal': modal,
}


# -- the runner ---------------------------------------------------------------------------


def _commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _versions() -> dict[str, str]:
    import pyamg
    import scipy
    return {'numpy': np.__version__, 'scipy': scipy.__version__, 'pyamg': pyamg.__version__,
            'python': sys.version.split()[0]}


def run(names: list[str], sizes: dict[str, Sizes]) -> dict[str, Phases]:
    '''Run the named workloads at `sizes`, printing each table as it completes.'''
    results: dict[str, Phases] = {}
    with quiet_logging():
        for name in names:
            print(f'\n== {name} ==')
            phases = WORKLOADS[name](sizes[name])
            width = max(len(label) for label, _ in phases)
            for label, seconds in phases:
                print(f'    {label:<{width}s}  {seconds:8.3f} s')
            results[name] = phases
    return results


def append_json(path: Path, results: dict[str, Phases], sizes_name: str) -> None:
    '''Append one record per phase, one JSON object per line.'''
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    commit, versions = _commit(), _versions()
    with path.open('a', encoding='utf-8') as out:
        for workload, phases in results.items():
            for phase, seconds in phases:
                out.write(json.dumps({
                    'commit': commit, 'timestamp': stamp, 'versions': versions,
                    'preset': sizes_name, 'workload': workload, 'phase': phase,
                    'seconds': round(seconds, 4),
                }) + '\n')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument('--quick', action='store_true', help='the small preset, about 30 s in all')
    parser.add_argument('--only', nargs='+', choices=sorted(WORKLOADS), metavar='WORKLOAD',
                        help='run only these workloads')
    parser.add_argument('--json', type=Path, metavar='PATH',
                        help='append one JSON record per phase to this file')
    args = parser.parse_args(argv)

    names = args.only if args.only else list(WORKLOADS)
    sizes_name = 'quick' if args.quick else 'full'
    sizes = QUICK if args.quick else FULL
    print(f'commit {_commit()}  preset {sizes_name}  '
          + '  '.join(f'{k} {v}' for k, v in _versions().items()))
    started = time.perf_counter()
    results = run(names, sizes)
    print(f'\ntotal {time.perf_counter() - started:.1f} s')
    if args.json is not None:
        append_json(args.json, results, sizes_name)


if __name__ == '__main__':
    main()
