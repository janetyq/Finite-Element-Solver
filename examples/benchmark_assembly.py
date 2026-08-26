"""Time assembly and the two solve backends versus mesh size, on a 3D elastic box.

Makes the scaling work concrete and guards against regressions: run it before and
after a change to see where the time goes. Sparse matrices moved the cost off the
solve and onto assembly; batching assembly moved it back onto the sparse
factorization, which dominates at any interesting 3D resolution. This benchmark is
what motivated the iterative backend, so it now times both: the direct `splu`
factor+solve against AMG-preconditioned CG. The direct cost grows super-linearly
with fill-in; the AMG-CG cost should overtake it as the mesh grows.

    uv run python -m examples.benchmark_assembly
    uv run python examples/cli.py run timing_benchmark
"""
import logging
import time
from dataclasses import dataclass

from fem.boundary import BCType, BoundaryConditions
from fem.backends import DirectBackend, IterativeBackend
from fem.materials import LinearElasticMaterial
from fem.mesh.structured import create_box_mesh
from fem.plot.plotter import Plotter
from fem.equations import linear_elastic
from fem.regions import everywhere
from fem.system import DiscreteSystem

from demo_registry import Demo, DemoResult, Figure

logging.disable(logging.CRITICAL)  # silence per-solve logging for clean timing

DEFAULT_SIZES = (5, 9, 13, 17, 21)


def _time(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


@dataclass
class Timing:
    """One box size's measurements: kept as numbers, not a formatted string, so
    `demo_timing_benchmark` can chart the scaling trend as well as print it."""
    n: int
    tets: int
    dofs: int
    assemble: float
    direct: float
    amg_cg: float

    def __str__(self) -> str:
        return (f'n={self.n:>3}  tets={self.tets:>8}  dofs={self.dofs:>8}  '
                f'assemble={self.assemble:>6.2f}s  direct={self.direct:>7.2f}s  '
                f'amg_cg={self.amg_cg:>6.2f}s')


def benchmark(n: int) -> Timing:
    mesh = create_box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(n, n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0, 0.0])
    material = LinearElasticMaterial(E=200.0, nu=0.3)

    # Building the LinearProblem assembles the stiffness and the load.
    problem, t_assemble = _time(
        lambda: linear_elastic(mesh, material, bc, source=lambda p: [1.0, 0.0, 0.0])
    )
    A, b = problem.tangent(None), problem.load

    # Each backend factors/preconditions in DiscreteSystem's constructor and solves
    # once; timing the whole construct+solve captures the setup each pays.
    _, t_direct = _time(lambda: DiscreteSystem(A, problem.constraints, DirectBackend()).solve(b))
    _, t_iter = _time(lambda: DiscreteSystem(A, problem.constraints, IterativeBackend()).solve(b))

    return Timing(n, len(mesh.elements), problem.space.n_dofs, t_assemble, t_direct, t_iter)


def demo_timing_benchmark(sizes=DEFAULT_SIZES):
    """Timing of assembly and both solve backends on a 3D elastic box over a range of
    sizes."""
    timings = [benchmark(n) for n in sizes]
    dofs = [t.dofs for t in timings]

    plotter = Plotter(title='Assembly and solve time vs problem size')
    ax = plotter.chart_ax(xlabel='degrees of freedom', ylabel='time (s)')
    ax.loglog(dofs, [t.assemble for t in timings], 'o-', label='assemble')
    ax.loglog(dofs, [t.direct for t in timings], 'o-', label='direct (splu)')
    ax.loglog(dofs, [t.amg_cg for t in timings], 'o-', label='AMG-CG')
    ax.grid(True, which='both', alpha=0.3)

    return DemoResult(
        [Figure(plotter,
                'Direct factorization grows super-linearly with the fill-in a 3D mesh '
                'brings; AMG-preconditioned CG scales closer to linearly, and overtakes '
                'it as the mesh grows: the crossover this benchmark exists to measure.')],
        text='\n'.join(str(t) for t in timings),
    )


DEMOS = [
    # The sweep shows the crossover, so the CLI
    # and the gallery run all five sizes. The test only needs to know that assembly and
    # both backends still compose, which n=5 answers in 0.01s where the full sweep
    # takes 11.6s, over half of it one sparse factorisation at n=21.
    Demo('timing_benchmark', demo_timing_benchmark, section='Accuracy & performance',
         smoke_kwargs={'sizes': (5,)}),
]


if __name__ == '__main__':
    print(demo_timing_benchmark().text)
