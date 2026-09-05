"""The chart and table of the timing benchmark, drawn from a `BenchmarkStudy`."""
from demo_registry import Demo, DemoResult, Figure

from demos.timing_benchmark import physics
from demos.timing_benchmark.physics import DEFAULT_SIZES, BenchmarkStudy, run
from fem.plot.plotter import Plotter


def _scaling_figure(s: BenchmarkStudy) -> Figure:
    plotter = Plotter(title='Assembly and solve time vs problem size')
    ax = plotter.chart_ax(xlabel='degrees of freedom', ylabel='time (s)')
    ax.loglog(s.dofs, [t.assemble for t in s.timings], 'o-', label='assemble')
    ax.loglog(s.dofs, [t.direct for t in s.timings], 'o-', label='direct (splu)')
    ax.loglog(s.dofs, [t.amg_cg for t in s.timings], 'o-', label='AMG-CG')
    ax.grid(True, which='both', alpha=0.3)
    return Figure(
        plotter,
        'Direct factorization grows super-linearly with the fill-in a 3D mesh '
        'brings; AMG-preconditioned CG scales closer to linearly, and overtakes '
        'it as the mesh grows: the crossover this benchmark exists to measure.')


def demo(sizes=DEFAULT_SIZES) -> DemoResult:
    """Timing of assembly and both solve backends on a 3D elastic box over a range of
    sizes."""
    s = run(sizes)
    return DemoResult([_scaling_figure(s)], text=s.table)


# The sweep shows the crossover, so the CLI and the gallery run all five sizes. The
# test only needs to know that assembly and both backends still compose, which n=5
# answers in 0.01s where the full sweep takes 11.6s, over half of it one sparse
# factorisation at n=21.
DEMO = Demo('timing_benchmark', demo, section='Accuracy & performance',
            show_source=physics, smoke_kwargs={'sizes': (5,)})
