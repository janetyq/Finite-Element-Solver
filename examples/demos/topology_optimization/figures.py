"""The figures and summary of the topology optimization demo, drawn from a
`TopologyStudy`."""
from functools import partial

import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.topology_optimization import physics
from demos.topology_optimization.physics import TopologyStudy, run
from domains import beam


def _comparison_figure(s: TopologyStudy) -> Figure:
    solid_disp = np.linalg.norm(s.solid.u.reshape(-1, 2), axis=1)
    # Explicit figsize: two 4:1 panels stacked, each filling its row.
    comparison = Plotter(2, 1, figsize=(6.5, 4.6),
                         title='Half the material, comparable stiffness')
    comparison.plot(s.solid.deformed_mesh(), solid_disp, mode='colored', idx=(0, 0),
                    label='|u|',
                    title=f'Solid: 100% material, compliance {s.compliance_solid:.3f}')
    comparison.plot(s.optimized.deformed_mesh(), s.history.rho[-1], mode='colored',
                    idx=(1, 0), label='density',
                    title=f'Optimized: 50% material, compliance {s.compliance_opt:.3f} '
                          f'({s.ratio:.2f}x)')
    return Figure(
        comparison,
        'The same simply supported beam under the same central load, solid and then '
        'with half its material removed by optimization, both drawn deformed. '
        'Compliance is the work the load does, so it measures deflection under load. '
        f'The optimized truss is only {s.ratio:.2f}x as compliant as the fully solid '
        'block on half the material; what it removed was near the neutral axis, '
        'where it was barely resisting the bending.',
        'comparison')


def _animation_figure(s: TopologyStudy) -> Figure:
    animation = Plotter(title='Topology optimization', panel_aspect=s.aspect)
    animation.plot_animation(s.mesh, s.history.rho, mode='colored', label='density')
    return Figure(
        animation,
        'Density evolving over the SIMP iterations, from an even grey to the '
        'black-and-white truss.',
        'animation')


def _conditions_figure(s: TopologyStudy) -> Figure:
    conditions = Plotter(panel_aspect=s.aspect)
    conditions.plot(s.mesh, mode='bc', bc=s.bc)
    return Figure(
        conditions,
        'Simply supported, pinned at one bottom corner (both directions held) with a '
        'vertical roller at the other (free to slide horizontally), and a downward '
        'load at the top centre.',
        'conditions', setup=True)


def _summary(s: TopologyStudy) -> str:
    return (f'compliance, solid (100% material)     {s.compliance_solid:.4f}\n'
            f'compliance, optimized (50% material)  {s.compliance_opt:.4f}\n'
            f'ratio                                 {s.ratio:.2f}x')


def demo(mesh, **kwargs) -> DemoResult:
    """SIMP topology optimization of a beam to half its material, compared with the
    solid beam."""
    s = run(mesh, **kwargs)
    return DemoResult([
        _comparison_figure(s),
        _animation_figure(s),
        _conditions_figure(s),
    ], text=_summary(s))


# A 4:1 simply supported (MBB) beam, the aspect that optimizes into the classic arch.
DEMO = Demo('topology_optimization', demo, section='Solids & structures',
            domain=partial(beam, 4.0, 1.0, 160), smoke_kwargs={'iters': 3},
            show_source=physics)
