"""The figure and table of the convergence demo, drawn from a `RatesStudy`."""
from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos._charts import tidy_log_axis
from demos.convergence import physics
from demos.convergence.physics import RatesStudy, run


def _plot_study(ax, study, label, colour, reference_order, xlabel):
    """One measured curve plus the power law it is being held to."""
    ax.loglog(study.step, study.error, 'o-', color=colour,
              label=f'{label} (order {study.fitted_order:.2f})')
    # Anchored at the coarsest point, so the two lines start together and any gap is
    # the measured rate differing from the reference rather than an offset between them.
    reference = study.error[0] * (study.step / study.step[0])**reference_order
    ax.loglog(study.step, reference, '--', color=colour, alpha=0.4,
              label=f'{xlabel}^{reference_order}')




def _rates_figure(s: RatesStudy) -> Figure:
    plotter = Plotter(2, 2, figsize=(10.0, 8.0),
                      title='Convergence against manufactured solutions')
    space = plotter.chart_ax(idx=(0, 0), xlabel='h', ylabel='L2 error')
    _plot_study(space, s.poisson, 'Poisson, P1', 'tab:blue', 2, 'h')
    _plot_study(space, s.elastic, 'Elasticity, P1', 'tab:green', 2, 'h')
    _plot_study(space, s.p2, 'Poisson, P2', 'tab:orange', 3, 'h')
    space.set_title('Space: P1 is second order, P2 third')
    tidy_log_axis(space, s.poisson.step)

    cost = plotter.chart_ax(idx=(0, 1), xlabel='degrees of freedom', ylabel='L2 error')
    cost.loglog(s.p1_dofs, s.poisson.error, 'o-', color='tab:blue', label='P1')
    cost.loglog(s.p2_dofs, s.p2.error, 'o-', color='tab:orange', label='P2')
    cost.set_title('Cost: P2 reaches a given accuracy first')
    cost.grid(True, which='both', alpha=0.3)
    cost.legend()

    time = plotter.chart_ax(idx=(1, 0), xlabel='dt', ylabel='L2 error')
    _plot_study(time, s.crank_nicolson, 'Crank-Nicolson', 'tab:blue', 2, 'dt')
    _plot_study(time, s.backward_euler, 'Backward Euler', 'tab:red', 1, 'dt')
    time.set_title("Time: the order is theta's to choose")
    tidy_log_axis(time, s.crank_nicolson.step)

    load = plotter.chart_ax(idx=(1, 1), xlabel='h', ylabel='L2 error')
    _plot_study(load, s.nodal, 'source at vertices', 'tab:red', 2, 'h')
    _plot_study(load, s.sampled, 'source at quadrature points', 'tab:blue', 2, 'h')
    load.set_title('Load: sampling the source wins the constant')
    tidy_log_axis(load, s.nodal.step)
    return Figure(
        plotter,
        'Top left: on the same meshes, halving h quarters the P1 error (order 2), '
        'for a scalar unknown and for a coupled vector one alike, and divides the P2 '
        'error by eight (order 3). Top right: the same errors against the number of '
        'unknowns; P2 spends more DOFs per element but reaches a given accuracy with '
        'fewer of them. Bottom left: the error against the time step, where backward '
        'Euler is first order and Crank-Nicolson second, for the same cost per step. '
        'Bottom right: an oscillatory source read only at the vertices against one '
        'sampled at the quadrature points. Both are second order; the sampled load '
        'is about 3x more accurate on every mesh.')


def _summary(s: RatesStudy) -> str:
    rows = ['                      fitted order   expected']
    for name, study, expected in s.table:
        rows.append(f'{name:<22}{study.fitted_order:>9.2f}{expected:>11}')
    return '\n'.join(rows)


def demo(**kwargs) -> DemoResult:
    """Convergence rates in space and time against manufactured solutions, P1 against
    P2, and the load built two ways."""
    s = run(**kwargs)
    return DemoResult([_rates_figure(s)], text=_summary(s))


# Builds its own refinement sequence; the smoke run keeps the two coarsest meshes.
DEMO = Demo('convergence', demo, section='Accuracy & performance',
            smoke_kwargs={'resolutions': (11, 21), 'elastic_resolutions': (9, 17),
                          'step_counts': (16, 32), 'poisson_3d_resolutions': (5, 9)},
            show_source=physics)
