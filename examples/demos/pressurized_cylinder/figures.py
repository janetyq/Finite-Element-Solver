"""The figure and summary of the pressurized-cylinder demo, drawn from a `CylinderStudy`."""
import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.pressurized_cylinder import physics
from demos.pressurized_cylinder.physics import CylinderStudy, hill_pressure, run


def _yield_figure(s: CylinderStudy) -> Figure:
    # Three snapshots of the wall on one shared scale, then the measured front
    # against Hill's curve. The von Mises field saturates at the flow stress, so the
    # plastic zone reads as the flat-colored annulus at the top of the scale.
    figure = Plotter(1, 4, figsize=(15.0, 3.6),
                     title='A pressure vessel yielding from the bore outward')
    top = max(float(vm.max()) for _, _, vm in s.showcase)
    for i, (p, solution, vm) in enumerate(s.showcase):
        figure.plot(solution, vm, mode='colored', idx=(0, i), label='von Mises',
                    clim=(0.0, top), colorbar=(i == len(s.showcase) - 1),
                    title=f'p = {p / s.limit_pressure:.0%} of the limit pressure')

    ax = figure.chart_ax(idx=(0, 3), xlabel='pressure / limit pressure',
                         ylabel='plastic front radius / bore radius')
    c = np.linspace(s.inner, s.outer, 200)
    ax.plot(hill_pressure(c, s.inner, s.outer, s.k) / s.limit_pressure, c / s.inner,
            color='tab:red', linestyle='--', label='Hill (perfectly plastic)')
    ax.plot(s.pressures / s.limit_pressure, s.fronts / s.inner, 'o',
            color='tab:blue', label='measured front')
    ax.axvline(s.first_yield / s.limit_pressure, color='gray', linestyle=':',
               label='first yield at the bore')
    ax.axhline(s.outer / s.inner, color='gray', linestyle='-.',
               label='outer surface: unbounded flow')
    worst = float(np.abs(s.fronts - s.hill_fronts).max()) / (s.outer - s.inner)
    ax.set_title(f'Front within {100 * worst:.0f}% of the wall of Hill')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')
    return Figure(
        figure,
        f'One quarter of a thick-walled cylinder (wall ratio '
        f'{s.outer / s.inner:.0f}) under rising internal pressure, von Mises stress '
        f'on one shared scale. At {s.showcase[0][0] / s.limit_pressure:.0%} of the '
        f'limit pressure the wall is elastic, most stressed at the bore. Past first '
        f'yield the bore cannot carry more: a plastic annulus (the flat color at the '
        f'top of the scale) spreads outward as the pressure rises, and the load is '
        f'carried by ever less elastic wall. The right chart reads that front '
        f'against Hill\'s classical elastic-plastic solution: measured within '
        f'{100 * worst:.0f}% of the wall thickness everywhere, the small lag being '
        f'the hardening the Ramberg-Osgood curve keeps and Hill\'s perfect '
        f'plasticity does not.',
        body=[
            'The material is Ramberg-Osgood deformation plasticity with a sharp '
            'hardening exponent, near elastic-perfectly-plastic. Deformation theory '
            'is valid here because the pressurization is monotonic: stress is a '
            'function of the current strain, each pressure is an independent '
            'equilibrium, and the sweep matches what incremental plasticity would '
            'predict along this loading path.',

            'What it cannot do is unload. Overpressurizing a vessel on purpose and '
            'releasing it leaves compressive residual stress at the bore '
            '(autofrettage), which is why gun barrels and high-pressure vessels are '
            'made this way; that residual state lives in the loading history, which '
            'takes flow-theory plasticity. The limit pressure is real either way: at '
            '2k ln(b/a) the whole wall flows and equilibrium is lost.',
        ])


def _summary(s: CylinderStudy) -> str:
    errors = np.abs(s.fronts - s.hill_fronts) / (s.outer - s.inner)
    return (f'wall ratio b/a            {s.outer / s.inner:.2f}\n'
            f'elements                  {len(s.mesh.elements)} (curved quadratic)\n'
            f'first yield pressure      {s.first_yield:.4f}   '
            f'(Hill: k (1 - a^2/b^2), k = sigma_y/sqrt(3))\n'
            f'limit pressure            {s.limit_pressure:.4f}   (Hill: 2k ln(b/a))\n'
            f'pressures swept           {s.pressures[0]:.3f} .. {s.pressures[-1]:.3f} '
            f'({len(s.pressures)} solves, each seeded with the last)\n'
            f'front vs Hill             within {100 * errors.max():.1f}% of the wall '
            f'(hardening exponent {s.hardening_exponent:.0f})')


def demo(**kwargs) -> DemoResult:
    """A thick-walled cylinder pressurized past first yield: the plastic front marches
    outward through the wall, tracked against Hill's classical solution."""
    s = run(**kwargs)
    return DemoResult([_yield_figure(s)], text=_summary(s))


# Builds its own quarter-annulus domain, so the curved bore carries the pressure.
DEMO = Demo('pressurized_cylinder', demo, section='Solids & structures',
            smoke_kwargs={'n_pressures': 3, 'max_area_fraction': 0.02,
                          'resolution': 0.08},
            show_source=physics)
