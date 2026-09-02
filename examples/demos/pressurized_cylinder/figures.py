"""The figure and summary of the pressurized-cylinder demo, drawn from a `CylinderStudy`."""
import numpy as np

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.pressurized_cylinder import physics
from demos.pressurized_cylinder.physics import CylinderStudy, hill_pressure, run


def _yield_figure(s: CylinderStudy) -> Figure:
    # Three snapshots of the wall, colored by von Mises stress in units of the yield
    # stress so 1.0 *is* the flow stress: the plastic zone reads directly as the
    # flat band at 1.0, and the measured front is drawn on it as a dashed arc.
    figure = Plotter(1, 4, figsize=(15.0, 3.6),
                     title='A pressure vessel yielding from the bore outward')
    top = max(float(vm.max()) for _, _, _, vm in s.showcase) / s.yield_stress
    for i, (p, front, solution, vm) in enumerate(s.showcase):
        yielded = front > s.inner
        state = (f'plastic to r = {front / s.inner:.2f}a' if yielded
                 else f'elastic (peak {vm.max() / s.yield_stress:.2f})')
        figure.plot(solution, vm / s.yield_stress, mode='colored', idx=(0, i),
                    label='von Mises / yield', clim=(0.0, top),
                    colorbar=(i == len(s.showcase) - 1),
                    title=f'p = {p / s.limit_pressure:.0%} of limit: {state}')
        if yielded:
            theta = np.linspace(0.0, np.pi / 2.0, 100)
            figure.get_ax((0, i)).plot(front * np.cos(theta), front * np.sin(theta),
                                       'w--', linewidth=1.2)

    ax = figure.chart_ax(idx=(0, 3), xlabel='pressure / limit pressure',
                         ylabel='plastic front radius / bore radius')
    c = np.linspace(s.inner, s.outer, 200)
    ax.plot(hill_pressure(c, s.inner, s.outer, s.k) / s.limit_pressure, c / s.inner,
            color='tab:red', linestyle='--', label='Hill (perfectly plastic)')
    ax.plot(s.pressures / s.limit_pressure, s.fronts / s.inner, 'o',
            color='tab:blue', label='measured front')
    ax.axvline(s.first_yield / s.limit_pressure, color='gray', linestyle=':',
               label='first yield at the bore')
    ax.axvline(1.0, color='black', linestyle=':',
               label='limit: whole wall flows')
    ax.axhline(s.outer / s.inner, color='gray', linestyle='-.',
               label='outer surface')
    # The takeaway number, written where the reserve it names actually is: between
    # first yield and collapse, below the curve where nothing else is drawn.
    middle = 0.5 * (s.first_yield / s.limit_pressure + 1.0)
    ax.annotate(f'+{100 * (s.reserve - 1):.0f}% pressure\nafter first yield',
                xy=(middle, 1.0 + 0.02 * (s.outer / s.inner - 1.0)),
                ha='center', va='bottom', fontsize='small', color='black')
    worst = float(np.abs(s.fronts - s.hill_fronts).max()) / (s.outer - s.inner)
    ax.set_title(f'Front within {100 * worst:.0f}% of the wall of Hill')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')
    return Figure(
        figure,
        f'One quarter of a thick-walled cylinder (wall ratio '
        f'{s.outer / s.inner:.0f}) under rising internal pressure, colored by von '
        f'Mises stress in units of the yield stress, so 1.0 on the scale is the flow '
        f'stress. Elastic at {s.showcase[0][0] / s.limit_pressure:.0%} of the limit '
        f'pressure, the classic Lame decay from the bore. Past first yield the bore '
        f'cannot be stressed harder: raising the pressure instead recruits more '
        f'wall, a flat band pinned at 1.0 spreading outward to the dashed front. '
        f'The right chart tracks that front against Hill\'s classical solution '
        f'(measured within {100 * worst:.0f}% of the wall thickness; the small lag '
        f'is the slight hardening Hill\'s perfectly plastic wall lacks), and it '
        f'steepens toward the limit: near collapse, a little more pressure sweeps '
        f'the front through a lot of wall.',
        body=[
            'Why the limit pressure is failure: the pressure is held by the integral '
            'of the stress difference sigma_theta - sigma_r across the wall, and a '
            'yielded ring contributes at most its capped flow value. When the front '
            'reaches the outer surface every ring is at its cap, the integral has '
            'hit its maximum 2k ln(b/a), and no stress state balances a higher '
            'pressure: the wall flows without bound. Failure here is equilibrium '
            'running out, not material breaking.',

            f'First yield is therefore not failure: this vessel carries '
            f'{100 * (s.reserve - 1):.0f}% more pressure after the bore yields, with '
            f'essentially no material hardening. The reserve is pure geometry, '
            f'2 ln(b/a) / (1 - a^2/b^2): a thick wall holds under-stressed material '
            f'for the redistribution to recruit, while for a thin wall the ratio '
            f'tends to 1 and first yield and collapse coincide. Sizing a thick '
            f'vessel by first yield alone understates its static capacity by that '
            f'factor; the same contained-yielding argument gives a beam its plastic '
            f'hinge reserve.',

            'The material is Ramberg-Osgood deformation plasticity with a sharp '
            'hardening exponent, near elastic-perfectly-plastic, valid here because '
            'the pressurization is monotonic. What it cannot do is unload: the '
            'compressive residual stress left by overpressurizing and releasing '
            '(autofrettage, the trick behind gun barrels and high-pressure vessels), '
            'and the cyclic checks real vessel codes add on top of static capacity, '
            'live in the loading history, which takes flow-theory plasticity.',
        ])


def _summary(s: CylinderStudy) -> str:
    errors = np.abs(s.fronts - s.hill_fronts) / (s.outer - s.inner)
    return (f'wall ratio b/a            {s.outer / s.inner:.2f}\n'
            f'elements                  {len(s.mesh.elements)} (curved quadratic)\n'
            f'first yield pressure      {s.first_yield:.4f}   '
            f'(Hill: k (1 - a^2/b^2), k = sigma_y/sqrt(3))\n'
            f'limit pressure            {s.limit_pressure:.4f}   (Hill: 2k ln(b/a))\n'
            f'reserve past first yield  {s.reserve:.2f}x   '
            f'(geometry: 2 ln(b/a) / (1 - a^2/b^2); -> 1 for a thin wall)\n'
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
