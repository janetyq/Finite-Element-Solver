"""The figures and summary of the finned heatsink demo, drawn from a `HeatsinkStudy`."""
import numpy as np
from matplotlib.lines import Line2D

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.heat import physics
from demos.heat.physics import (
    FIN_LENGTH, FIN_THICKNESS, HeatsinkStudy, run, theory_efficiency,
)


def _mark_base(ax, width, kind):
    """Draw the base condition just below the domain, off the coloured field so it does not
    clash with the warm colormap: upward arrows for a Neumann heat flux, a bar for a held
    Dirichlet temperature. The Robin film (every other surface) is left to the legend."""
    y0 = -0.22
    if kind == 'flux':
        xs = np.linspace(0.1 * width, 0.9 * width, 8)
        ax.quiver(xs, np.full_like(xs, y0), np.zeros_like(xs), np.ones_like(xs),
                  color='red', angles='xy', scale_units='xy', scale=1 / 0.17, width=0.010,
                  headwidth=4, headlength=5, clip_on=False, zorder=6)
    else:
        ax.plot([0.05 * width, 0.95 * width], [y0, y0], color='tab:blue', lw=4,
                solid_capstyle='round', clip_on=False, zorder=6)
    ax.set_ylim(bottom=y0 - 0.12)


def _comparison_figure(s: HeatsinkStudy) -> Figure:
    # One colour scale across all four (ambient to the block's fixed-power peak), so the
    # panels compare directly, one shared bar per row on the right.
    clim = (s.u_ambient, max(float(s.u_block_p.max()), s.u_hot))
    comparison = Plotter(2, 2, panel_aspect=1.6, axis_labels=False, figsize=(10.5, 7.2),
                         title='Heatsink vs a solid block of the same size')
    comparison.plot(s.block, s.u_block_p, mode='colored', idx=(0, 0), cmap='inferno',
                    clim=clim, colorbar=False,
                    title=f'Same power in: solid block\nbase +{s.block_rise:.0f} C  '
                          f'(R = {s.r_block:.2f})')
    comparison.plot(s.mesh, s.u_fin_p, mode='colored', idx=(0, 1), cmap='inferno', clim=clim,
                    label='temperature',
                    title=f'Same power in: finned\nbase +{s.fin_rise:.0f} C  '
                          f'(R = {s.r_fin:.2f})')
    comparison.plot(s.block, s.u_block_t, mode='colored', idx=(1, 0), cmap='inferno',
                    clim=clim, colorbar=False,
                    title=f'Base held at {s.u_hot:.0f}: solid block\nsheds Q = {s.q_block:.0f}')
    comparison.plot(s.mesh, s.u_fin_t, mode='colored', idx=(1, 1), cmap='inferno', clim=clim,
                    label='temperature',
                    title=f'Base held at {s.u_hot:.0f}: finned\nsheds Q = {s.q_fin:.0f}  '
                          f'({s.effectiveness:.1f}x on {s.metal_ratio:.2f}x the metal)')
    # Mark only the base, below the field: arrows for the Neumann flux (fixed-power row),
    # a bar for the held Dirichlet base (fixed-temperature row). The Robin film is every
    # other surface, named in the legend.
    for idx, kind in (((0, 0), 'flux'), ((0, 1), 'flux'), ((1, 0), 'held'), ((1, 1), 'held')):
        _mark_base(comparison.get_ax(idx), s.width, kind)
        comparison.get_ax(idx).tick_params(left=False, bottom=False,
                                           labelleft=False, labelbottom=False)
    comparison.fig.legend(handles=[
        Line2D([], [], color='red', marker='^', linestyle='', markersize=9,
               label='Neumann: heat flux into the base'),
        Line2D([], [], color='tab:blue', lw=4, label='Dirichlet: base held hot'),
        Line2D([], [], color='tab:orange', lw=3, label='Robin: film on all other surfaces'),
    ], loc='outside lower center', ncol=3, frameon=False, fontsize='small')
    return Figure(
        comparison,
        'The finned sink against a solid block of the same bounding box, posed two '
        'ways. Top, the same heat flux into each base (a chip of fixed power). The '
        f'block runs {s.block_rise:.0f} C above ambient, the finned sink '
        f'only {s.fin_rise:.0f} C, roughly halving the thermal resistance '
        f'(R {s.r_block:.2f} -> {s.r_fin:.2f}). Bottom, each base held at {s.u_hot:.0f}. The '
        f'finned sink sheds {s.effectiveness:.1f}x the heat with {s.metal_ratio:.2f}x the '
        'metal, since the fins trade material for surface area.',
        'comparison', thumbnail=True)


def _efficiency_figure(s: HeatsinkStudy) -> Figure:
    efficiency = Plotter(1, 1, title='Fin efficiency against beam theory')
    ax = efficiency.chart_ax(xlabel='fin length L', ylabel='fin efficiency (heat shed / ideal)')
    dense = np.linspace(min(s.fin_lengths), max(s.fin_lengths), 100)
    ax.plot(dense, theory_efficiency(s.kappa, FIN_THICKNESS, dense), '-', color='tab:red',
            alpha=0.6, label='theory  tanh(mL)/mL')
    ax.plot(s.fin_lengths, s.eta_fem, 'o', color='tab:blue', label='computed')
    ax.axvline(FIN_LENGTH, color='0.6', ls=':', label=f"this sink's fins (L = {FIN_LENGTH})")
    ax.set_title('Longer fins shed more, but run less efficiently')
    ax.grid(alpha=0.3)
    ax.legend()
    return Figure(
        efficiency,
        'Fin efficiency, the heat a fin sheds over what it would shed with all of it '
        'at the base temperature, against the beam-theory law tanh(mL)/(mL). The '
        'computed fins track it closely. Efficiency falls as fins lengthen, because a '
        "long fin runs cold toward the tip and carries less of its share. This sink's "
        f'fins (L = {FIN_LENGTH}) sit near {s.eta_here:.0%}, trading efficiency for '
        'surface area.',
        'efficiency')


def _animation_figure(s: HeatsinkStudy) -> Figure:
    # Temperature and heat flux side by side, stepping together: a warm colormap for a
    # warming shape on a scale fixed from ambient to the heated base, and the flux on
    # its own scale.
    animation = Plotter(1, 2, panel_aspect=1.6, title='Heatsink warming up')
    animation.plot_animation(
        s.mesh, s.u_values, mode='colored', label='temperature', cmap='inferno',
        clim=(s.u_ambient, s.u_hot), idx=(0, 0),
        titles=[f't = {t:.2f}   base at {s.base_temperature(t):.0f}' for t in s.t_values])
    animation.plot_animation(
        s.mesh, s.flux_values, mode='colored', label='|grad u|', cmap='viridis', idx=(0, 1),
        titles=[f't = {t:.2f}   heat shed {shed:.1f}'
                for t, shed in zip(s.t_values, s.shed_values)])
    return Figure(
        animation,
        'The finned sink warming from a cold start, the transient heat equation '
        'stepped by Crank-Nicolson. Left, temperature: the warming front climbs each '
        'fin and settles into the fin gradient, hot at the root and about '
        f'{s.tip:.0f} at the tips; the title tracks the base temperature as it switches '
        'on. Right, the heat flux magnitude recovered from each step: largest in the '
        'base and at the fin roots, where the gradient is steepest, and fading toward '
        'the tips as the fins run cold; the title tracks the heat shed to ambient '
        f'through the film, which climbs toward the steady {s.q_fin:.1f}.',
        'animation', frames=len(s.t_values))


def _setup_figure(s: HeatsinkStudy) -> Figure:
    setup = Plotter(1, 2, title='How the heatsink is posed')
    setup.plot(s.mesh, mode='bc', bc=s.bc, title='Boundary conditions', idx=(0, 0))
    schedule = setup.chart_ax(idx=(0, 1), xlabel='t', ylabel='base temperature')
    t_dense = np.linspace(0.0, float(s.t_values[-1]), 400)
    schedule.plot(t_dense, [s.base_temperature(t) for t in t_dense], color='tab:red',
                  label='base (Dirichlet)')
    schedule.axhline(s.u_ambient, color='tab:orange', ls='--', label='ambient (Robin film)')
    schedule.set_ylim(s.u_ambient - 10, s.u_hot + 10)
    schedule.set_title(f'The base switches on over {s.ramp:.1f} s')
    schedule.grid(alpha=0.3)
    schedule.legend(loc='lower right')
    return Figure(
        setup,
        'Left, the conditions. The bottom face is a chip switching on: its '
        f'temperature ramps from ambient to {s.u_hot:.0f} over the first {s.ramp:.1f} s '
        'and then holds (right). Every other surface carries a Robin film, '
        'du/dn + kappa*(u - u_ambient) = 0, shedding heat to ambient. The sink starts '
        'cold at ambient, so the transient is a warm-up to the steady dissipating state.',
        'conditions', setup=True)


def _summary(s: HeatsinkStudy) -> str:
    return (f'thermal resistance R (base rise per unit power):\n'
            f'  solid block   {s.r_block:.3f}\n'
            f'  finned sink   {s.r_fin:.3f}   ({s.r_block/s.r_fin:.1f}x lower)\n'
            f'heat shed with the base held {s.u_hot:.0f} (ambient {s.u_ambient:.0f}):\n'
            f'  solid block   {s.q_block:.1f}\n'
            f'  finned sink   {s.q_fin:.1f}   ({s.effectiveness:.1f}x, on '
            f'{s.metal_ratio:.2f}x the metal)\n'
            f'fin efficiency at L = {FIN_LENGTH}:  {s.eta_here:.2f}  (beam theory close)')


def demo(**kwargs) -> DemoResult:
    """Warm a finned heatsink from a cold start, then compare it with a solid block and
    with beam theory."""
    s = run(**kwargs)
    return DemoResult([
        _comparison_figure(s),
        _efficiency_figure(s),
        _animation_figure(s),
        _setup_figure(s),
    ], text=_summary(s))


# Builds its own heatsink and a solid-block baseline, so it takes no domain.
DEMO = Demo('heat', demo, section='Meshing & solving PDEs',
            show_source=physics,
            smoke_kwargs={'max_area_fraction': 0.03, 'steps': 4, 'fin_lengths': (0.8, 2.0)})
