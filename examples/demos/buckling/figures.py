"""The figures and summary of the buckling demo, drawn from a `BucklingStudy`."""
import numpy as np
from demo_registry import Demo, DemoResult, Figure

from demos._charts import hide_x_ticks, share_panel_limits
from demos.buckling import physics
from demos.buckling.physics import BucklingStudy, elastica_load, euler_load, run
from fem.plot.plotter import Plotter


def _buckled(s: BucklingStudy, solution, i):
    """The mesh deformed by mode `i`, scaled so its bow is a fixed fraction of span,
    and the signed transverse displacement to colour it by."""
    mode = solution.mode(i)
    transverse = mode.component(0)[:len(s.mesh.vertices)]
    scale = 0.14 * s.length / np.abs(transverse).max()
    return mode.deformed_mesh(scale), scale * transverse




def _modes_figure(s: BucklingStudy) -> Figure:
    # Upright columns in a row, with one glyph-and-colour key below all of them.
    modes = Plotter(1, s.n_modes, figsize=(3.2 * s.n_modes, 6.0), axis_labels=False,
                    title='Buckling modes of a pinned-pinned column')
    for i in range(s.n_modes):
        shape, colour = _buckled(s, s.pinned, i)
        modes.plot(shape, colour, mode='colored', idx=(0, i), cmap='coolwarm', colorbar=False,
                   title=f'Mode {i+1}: P_cr = {s.pinned_loads[i]:.3g}\n'
                         f'({i+1} half-wave{"s" if i else ""})')
        # The pin/load glyphs, on the deformed shape so the load rides the moving end.
        modes.overlay_supports(s.mesh, s.pinned_bc, idx=(0, i), coords=shape.vertices)
        hide_x_ticks(modes, (0, i))
    share_panel_limits(modes, s.n_modes)
    modes.fig.supxlabel(
        'Blue triangles: the pinned ends, held sideways but free to rotate.\n'
        'Red arrow: the compressive load.\n'
        'Colour: sideways deflection; its sign and amplitude are arbitrary.',
        fontsize='medium')
    return Figure(
        modes,
        'A pinned column buckles into half-sine waves. Mode 1 is a single half-wave '
        'at the lowest load, the shape a real column takes. Each higher mode adds a '
        'half-wave and costs n^2 as much (mode 2 is ~4x mode 1), and is reached only '
        'if the lower ones are braced out. A support at mid-span, a node of mode 2 '
        'but not of mode 1, buys the jump to it. The shapes are the eigenvectors of '
        'K phi = -lambda K_g phi and the load factors its eigenvalues.',
        'modes', thumbnail=True)


def _post_buckling_figure(s: BucklingStudy) -> Figure:
    paths = Plotter(1, 2, figsize=(11.0, 4.6), title='Past the critical load: the equilibrium path')
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    finest = s.paths[-1]

    full = paths.chart_ax(idx=(0, 0), xlabel='mid-span deflection w / L',
                          ylabel='load factor lambda / lambda_cr')
    zoom = paths.chart_ax(idx=(0, 1), xlabel='mid-span deflection w / L',
                          ylabel='load factor lambda / lambda_cr')
    for ax in (full, zoom):
        ax.axhline(1.0, color='tab:red', ls='--', alpha=0.7,
                   label='lambda_cr (linearised buckling)')
        for path, colour in zip(s.paths, colours, strict=False):
            ax.plot(path.deflections / s.length, path.load_ratios, '-', color=colour,
                    lw=1.6, label=path.label)
        # The elastica series, drawn over the smallest imperfection's range: the load an
        # ideal column carries once it has bowed by w.
        w = np.linspace(0.0, finest.deflections.max(), 120)
        ax.plot(w / s.length, elastica_load(w, s.length), ':', color='black', lw=1.4,
                label='elastica  1 + (pi w / L)^2 / 8')
        ax.grid(True, alpha=0.3)
    # The knees are separated by a percent of lambda_cr and reached in the first percent
    # of the span, so the left panel is that corner and the right one everything after.
    full.set_title('The knee sits at the critical load')
    full.set_xlim(0.0, 0.03)
    full.set_ylim(0.0, 1.05)
    zoom.set_title('Past it: the load a bowed column carries')
    zoom.set_xlim(0.0, finest.deflections.max() / s.length)
    zoom.set_ylim(0.9, 1.02)
    return Figure(
        paths,
        'Linearised buckling gives the load; the path says what happens at it. A column '
        'with a small imperfection in the shape of its first mode carries load stiffly, '
        'knees over at lambda_cr, and then keeps bowing at almost constant load: the '
        'critical load is a ceiling on the load, not on the deflection. Smaller '
        'imperfections give sharper knees, flattening onto lambda_cr as the imperfection '
        'goes to zero, which is the perfect column\'s bifurcation the path is rounding '
        'off. The zoom shows what the eigenproblem cannot: past the knee the load rises '
        'again, along the elastica\'s 1 + Theta^2 / 8, so a bowed column is stable and '
        'still carrying. The ends are knife edges here, each held at one point, so the '
        'sections rotate as freely as the elastica assumes.',
        'post_buckling')


def _end_conditions_figure(s: BucklingStudy) -> Figure:
    n = len(s.ends)
    factor_plots = Plotter(1, n, figsize=(2.4 * n, 6.6), axis_labels=False,
                           title='End conditions set the effective length')
    for col, end in enumerate(s.ends):
        shape, colour = _buckled(s, end.solution, 0)
        factor_plots.plot(shape, colour, mode='colored', idx=(0, col), cmap='coolwarm',
                          colorbar=False,
                          title=f'{end.name}\nK = {end.K_measured:.2f} (Euler {end.K_ideal:g})\n'
                                f'P_cr = {end.load:.3g}')
        # Each end's supports drawn on it: a wall clamps, triangles pin, arrows load.
        factor_plots.overlay_supports(s.mesh, end.bc, idx=(0, col), coords=shape.vertices)
    share_panel_limits(factor_plots, n)
    return Figure(
        factor_plots,
        'The same slender column held four ways, buckling at loads spanning 16x. '
        'Clamping an end against rotation shortens the effective length K*L the '
        'column buckles over, from 2L free-standing down to L/2 with both ends fixed, '
        'and the load goes as 1/K^2. The measured K sits within a few percent of '
        'Euler\'s 2, 1, 1/2 and ~0.7; the small excess is a real continuum effect, a '
        'clamp in a solid adding a little Saint-Venant stiffening an ideal beam has none of.',
        'end_conditions')


def _laws_figure(s: BucklingStudy) -> Figure:
    laws = Plotter(1, 2, title="Against Euler's column theory")
    curve = laws.chart_ax(idx=(0, 0), xlabel='length L', ylabel='critical load P_cr')
    curve.loglog(s.sweep_lengths, s.sweep_loads, 'o', color='tab:blue',
                 label=f'computed (slope {s.slope:.2f})')
    dense_L = np.linspace(s.sweep_lengths.min(), s.sweep_lengths.max(), 100)
    curve.loglog(dense_L, euler_load(dense_L, s.height), '-', color='tab:red',
                 alpha=0.6, label='Euler  pi^2 E* I / L^2')
    curve.set_title('Pinned column: P_cr goes as 1/L^2')
    curve.grid(True, which='both', alpha=0.3)

    names = [e.name for e in s.ends]
    bars = laws.chart_ax(idx=(0, 1), xlabel='', ylabel='effective-length factor K')
    x = np.arange(len(names))
    bars.bar(x - 0.2, [e.K_ideal for e in s.ends], 0.4, color='tab:red', alpha=0.6,
             label='Euler')
    bars.bar(x + 0.2, [e.K_measured for e in s.ends], 0.4, color='tab:blue', label='computed')
    bars.set_xticks(x, names, rotation=20, ha='right', fontsize='small')
    bars.set_title('Effective-length factor by end condition')
    bars.grid(True, axis='y', alpha=0.3)
    return Figure(
        laws,
        'Euler\'s column formula gives the buckling load of an ideal slender elastic '
        'column, P_cr = pi^2 E* I / (K L)^2. Left: sweeping the length of a pinned '
        'column, the critical '
        'load falls as 1/L^2 (a slope of -2 on log-log) and lands on it, with '
        'E* = E/(1-nu^2) the plane-strain modulus a 2D solve sees. Right: the '
        'effective-length factor K read back from each end condition\'s buckling load, '
        'against the textbook values.',
        'laws')


def _conditions_figure(s: BucklingStudy) -> Figure:
    conditions = Plotter(panel_aspect=0.7)   # tall and narrow, matching the upright column
    conditions.plot(s.mesh, mode='bc', conditions=s.pinned_bc)
    return Figure(
        conditions,
        'A pinned-pinned column: both ends held across their width (u_y = 0) so they '
        'stay in line but can still rotate, one point anchoring the axial slide, and a '
        'compressive traction on the right. The transverse support and the axial load '
        'share the loaded edge, a roller carrying a tangential traction.',
        'conditions', setup=True)


def _summary(s: BucklingStudy) -> str:
    ratios = '   '.join(f'{name}/pinned {ratio:.2f}' for name, ratio in s.load_ratios.items())
    return ('Euler (1744): an ideal slender column buckles at P_cr = pi^2 E* I / (K L)^2.\n'
            'This demo reproduces it three ways: mode shapes, end conditions, slenderness.\n\n'
            'effective-length factor K (measured vs Euler):\n'
            + '\n'.join(f'  {e.name:<14} {e.K_measured:.3f}  (Euler {e.K_ideal:g})'
                        for e in s.ends)
            + f'\nslenderness law    P_cr ~ L^{s.slope:.2f}   (Euler exponent -2)\n'
            + f'buckling-load ratios (Euler 0.25 : 4 : 2.05):  {ratios}\n\n'
            + 'post-buckling path, lambda/lambda_cr once the column has bowed by w:\n'
            + '\n'.join(f'  {p.label:<34} {_at_deflection(p, s, 0.01):.3f} (w = 0.01 L)   '
                        f'{_at_deflection(p, s, 0.05):.3f} (w = 0.05 L)'
                        for p in s.paths)
            + '\n  elastica at w = 0.05 L: '
            + f'{float(elastica_load(0.05 * s.length, s.length)):.3f}')


def _at_deflection(path, s: BucklingStudy, fraction: float) -> float:
    """The load ratio the path carries at a deflection of `fraction` of the span."""
    return float(np.interp(fraction * s.length, path.deflections, path.load_ratios))


def demo(**kwargs) -> DemoResult:
    """Buckling loads and modes of a slender column, checked against Euler's column
    formula."""
    s = run(**kwargs)
    return DemoResult([
        _modes_figure(s),
        _post_buckling_figure(s),
        _end_conditions_figure(s),
        _laws_figure(s),
        _conditions_figure(s),
    ], text=_summary(s))


# Builds its own columns (several lengths, four end conditions), so it takes no domain.
DEMO = Demo('buckling', demo, section='Solids & structures',
            smoke_kwargs={'n_length': 12, 'n_across': 4, 'n_modes': 2,
                          'sweep_lengths': (12.0, 18.0),
                          'imperfections': (1e-3, 1e-4), 'path_steps': 6},
            show_source=physics)
