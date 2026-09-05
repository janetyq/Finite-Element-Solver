"""The figures and summary of the L-bracket demo, drawn from a `BracketStudy`."""
import numpy as np
from demo_registry import Demo, DemoResult, Figure

from demos._charts import conditions_figure, tidy_log_axis
from demos.bracket import physics
from demos.bracket.physics import BracketStudy, run
from fem.plot.plotter import Plotter


def _fields_figure(s: BracketStudy) -> Figure:
    # Independent colour scales: the sharp peak would wash the fillet's to a flat colour.
    fields = Plotter(1, 2, title='An L-bracket under a tip load')
    for i, (name, bracket) in enumerate((('Sharp corner', s.sharp),
                                         (f'Fillet r = {s.fillet_radius:g}', s.rounded))):
        mesh, solution = bracket.mesh, bracket.solution
        # Passing the solution draws the P2 field on its tessellation; warp=True deforms it.
        fields.plot(solution, solution.nodal_von_mises(method='l2'), mode='colored',
                    idx=(0, i), warp=True, label='von Mises stress',
                    title=f'{name}\n{len(mesh.elements)} elements, '
                          f'corner peak {bracket.peak:.0f}')
        # Support glyphs at the deformed vertex positions, so the load follows the tip.
        fields.overlay_supports(mesh, s.bc, idx=(0, i), coords=solution.deformed_mesh().vertices)
    return Figure(
        fields,
        'The same bracket, clamped at the top and pulled down at the horizontal '
        'tip, with a sharp inner corner (left) and a filleted one (right). Stress '
        'crowds into the re-entrant corner; the fillet spreads it over a radius and '
        f'cuts the peak by about {s.reduction:.0f}% at this resolution. The colour '
        'scales are independent, since the sharp peak would otherwise flatten the '
        "fillet's own concentration to nothing.",
        'fields', thumbnail=True)


def _singularity_figure(s: BracketStudy) -> Figure:
    sweep = Plotter(1, 1, title='The corner peak against mesh refinement')
    ax = sweep.chart_ax(xlabel='elements', ylabel='von Mises at the inner corner')
    ax.semilogx(s.sharp.sizes, s.sharp.peaks, 'o-', color='tab:red', label='sharp (singular)')
    ax.semilogx(s.rounded.sizes, s.rounded.peaks, 'o-', color='tab:blue',
                label=f'fillet r = {s.fillet_radius:g} (converges)')
    ax.set_title('Sharp corner keeps climbing; the fillet settles')
    # The element counts span under a decade, where a log axis crowds its minor labels.
    ax.grid(True, which='both', alpha=0.3)
    sizes_all = np.concatenate([s.sharp.sizes, s.rounded.sizes])
    mantissas = np.array([1, 1.2, 1.5, 1.8, 2, 2.5, 3, 4, 5, 7])
    nice = np.concatenate([mantissas * 10**k for k in (2, 3, 4)])
    ticks = nice[(nice >= sizes_all.min()) & (nice <= sizes_all.max())]
    if len(ticks) >= 2:
        tidy_log_axis(ax, ticks)
    ax.legend()
    return Figure(
        sweep,
        'The corner von Mises peak as the mesh is adaptively refined into the '
        'corner. The sharp corner is a stress singularity, so its peak climbs without '
        'bound; the "stress" there is a property of the mesh, not of the part. The '
        'fillet removes the singularity, and its peak settles on a finite value. This '
        'is why real parts round their inner corners.',
        'singularity')


def _conditions_figure(s: BracketStudy) -> Figure:
    return conditions_figure(
        s.sharp.mesh, s.bc,
        'The upright limb is clamped at the top; a downward traction pulls the '
        'horizontal tip. Everything else, the inner corner included, is '
        'traction-free.',
        panel_aspect=1.0)


def _summary(s: BracketStudy) -> str:
    return (f'corner von Mises peak (sharp)   {s.sharp.peak:.1f}  '
            f'over {s.sharp.sizes[-1]} elements, still climbing\n'
            f'corner von Mises peak (fillet)  {s.rounded.peak:.1f}  '
            f'over {s.rounded.sizes[-1]} elements, converged\n'
            f'reduction from the fillet       {s.reduction:.0f}%')


def demo(**kwargs) -> DemoResult:
    """The stress at an L-bracket's inner corner, sharp and filleted, as the mesh
    refines into it.

    Solved on quadratic (P2) elements: the sharp bracket on straight `QuadraticTriangleElement`,
    the filleted one on `IsoparametricTriangleElement` so the arc is a true circle rather than
    a polygon. The recovery estimator drives refinement (it reads the curved fillet's flux
    correctly), and the peak is read from the recovered nodal von Mises."""
    s = run(**kwargs)
    return DemoResult([
        _fields_figure(s),
        _singularity_figure(s),
        _conditions_figure(s),
    ], text=_summary(s))


# Builds its own L-brackets (sharp and filleted) from outlines, so it takes no domain.
DEMO = Demo('bracket', demo, section='Solids & structures',
            smoke_kwargs={'max_area_fraction': 0.08, 'n_rounds': 2},
            show_source=physics)
