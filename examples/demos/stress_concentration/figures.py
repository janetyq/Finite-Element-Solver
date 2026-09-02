"""The figure and summary of the plate-with-a-hole demo, drawn from a `PlateStudy`."""
from matplotlib.collections import LineCollection

from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.stress_concentration import physics
from demos.stress_concentration.physics import PlateStudy, run


def _pipeline_figure(s: PlateStudy) -> Figure:
    # One figure, three plots: mesh with outline and conditions, the stress, the chart.
    figure = Plotter(1, 3, figsize=(14.0, 3.6),
                     title='From an outline to a stress concentration')
    figure.plot(s.mesh, mode='bc', conditions=s.bc, idx=(0, 0),
                title=f'{len(s.mesh.elements)} triangles (refined from {s.n_initial}), '
                      'with conditions')
    ax0 = figure.get_ax((0, 0))
    # The triangulation under the conditions: thin and grey, below the glyphs.
    ax0.triplot(s.mesh.vertices[:, 0], s.mesh.vertices[:, 1], s.mesh.elements,
                color='0.55', linewidth=0.2, zorder=1.5)
    # The input segments over the triangulation: which of the outline the mesher kept.
    ax0.add_collection(LineCollection(
        list(s.pslg.vertices[s.pslg.segments]), colors='blue', linewidths=1.0, zorder=2.0))
    # Passing the solution draws the P2 field on its own tessellation, rim and all.
    figure.plot(s.solution, s.sigma_xx, mode='colored', idx=(0, 1), label='sigma_xx',
                title='Stress concentration (sigma_xx)')
    ax = figure.chart_ax(idx=(0, 2), xlabel='y', ylabel='sigma_xx / applied')
    # Two runs, below the hole and above it, so nothing is drawn across the gap.
    below = s.y_strip < s.height/2
    ax.plot(s.y_strip[below], s.ratio_strip[below], 'o-', color='tab:blue', markersize=2,
            label='through the hole centre')
    ax.plot(s.y_strip[~below], s.ratio_strip[~below], 'o-', color='tab:blue', markersize=2)
    ax.axhline(s.finite_kt, color='tab:red', linestyle='--',
               label=f'finite plate (Howland): {s.finite_kt:.2f}x')
    ax.axhline(3.0, color='tab:red', linestyle=':', label='infinite plate (Kirsch): 3x')
    ax.axhline(1.0, color='gray', linestyle=':', label='far field')
    ax.set_title(f'Peak {s.peak:.2f}x the applied stress')
    ax.grid(alpha=0.3)
    # Over the flat far field, where the legend hides nothing.
    ax.legend(loc='center left', fontsize='small')
    return Figure(
        figure,
        f'The whole pipeline in one row. Left: the mesh after adaptive refinement, '
        f'grown from {s.n_initial} triangles to {len(s.mesh.elements)} where the '
        f'recovery estimator found the most error, with the input outline in blue '
        f'and the conditions drawn on it; the rim and long edges carry none and are '
        f'traction-free. Middle: the stress sigma_xx on curved quadratic elements, '
        f'crowding into the material either side of the hole and relaxing to the '
        f'applied value within about a diameter. Right: that stress along a strip '
        f'through the hole centre, peaking at {s.peak:.2f}x the applied value at the '
        f'rim. The classic Kirsch factor of 3 is for a hole in an infinite plate; '
        f"Howland's value for a hole {s.hole_over_width:.2f} of this plate's width is "
        f'{s.finite_kt:.2f}.',
        'pipeline',
        thumbnail=True)


def _plasticity_figure(s: PlateStudy) -> Figure:
    # The elastic and yielded von Mises fields on one shared scale, and both read
    # along the strip the elastic chart used, so the flattening is point-for-point.
    figure = Plotter(1, 3, figsize=(14.0, 3.6),
                     title='The same plate in a metal that yields')
    clim = (0.0, s.elastic_vm_peak * s.traction)
    figure.plot(s.solution, s.vm_elastic, mode='colored', idx=(0, 0),
                label='von Mises', clim=clim, colorbar=False,
                title=f'Elastic: peak {s.elastic_vm_peak:.2f}x the applied stress')
    figure.plot(s.plastic_solution, s.vm_plastic, mode='colored', idx=(0, 1),
                label='von Mises', clim=clim,
                title=f'Ramberg-Osgood: peak {s.plastic_vm_peak:.2f}x')
    ax = figure.chart_ax(idx=(0, 2), xlabel='y', ylabel='von Mises / applied')
    below = s.y_strip < s.height / 2
    for run_mask in (below, ~below):
        ax.plot(s.y_strip[run_mask], s.vm_strip_elastic[run_mask], 'o-',
                color='tab:blue', markersize=2,
                label='elastic' if run_mask is below else None)
        ax.plot(s.y_strip[run_mask], s.vm_strip_plastic[run_mask], 'o-',
                color='tab:orange', markersize=2,
                label='yielding metal' if run_mask is below else None)
    ax.axhline(s.yield_stress / s.traction, color='tab:red', linestyle='--',
               label=f'yield stress: {s.yield_stress / s.traction:.1f}x')
    ax.set_title(f'Plastic zone: {100 * s.plastic_zone_over_hole:.0f}% of the hole\'s area')
    ax.grid(alpha=0.3)
    ax.legend(loc='center left', fontsize='small')
    return Figure(
        figure,
        f'The elastic panel asks the rim to carry {s.elastic_vm_peak:.2f}x the applied '
        f'stress, but this metal yields at {s.yield_stress / s.traction:.1f}x: that '
        f'stress cannot exist in it. Re-solved with a Ramberg-Osgood law (deformation '
        f'plasticity, valid for this monotonic load), the rim flows, sheds load to its '
        f'neighbours, and the peak flattens to {s.plastic_vm_peak:.2f}x, just above '
        f'yield, while the plastic lobes beside the hole grow to '
        f'{100 * s.plastic_zone_over_hole:.0f}% of its area. The right chart reads both '
        f'fields along the strip through the hole centre: same equilibrium, same total '
        f'load carried, redistributed by the material\'s refusal to exceed its flow '
        f'stress.',
        'plasticity')


def _summary(s: PlateStudy) -> str:
    return (f'outline points           {len(s.pslg.vertices)}  '
            f'(rectangle + polygonalised rim)\n'
            f'initial elements         {s.n_initial}\n'
            f'adaptively refined to    {len(s.mesh.elements)}\n'
            f'worst angle, initial     {s.initial_worst_angle:.1f} deg   '
            f'(asked for {s.min_angle})\n'
            f'worst angle, refined     {s.worst_angle:.1f} deg   '
            f'(red-green carries no angle guarantee)\n'
            f'boundary edges           {len(s.mesh.boundary)}   '
            f'({s.rim_facets} of them the hole rim, up from {s.initial_rim_facets} before '
            f'refinement)\n'
            f'applied traction         {s.traction:.3g}\n'
            f'hole diameter / height   {s.hole_over_width:.2f}\n'
            f'peak sigma_xx / applied  {s.peak:.2f}   '
            f'(Howland, finite plate: {s.finite_kt:.2f}; Kirsch, infinite plate: 3)\n'
            f'yield stress / applied   {s.yield_stress / s.traction:.1f}\n'
            f'peak von Mises / applied {s.elastic_vm_peak:.2f} elastic -> '
            f'{s.plastic_vm_peak:.2f} yielded\n'
            f'plastic zone / hole area {100 * s.plastic_zone_over_hole:.0f}%')


def demo(**kwargs) -> DemoResult:
    """A plate with a hole, from outline to the stress concentration at its rim, against
    Kirsch and Howland; then the same plate in a metal that yields, where the
    concentration flattens to the flow stress."""
    s = run(**kwargs)
    return DemoResult([_pipeline_figure(s), _plasticity_figure(s)], text=_summary(s))


# The pipeline demo builds its own domain, from an outline through to a stress.
DEMO = Demo('stress_concentration', demo, section='Solids & structures',
            smoke_kwargs={'max_area_fraction': 0.05, 'refinement_iters': 3,
                          'refinement_budget': 200},
            show_source=physics)
