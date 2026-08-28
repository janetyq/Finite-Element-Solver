"""The figure and table of the outline-to-mesh demo, drawn from an `OutlineStudy`."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from fem.plot.plotter import Plotter
from fem.plot.helpers import plot_mesh
from fem.mesh.svg import douglas_peucker

from demo_registry import Demo, DemoResult, Figure
from demos.outline_to_mesh import physics
from demos.outline_to_mesh.physics import (
    DEFAULT_SIMPLIFICATION_TOLERANCE, DEFAULT_SVG_FILE, OutlineStudy, close_ring,
    curve_extent, get_curve_from_svg, run, save_curve)


def _explore_simplification(curve, save_file='douglas_peucker_output.json',
                            tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE):
    """Open a slider to explore the Douglas-Peucker simplification of `curve`, starting
    from `tolerance` (a fraction of the curve's extent), and return the curve simplified
    at whatever the slider was left on."""
    d = curve_extent(curve)
    fig, ax = plt.subplots()  # a widget figure, not a Plotter: this path is interactive
    closed_curve = close_ring(curve)
    ax.plot(closed_curve[:, 0], closed_curve[:, 1], color='gray', alpha=0.5)
    plt.subplots_adjust(bottom=0.15)

    initial = close_ring(douglas_peucker(curve, tolerance * d))
    sampled_plot = plt.plot(initial[:, 0], initial[:, 1], 'b-')[0]
    # Starting at zero would leave an untouched slider handing the full outline
    # downstream, which Ruppert's does not finish triangulating.
    slider = Slider(plt.axes((0.15, 0.04, 0.6, 0.03)), 'Epsilon ', 0, d/20,
                    valinit=tolerance * d)
    button = Button(plt.axes((0.85, 0.04, 0.1, 0.04)), 'Save')

    def update(val):
        dp = close_ring(douglas_peucker(curve, slider.val))
        sampled_plot.set_xdata(dp[:, 0])
        sampled_plot.set_ydata(dp[:, 1])
        fig.canvas.draw_idle()

    def save(event):
        save_curve(douglas_peucker(curve, slider.val), save_file)
        print(f'Saved points to {save_file}')

    slider.on_changed(update)
    button.on_clicked(save)
    ax.set_aspect('equal')
    plt.show()

    return douglas_peucker(curve, slider.val)


def _mesh_zoom_inset(ax, mesh, box, loc=(0.57, 0.57, 0.42, 0.42)):
    """Overlay a zoomed inset on `ax` revealing the bare triangulation over `box`.

    `box` is `(x0, x1, y0, y1)` in data coordinates. The inset shows the actual mesh
    under a field drawn fine enough to read smooth. Drawn on white so the lines stay
    legible where the field is dark.
    """
    inset = ax.inset_axes(loc)
    plot_mesh(inset, mesh, color='0.2', linewidth=0.35)
    inset.set_xlim(box[0], box[1])
    inset.set_ylim(box[2], box[3])
    inset.set_aspect('equal')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor('0.35')
    ax.indicate_inset_zoom(inset, edgecolor='0.35', linewidth=0.8, alpha=0.9)


def _zoo_figure(s: OutlineStudy) -> Figure:
    plotter = Plotter(2, 2, axis_labels=False, figsize=(10.5, 10.0),
                      title="One pipeline, any outline: Douglas-Peucker, Ruppert's, solve")
    for k, shape in enumerate(s.shapes):
        idx = divmod(k, 2)
        # A colour scale per cell (the domains differ in size by orders of magnitude) and
        # no colorbar: the shape matters, not the amplitude.
        clim = (0.0, float(shape.u.max()))
        plotter.plot(shape.mesh, shape.u, mode='colored', idx=idx, colorbar=False,
                     clim=clim, empty=True,
                     title=f'{shape.name}: {shape.n_triangles} triangles')
        plot_mesh(plotter.get_ax(idx), shape.mesh, color='0.9', linewidth=0.1)
        if shape.name == 'California':
            # Reveal the real mesh under the smooth field, zoomed onto the San Francisco
            # Bay, where the traced coastline is most intricate.
            v = np.asarray(shape.mesh.vertices)
            lo, hi = v.min(axis=0), v.max(axis=0)
            span = hi - lo
            box = (lo[0] + 0.11 * span[0], lo[0] + 0.23 * span[0],
                   lo[1] + 0.48 * span[1], lo[1] + 0.62 * span[1])
            _mesh_zoom_inset(plotter.get_ax(idx), shape.mesh, box)
    return Figure(
        plotter,
        'Four outlines through one pipeline. Each becomes a planar straight-line '
        'graph, is simplified with Douglas-Peucker where it was traced densely '
        "(California, cloud), then triangulated by Ruppert's algorithm to a "
        'minimum-angle and maximum-area bound. On each mesh the same Poisson '
        'problem is solved, the dome of -div(grad u) = 1 with u = 0 on the '
        'boundary: tallest where the domain is widest and pinched to zero at every '
        'edge and hole. The outlines make different demands. California meshes as '
        "disconnected islands; the cloud's boundary follows its true Bezier "
        "curves; the gear bore is a hole by the even-odd rule; the star's notches "
        'are corners sharper than the bound, which Ruppert meets at the input '
        "angle. The inset zooms into California's mesh, which resolves the traced "
        'coastline and its offshore islands.')


def _summary(s: OutlineStudy) -> str:
    rows = ['outline        pts  triangles  min angle']
    for shape in s.shapes:
        rows.append(f'{shape.name:<14}{shape.n_points:>4}{shape.n_triangles:>10}'
                    f'{shape.worst_angle:>8.0f}')
    return '\n'.join(rows)


def demo(interactive=False, **kwargs) -> DemoResult:
    """Four outlines, traced and generated, meshed and solved with one pipeline."""
    # --interactive first opens a slider to explore the Douglas-Peucker simplification on
    # the California outline.
    if interactive:
        _explore_simplification(get_curve_from_svg(DEFAULT_SVG_FILE))
    s = run(**kwargs)
    return DemoResult([_zoo_figure(s)], text=_summary(s))


# Builds its own outlines, so it takes no domain.
DEMO = Demo('outline_to_mesh', demo, section='Meshing & solving PDEs',
            show_source=physics,
            smoke_kwargs={'svg_tolerance': 0.005, 'max_area_fraction': 0.04})
