"""Meshing demos: turning outlines into triangulations, and solving on them.

Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run outline_to_mesh
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from fem.boundary import BoundaryConditions, BCType
from fem.equations import Poisson
from fem.geometry import calculate_triangle_min_angle
from fem.plot.plotter import Plotter
from fem.plot.helpers import plot_mesh
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker
from fem.regions import everywhere
from fem.solver import Solver

from demo_registry import Demo, DemoResult, Figure
from domains import gear_pslg, star_pslg

# Resolved against the repo, so a demo does not depend on where it was launched from.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'california.svg')
CLOUD_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'cloud.svg')

# Douglas-Peucker: drop points that deviate less than this fraction of the curve's
# bounding-box extent. Ruppert's cost grows steeply in point count.
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.005

def get_curve_from_svg(svg_file):
    output = read_svg_to_list_of_path_points(svg_file)
    curve = max(output, key=lambda x: len(x)) # get the longest path
    return np.array(curve)

def close_ring(points):
    """`points` with its first vertex repeated at the end, for plotting.

    A closed SVG path comes back as a ring whose closing edge is implied, as
    `PSLG.from_loops` assumes; `ax.plot` needs it spelled out.
    """
    return np.vstack([points, points[:1]])

def simplify_curve(curve, save_file='douglas_peucker_output.json',
                   tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE, interactive=False):
    """Simplify `curve` with Douglas-Peucker, returning the simplified curve.

    `tolerance` is a fraction of the curve's extent. `interactive=True` opens a slider
    to explore it, starting from `tolerance`, and returns whatever it was left on.
    """
    d = max(np.max(curve, axis=0) - np.min(curve, axis=0))
    if not interactive:
        return douglas_peucker(curve, tolerance * d)

    fig, ax = plt.subplots()  # a widget figure, not a Plotter: this path is interactive
    closed_curve = close_ring(curve)
    ax.plot(closed_curve[:, 0], closed_curve[:, 1], color='gray', alpha=0.5)
    plt.subplots_adjust(bottom=0.15)

    initial = close_ring(douglas_peucker(curve, tolerance * d))
    sampled_plot = plt.plot(initial[:, 0], initial[:, 1], 'b-')[0]
    # Starting at zero would leave an untouched slider handing the full outline
    # downstream, which Ruppert's does not finish triangulating.
    slider = Slider(plt.axes([0.15, 0.04, 0.6, 0.03]), 'Epsilon ', 0, d/20,
                    valinit=tolerance * d)
    button = plt.Button(plt.axes([0.85, 0.04, 0.1, 0.04]), 'Save')

    def update(val):
        epsilon = slider.val
        dp = close_ring(douglas_peucker(curve, epsilon))
        sampled_plot.set_xdata(dp[:, 0])
        sampled_plot.set_ydata(dp[:, 1])
        fig.canvas.draw_idle()

    def save(event):
        epsilon = slider.val
        sampled_curve = douglas_peucker(curve, epsilon)
        with open(save_file, 'w') as f:
            json.dump(sampled_curve.tolist(), f)
        print(f'Saved points to {save_file}')

    slider.on_changed(update)
    button.on_clicked(save)
    ax.set_aspect('equal')
    plt.show()

    return douglas_peucker(curve, slider.val)

def _zoo_shapes(svg_tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE):
    """The outlines the zoo meshes, as (name, PSLG) pairs.

    California and the cloud are traced from `files/*.svg` and simplified on the way in;
    the star and gear are generated (`domains.py`). Each puts a different demand on the
    mesher: disconnected islands, a curved boundary, sharp reentrant corners, and
    repeated teeth around a circular bore.
    """
    return [
        ('California', read_svg_to_pslg(DEFAULT_SVG_FILE, tolerance=svg_tolerance)),
        ('Cloud', read_svg_to_pslg(CLOUD_SVG_FILE, tolerance=svg_tolerance)),
        ('Gear', gear_pslg()),
        ('Star', star_pslg()),
    ]

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

def demo_outline_to_mesh(min_angle=28, max_area_fraction=0.0008, svg_tolerance=0.001,
                         interactive=False):
    """Four outlines, traced and generated, meshed and solved with one pipeline."""
    # --interactive first opens a slider to explore the Douglas-Peucker simplification on
    # the California outline. `svg_tolerance` is finer than the default so the coastline
    # keeps its detail (the raw trace has ~1700 points).
    if interactive:
        simplify_curve(get_curve_from_svg(DEFAULT_SVG_FILE), interactive=True)

    shapes = _zoo_shapes(svg_tolerance)
    plotter = Plotter(2, 2, axis_labels=False, figsize=(10.5, 10.0),
                      title="One pipeline, any outline: Douglas-Peucker, Ruppert's, solve")
    rows = ['outline        pts  triangles  min angle']
    for k, (name, pslg) in enumerate(shapes):
        idx = divmod(k, 2)
        pslg.validate()
        mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area()).refine()
        # The Poisson "dome": a unit source pinned to zero on every boundary, so the
        # field is a picture of the domain itself, tallest where it is widest.
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, everywhere(), 0)
        u = Solver(mesh, Poisson(source=1.0), bc).solve().u
        # A colour scale per cell (the domains differ in size by orders of magnitude) and
        # no colorbar: the shape matters, not the amplitude.
        clim = (0.0, float(u.max()))
        plotter.plot(mesh, u, mode='colored', idx=idx, colorbar=False, clim=clim,
                     empty=True, title=f'{name}: {len(mesh.elements)} triangles')
        plot_mesh(plotter.get_ax(idx), mesh, color='0.9', linewidth=0.1)
        if name == 'California':
            # Reveal the real mesh under the smooth field, zoomed onto the San Francisco
            # Bay, where the traced coastline is most intricate.
            v = np.asarray(mesh.vertices)
            lo, hi = v.min(axis=0), v.max(axis=0)
            span = hi - lo
            box = (lo[0] + 0.11 * span[0], lo[0] + 0.23 * span[0],
                   lo[1] + 0.48 * span[1], lo[1] + 0.62 * span[1])
            _mesh_zoom_inset(plotter.get_ax(idx), mesh, box)
        worst = calculate_triangle_min_angle(
            np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()
        rows.append(f'{name:<14}{len(pslg.vertices):>4}{len(mesh.elements):>10}'
                    f'{worst:>8.0f}')

    return DemoResult([
        Figure(plotter,
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
               'coastline and its offshore islands.')],
        text='\n'.join(rows))


DEMOS = [
    # Builds its own outlines, so it takes no domain.
    Demo('outline_to_mesh', demo_outline_to_mesh, section='Meshing a domain',
         smoke_kwargs={'svg_tolerance': 0.005, 'max_area_fraction': 0.04}),
    # Builds its own two meshes (a structured grid and a Ruppert's mesh), so it takes no domain.
]
