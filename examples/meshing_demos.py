"""Meshing demos: turning a shape into a triangulation, and naming parts of it.

Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run mesh_from_svg
"""
import json
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Slider

from fem.geometry import calculate_triangle_min_angle
from fem.plot.plotter import Plotter
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker
from fem.regions import in_box, intersect, on_plane

from demo_registry import Demo, DemoResult, Figure
from domains import beam

# Resolved against the repo rather than the working directory: the input files ship
# with the project, so a demo should not depend on where it was launched from. Output
# paths stay relative, and so follow the caller's directory.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'california.svg')
CLOUD_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'cloud.svg')

# Douglas-Peucker: drop points that deviate less than this fraction of the curve's
# bounding-box extent. Ruppert's cost grows steeply in point count, so simplifying
# first is what keeps the demo interactive.
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.005

# Ruppert's: any triangle whose area exceeds this fraction of the outline's total area
# gets its circumcenter inserted. The angle bound controls element *shape* but says
# nothing about size, so without this a large region comes back as a handful of
# enormous triangles.
DEFAULT_MAX_AREA_FRACTION = 0.005

def demo_regions(mesh):
    """Name parts of a domain by position, which is how a boundary condition says where
    it applies, and survives a remesh the way a vertex index could not."""
    # The alternative is naming vertex indices, and an index means nothing after a
    # remesh renumbers them. Everything here is written against coordinates, so the
    # same three lines select the same three places on any mesh of this beam, which
    # lets a generated mesh carry boundary conditions at all.
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    clamped = on_plane(0, 0.0)
    loaded = intersect(on_plane(0, w), in_box([None, 0.2*h], [None, 0.8*h]))
    far_half = in_box([w/2, None], [None, None])

    # Sized for the domain plus a row of labels under it: the axes are equal-aspect, so
    # a 4:1 beam in the default square-ish figure is a thin strip, and a legend inside
    # one covers the mesh it is annotating.
    figsize = (9.0, 3.6)

    # The claim these regions are worth anything rests on: resolved fresh against
    # whatever mesh is current, not tied to one triangulation's vertex numbering. Shown
    # rather than asserted: the same three predicates land on the same physical
    # patches on a second, differently-resolved mesh of this beam, whose vertices are
    # numbered nothing like the first's.
    finer = beam(w, h, 90)
    resolved = Plotter(1, 2, title='The same regions, resolved on two different meshes',
                       axis_labels=False, figsize=figsize)
    for col, m in enumerate((mesh, finer)):
        m_centroids = m.vertices[m.elements].mean(axis=1)
        resolved.plot(m, mode='mesh', idx=(0, col), title=f'{len(m.elements)} triangles')
        resolved.plot_highlights(m, [np.flatnonzero(far_half(m_centroids))], ['lightblue'],
                                 [''], mode='elements', idx=(0, col))
        resolved.plot_highlights(
            m,
            [np.flatnonzero(clamped(m.vertices)), np.flatnonzero(loaded(m.vertices))],
            ['red', 'green'], ['', ''], idx=(0, col),
        )
    legend_handles = [
        Patch(facecolor='lightblue', alpha=0.2, label='in_box: far half'),
        Line2D([], [], marker='o', linestyle='', color='red', markersize=5,
              label='on_plane: clamped edge'),
        Line2D([], [], marker='o', linestyle='', color='green', markersize=5,
              label='intersect: loaded patch'),
    ]
    resolved.fig.legend(handles=legend_handles, loc='outside lower center', ncol=3,
                        frameon=False)

    return DemoResult([
        Figure(resolved,
               'Three regions selected by position (on_plane, in_box, and their '
               f'intersect) resolved on two different meshes of the same beam: '
               f'{len(mesh.elements)} triangles, then {len(finer.elements)}, differently '
               'numbered. Every region lands on the same physical patch either way, which '
               'is what lets a boundary condition be placed once and survive whatever '
               'remeshing happens after. The regions themselves are geometric, not '
               "boundary-aware: a plane through the domain's middle would keep only the "
               'two vertices where it meets the edge, once resolved against one.'),
    ])

def get_curve_from_svg(svg_file):
    output = read_svg_to_list_of_path_points(svg_file)
    curve = max(output, key=lambda x: len(x)) # get the longest path
    return np.array(curve)

def close_ring(points):
    """`points` with its first vertex repeated at the end, for plotting.

    A curve read from a closed SVG path comes back as a ring: the closing edge
    from its last point back to its first is implied by wraparound rather than
    stored, which is what `PSLG.from_loops` also assumes. `ax.plot` has no such
    convention, so drawing the points as given leaves that edge missing.
    """
    return np.vstack([points, points[:1]])

def simplify_curve(curve, save_file='douglas_peucker_output.json',
                   tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE, interactive=False):
    """Simplify `curve` with Douglas-Peucker, returning the simplified curve.

    `tolerance` is a fraction of the curve's extent. `interactive=True` opens a slider
    to explore it instead, starting from `tolerance`, and returns whatever it was left
    on. Simplifying directly is the default so that every caller (including ones with
    nobody watching) gets the same result without having to ask for it.
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

def demo_douglas_peucker(svg_file=CLOUD_SVG_FILE, tolerances=(0.005, 0.02, 0.05, 0.15)):
    """Simplify a curve with Douglas-Peucker at a few tolerances, to show what the
    parameter does to the outline before it ever reaches Ruppert's algorithm."""
    curve = get_curve_from_svg(svg_file)
    d = max(np.max(curve, axis=0) - np.min(curve, axis=0))
    closed_curve = close_ring(curve)

    plotter = Plotter(1, len(tolerances), title='Douglas-Peucker at increasing tolerance')
    for i, tolerance in enumerate(tolerances):
        simplified = close_ring(douglas_peucker(curve, tolerance * d))
        ax = plotter.get_ax((0, i))
        ax.plot(closed_curve[:, 0], closed_curve[:, 1], color='gray', linewidth=1.0)
        ax.plot(simplified[:, 0], simplified[:, 1], 'b-o', markersize=3)
        ax.set_title(f'tolerance={tolerance} ({len(simplified) - 1} pts)')

    return DemoResult([
        Figure(plotter,
               f'The same {len(curve)}-point outline simplified at four tolerances, each '
               "a fraction of the outline's extent. A looser tolerance keeps only points "
               'far enough from the line between their neighbours, which is why the '
               'rounder parts of the cloud are the first detail to go.',
               'tolerances'),
    ])

def rupperts_mesh(pslg, min_angle=20, max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Triangulate a PSLG with Ruppert's algorithm; returns (mesh, algorithm)."""
    pslg.validate()
    max_area = None
    if max_area_fraction is not None:
        max_area = max_area_fraction * pslg.area()
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle, max_area=max_area)
    return rupperts.refine(), rupperts

def _draw_rupperts_mesh(plotter, idx, mesh, rupperts, min_angle):
    """Draw the triangulation and its input segments on one panel; return the caption
    fragment describing what came out (how many outlines and triangles, and the angle
    bound the mesh actually held to)."""
    plotter.plot(mesh, mode='mesh', idx=idx, title='Triangulated mesh')
    ax = plotter.get_ax(idx)
    # One collection rather than a plot call per segment: an outline that has been
    # refined runs to hundreds of them.
    ax.add_collection(LineCollection(rupperts.vertices[rupperts.segments],
                                     colors='blue', linewidths=1.0))
    outlines = len(np.unique(rupperts.segment_loops))
    # An input corner sharper than the bound keeps its own angle, so read the
    # bound off the mesh rather than claiming the one that was asked for.
    worst = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()
    held = (f'every angle at least {min_angle} degrees' if worst >= min_angle else
            f'every angle at least {min_angle} degrees bar the input corners already '
            f'sharper than that, the worst {worst:.0f}')
    noun = 'outline' if outlines == 1 else 'outlines'
    return (f"Ruppert's refinement of {outlines} {noun} (blue) into {len(mesh.elements)} "
            f'triangles, {held}')

def demo_mesh_from_svg(svg_file=DEFAULT_SVG_FILE, tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE,
                       interactive=False, min_angle=20,
                       max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Turn an SVG drawing into a mesh: simplify each outline with Douglas-Peucker, then
    triangulate with Ruppert's algorithm."""
    # --interactive opens a slider previewing the simplification tolerance on the largest
    # outline; the tolerance used for meshing is always `tolerance` (per-loop, via
    # read_svg_to_pslg).
    # The two steps are one demo because the first exists for the second: Ruppert's cost
    # is superlinear in the point count it is handed, and an SVG outline traced at screen
    # resolution has thousands. Simplification is what makes the triangulation finish.
    curve = get_curve_from_svg(svg_file)
    simplified = simplify_curve(curve, tolerance=tolerance, interactive=interactive)
    pslg = read_svg_to_pslg(svg_file, tolerance=tolerance)
    mesh, rupperts = rupperts_mesh(pslg, min_angle=min_angle,
                                   max_area_fraction=max_area_fraction)

    # The two stages side by side: the simplified outline handed to Ruppert's (left) and
    # the triangulation it returns (right). `panel_aspect` matches the outline's own
    # width:height so the two tall panels fill the figure rather than floating in it.
    plotter = Plotter(1, 2, title='From an SVG outline to a mesh', axis_labels=False,
                      panel_aspect=0.86)
    ax = plotter.get_ax((0, 0))
    ax.set_title('Douglas-Peucker simplification')
    ax.set_aspect('equal')
    closed_curve = close_ring(curve)
    closed_simplified = close_ring(simplified)
    ax.plot(closed_curve[:, 0], closed_curve[:, 1], color='gray', linewidth=1.0,
            label=f'original ({len(curve)} pts)')
    ax.plot(closed_simplified[:, 0], closed_simplified[:, 1], 'b-',
            label=f'simplified ({len(simplified)} pts)')
    ax.legend(loc='lower left', fontsize=8, frameon=False)
    mesh_caption = _draw_rupperts_mesh(plotter, (0, 1), mesh, rupperts, min_angle)

    return DemoResult([
        Figure(plotter,
               f'Left: the largest outline reduced from {len(curve)} points to '
               f'{len(simplified)} by Douglas-Peucker, at a tolerance set as a fraction of '
               f"the outline's extent (Ruppert's cost grows steeply in the point count). "
               f'Right: {mesh_caption}. The mesh covers what the outlines enclose and '
               f'nothing else, and carries the {len(mesh.boundary)} boundary edges a solver '
               'needs to put conditions on.'),
    ])


DEMOS = [
    # Both mesh to a size cap, which is what makes the figures worth looking at and
    # also most of their cost; the smoke run only needs the code paths. Loosen the cap
    # and nothing else: simplifying the outline further is not reliably cheaper,
    # because it sharpens corners, and refinement spends extra elements around those.
    Demo('mesh_from_svg', demo_mesh_from_svg, section='Meshing a domain',
         smoke_kwargs={'max_area_fraction': 0.05}),
    Demo('douglas_peucker', demo_douglas_peucker, section='Meshing a domain'),
    # Coarse, so individual edges and the selected vertices stay legible, and a beam
    # so the regions are the cantilever's own.
    Demo('regions', demo_regions, section='Meshing a domain',
         domain=partial(beam, 4.0, 1.0, 24)),
]
