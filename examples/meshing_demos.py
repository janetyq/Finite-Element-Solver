"""Meshing demos: turning a shape into a triangulation, and naming parts of it.

Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run mesh_from_svg
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from fem.boundary import BoundaryConditions, BCType
from fem.equations import LinearElastic
from fem.geometry import calculate_triangle_min_angle
from fem.plot.plotter import Plotter
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import PSLG, read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker
from fem.regions import on_plane
from fem.solver import Solver

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

def demo_regions(length=4.0, height=1.0, n_structured=60, min_angle=30,
                 max_area_fraction=0.0009, E=200.0, nu=0.3, traction=0.5):
    """Solve one cantilever on two unrelated triangulations of the same beam, a structured
    grid and an unstructured Ruppert's mesh, to show position-based boundary conditions
    resolving against whichever mesh is current: one specification, two meshes, one solve."""
    # The alternative is naming vertex indices, and an index means nothing after a remesh
    # renumbers them. The clamp and the load below are written once against coordinates, so
    # the same two lines select the same physical edges on any triangulation of the beam.
    # The two meshes here are the same size but built two different ways, so their vertices
    # are numbered nothing alike; the same specification resolves against each and drives
    # the same solve, which is what makes a mesh interchangeable at all. A whole-edge load,
    # not a sub-patch: an edge holds the same total length on any mesh, where a patch's
    # boundary would fall between different nodes on each and apply a slightly different
    # resultant, a mesh dependence that has nothing to do with the point being made here.
    clamped = on_plane(0, 0.0)
    loaded = on_plane(0, length)

    def make_bc():
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, clamped, [0, 0])         # clamp the left edge
        bc.add(BCType.NEUMANN, loaded, [0, -traction])    # pull the right edge down
        return bc

    pslg = PSLG.from_loops([np.array([[0.0, 0.0], [length, 0.0],
                                      [length, height], [0.0, height]])])
    pslg.validate()
    meshes = [
        ('structured grid', beam(length, height, n_structured)),
        ("Ruppert's mesh", RuppertsAlgorithm(pslg, min_angle=min_angle,
                                             max_area=max_area_fraction * pslg.area()).refine()),
    ]
    solutions = [Solver(m, LinearElastic(E, nu), make_bc()).solve() for _, m in meshes]
    disp = [np.linalg.norm(s.u.reshape(-1, 2), axis=1) for s in solutions]
    tips = [float(d.max()) for d in disp]
    clim = (0.0, max(tips))

    # Two rows: the bare meshes with the selected regions on top (where the conditions go),
    # the deformed solves below (what they drive). Sized so the 4:1 panels are not slivers.
    resolved = Plotter(2, 2, title='One specification, two meshes, the same solve',
                       axis_labels=False, figsize=(11.0, 5.6))
    for col, ((name, m), s, d, tip) in enumerate(zip(meshes, solutions, disp, tips)):
        resolved.plot(m, mode='mesh', idx=(0, col), title=f'{name}: {len(m.elements)} triangles')
        resolved.plot_highlights(
            m, [np.flatnonzero(clamped(m.vertices)), np.flatnonzero(loaded(m.vertices))],
            ['red', 'lime'], ['', ''], idx=(0, col))
        resolved.plot(s.deformed_mesh(), d, mode='colored', idx=(1, col),
                      label='displacement |u|', clim=clim, title=f'tip |u| = {tip:.3f}')
    legend_handles = [
        Line2D([], [], marker='o', linestyle='', color='red', markersize=6,
               label='on_plane: clamped edge'),
        Line2D([], [], marker='o', linestyle='', color='lime', markersize=6,
               label='on_plane: loaded edge'),
    ]
    resolved.fig.legend(handles=legend_handles, loc='outside lower center', ncol=2,
                        frameon=False)

    spread = 100 * abs(tips[1] - tips[0]) / tips[1]
    return DemoResult([
        Figure(resolved,
               'One cantilever, clamped on the left edge and pulled down along the right, '
               'solved on two unrelated triangulations of the same beam: a structured grid '
               f"({len(meshes[0][1].elements)} triangles) and an unstructured Ruppert's mesh "
               f"({len(meshes[1][1].elements)}), numbered nothing alike. Top: the clamp (red) "
               'and load (green), placed by position rather than by vertex index, land on the '
               'same physical edges on each mesh. Bottom: they drive the same solve, the tip '
               f'deflections agreeing to within {spread:.1f}%. This is what lets a condition '
               'be written once and survive whatever remeshing happens after, including '
               'adaptive refinement rebuilding the mesh repeatedly.'),
    ], text=(f'structured grid   {len(meshes[0][1].elements):>5} triangles, tip |u| = {tips[0]:.4f}\n'
             f"Ruppert's mesh    {len(meshes[1][1].elements):>5} triangles, tip |u| = {tips[1]:.4f}\n"
             f'difference        {spread:.1f}%'))

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
    # Builds its own two meshes (a structured grid and a Ruppert's mesh) so it takes no
    # domain. The smoke run shrinks both to a handful of triangles.
    Demo('regions', demo_regions, section='Meshing a domain',
         smoke_kwargs={'n_structured': 6, 'max_area_fraction': 0.05}),
]
