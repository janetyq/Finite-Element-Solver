"""Meshing demos: turning outlines into triangulations, and solving on them.

Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run outline_zoo
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

from fem.boundary import BoundaryConditions, BCType
from fem.equations import LinearElastic, Poisson
from fem.geometry import calculate_triangle_min_angle
from fem.plot.plotter import Plotter
from fem.plot.helpers import plot_mesh
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import PSLG, read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker
from fem.regions import everywhere, on_plane
from fem.solver import Solver

from demo_registry import Demo, DemoResult, Figure
from domains import beam, gear_pslg, star_pslg

# Resolved against the repo rather than the working directory: the input files ship
# with the project, so a demo should not depend on where it was launched from. Output
# paths stay relative, and so follow the caller's directory.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'california.svg')
CLOUD_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'cloud.svg')

# Douglas-Peucker: drop points that deviate less than this fraction of the curve's
# bounding-box extent. Ruppert's cost grows steeply in point count, so simplifying
# first is what keeps the demo interactive.
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.005

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

    `box` is `(x0, x1, y0, y1)` in data coordinates. The main panels are drawn fine
    enough that the field reads smooth, which can look like a resolution ceiling; the
    inset shows the actual mesh under one of them, so the density reads as a display
    choice rather than a limit. Drawn on white rather than over the field, so the lines
    stay legible where the near-boundary field is dark.
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

def demo_outline_zoo(min_angle=28, max_area_fraction=0.0008, interactive=False):
    """Mesh four closed outlines, traced and generated, and solve the same Poisson 'dome'
    on each, to show one pipeline (simplify, Ruppert refine, solve) turning any shape into
    a domain a PDE runs on."""
    # --interactive first opens a slider to explore the Douglas-Peucker simplification on
    # the California outline; the zoo itself simplifies the traced SVGs at a fixed tolerance.
    if interactive:
        simplify_curve(get_curve_from_svg(DEFAULT_SVG_FILE), interactive=True)

    shapes = _zoo_shapes()
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
        # no colorbar: the shape is the point, not the amplitude. A whisper of a wireframe
        # keeps the mesh present without turning the field into a net.
        clim = (0.0, float(u.max()))
        plotter.plot(mesh, u, mode='colored', idx=idx, colorbar=False, clim=clim,
                     empty=True, title=f'{name}: {len(mesh.elements)} triangles')
        plot_mesh(plotter.get_ax(idx), mesh, color='0.9', linewidth=0.1)
        if name == 'California':
            # Reveal the real mesh under the smooth field, zoomed into the central coast.
            v = np.asarray(mesh.vertices)
            lo, hi = v.min(axis=0), v.max(axis=0)
            span = hi - lo
            box = (lo[0] + 0.08 * span[0], lo[0] + 0.22 * span[0],
                   lo[1] + 0.47 * span[1], lo[1] + 0.65 * span[1])
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
               'angle. The mesh is drawn fine enough that the fields read smooth, not as '
               "a resolution ceiling: the inset zooms into California's mesh, and "
               'refinement (see the adaptive-refinement demo) drives it as fine as '
               'wanted.')],
        text='\n'.join(rows))


DEMOS = [
    # Builds its own outlines, so it takes no domain. The smoke run loosens the area cap
    # to a handful of triangles per shape, which still exercises every generator and the
    # traced-SVG simplify -> Ruppert -> solve path.
    Demo('outline_zoo', demo_outline_zoo, section='Meshing a domain',
         smoke_kwargs={'max_area_fraction': 0.04}),
    # Builds its own two meshes (a structured grid and a Ruppert's mesh) so it takes no
    # domain. The smoke run shrinks both to a handful of triangles.
    Demo('regions', demo_regions, section='Meshing a domain',
         smoke_kwargs={'n_structured': 6, 'max_area_fraction': 0.05}),
]
