"""Meshing demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run mesh_plotting
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider

from fem.plot.plotter import Plotter
from fem.mesh.ruppert import create_rect_mesh, RuppertsAlgorithm
from fem.mesh.svg import read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker, PSLG

from demo_registry import Demo, DemoResult, Figure

# Resolved against the repo rather than the working directory: the input files ship
# with the project, so a demo should not depend on where it was launched from. Output
# paths stay relative, and so follow the caller's directory.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'california.svg')

# Douglas-Peucker: drop points that deviate less than this fraction of the curve's
# bounding-box extent. Ruppert's cost grows steeply in point count, so simplifying
# first is what keeps the demo interactive.
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.005

# Ruppert's: any triangle whose area exceeds this fraction of the outline's total area
# gets its circumcenter inserted. The angle bound controls element *shape* but says
# nothing about size, so without this a large region comes back as a handful of
# enormous triangles.
DEFAULT_MAX_AREA_FRACTION = 0.005

def demo_uniform_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40), save_file='mesh.json'):
    """Build a uniform rectangular mesh, save it to disk, and plot what was written."""
    mesh = create_rect_mesh(corners, resolution=resolution)
    mesh.save(save_file)

    plotter = Plotter(title=f'Uniform mesh {resolution[0]}x{resolution[1]}')
    plotter.plot(mesh, mode='mesh')
    return DemoResult(
        [Figure(plotter, f'A structured triangulation at {resolution[0]}x{resolution[1]}, '
                         'the input the solver demos load.')],
        artifacts=[Path(save_file)],
    )

def demo_mesh_plotting(mesh):
    """Plot a mesh colored by element-centroid x, then highlight elements/vertices on one side."""
    plotter = Plotter(title='Mesh plot (color=x)')
    plotter.plot(mesh, mode='colored', values=mesh.vertices[mesh.elements].mean(axis=1)[:, 0])

    min_x = np.min(mesh.vertices[:, 0])
    mid_x = np.mean(mesh.vertices[:, 0])

    # Mesh plotting examples with color
    e_idxs = [e_idx for e_idx, element in enumerate(mesh.elements) if np.mean(mesh.vertices[element], axis=0)[0] > mid_x]
    v_idxs = [v_idx for v_idx, vert in enumerate(mesh.vertices) if vert[0] < min_x + 1e-3]

    highlight_plotter = Plotter(title='Highlighted plot')
    highlight_plotter.plot(mesh, mode='mesh')
    highlight_plotter.plot_highlights(mesh, [e_idxs], ['blue'], ['right blue elements'], mode='elements')
    highlight_plotter.plot_highlights(mesh, [v_idxs], ['red'], ['left red vertices'], mode='vertices')
    return DemoResult([
        Figure(plotter, 'Elements coloured by the x of their centroid.', 'colored'),
        Figure(highlight_plotter,
               'Selecting by position: elements right of the midline, vertices on the '
               'left edge.', 'highlights'),
    ])

def get_curve_from_svg(svg_file):
    output = read_svg_to_list_of_path_points(svg_file)
    curve = max(output, key=lambda x: len(x)) # get the longest path
    return np.array(curve)

def demo_douglas_peucker(curve, save_file='douglas_peucker_output.json',
                         tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE, interactive=False):
    """Simplify `curve` with Douglas-Peucker, returning the simplified curve.

    `tolerance` is a fraction of the curve's extent. `interactive=True` opens a slider
    to explore it instead, starting from `tolerance`, and returns whatever it was left
    on. Simplifying directly is the default so that every caller -- including ones with
    nobody watching -- gets the same result without having to ask for it.
    """
    d = max(np.max(curve, axis=0) - np.min(curve, axis=0))
    if not interactive:
        return douglas_peucker(curve, tolerance * d)

    fig, ax = plt.subplots()  # a widget figure, not a Plotter: this path is interactive
    ax.plot(curve[:, 0], curve[:, 1], color='gray', alpha=0.5)
    plt.subplots_adjust(bottom=0.15)

    initial = douglas_peucker(curve, tolerance * d)
    sampled_plot = plt.plot(initial[:, 0], initial[:, 1], 'b-')[0]
    # Starting at zero would leave an untouched slider handing the full outline
    # downstream, which Ruppert's does not finish triangulating.
    slider = Slider(plt.axes([0.15, 0.04, 0.6, 0.03]), 'Epsilon ', 0, d/20,
                    valinit=tolerance * d)
    button = plt.Button(plt.axes([0.85, 0.04, 0.1, 0.04]), 'Save')

    def update(val):
        epsilon = slider.val
        dp = douglas_peucker(curve, epsilon)
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

def rupperts_mesh(pslg, min_angle=20, max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Triangulate a PSLG with Ruppert's algorithm; returns (mesh, algorithm)."""
    pslg.validate()
    max_area = None
    if max_area_fraction is not None:
        max_area = max_area_fraction * pslg.area()
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle, max_area=max_area)
    return rupperts.refine(), rupperts

def demo_rupperts(pslg, min_angle=20, max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Triangulate a PSLG with Ruppert's algorithm and plot the result."""
    mesh, rupperts = rupperts_mesh(pslg, min_angle=min_angle,
                                   max_area_fraction=max_area_fraction)

    plotter = Plotter(title='Triangulated mesh')
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    # One collection rather than a plot call per segment: an outline that has been
    # refined runs to hundreds of them.
    ax.add_collection(LineCollection(rupperts.vertices[rupperts.segments],
                                     colors='blue', linewidths=1.0))
    outlines = len(np.unique(rupperts.segment_loops))
    return DemoResult([Figure(
        plotter,
        f"Ruppert's refinement of {outlines} outlines (blue) into {len(mesh.elements)} "
        f'triangles, every angle at least {min_angle} degrees. The mesh covers what the '
        f'outlines enclose and nothing else, and carries the {len(mesh.boundary)} boundary '
        'edges a solver needs to put conditions on.')])

def demo_plate_with_hole(min_angle=25, max_area_fraction=0.004):
    """Mesh a plate with a hole in it, colouring the boundary by which outline it came from.

    The shape a flow-around-an-obstacle problem needs: one loop inside another is
    a hole under the even-odd rule, and the two boundaries have to be separable
    for the obstacle and the outer wall to take different conditions."""
    plate = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    hole = np.column_stack([1.6 + 0.55*np.cos(angles), 1.5 + 0.55*np.sin(angles)])

    pslg = PSLG.from_loops([plate, hole])
    mesh, rupperts = rupperts_mesh(pslg, min_angle=min_angle,
                                   max_area_fraction=max_area_fraction)

    plotter = Plotter(title='Plate with a hole')
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    for loop_id, colour, label in ((0, 'blue', 'outer wall'), (1, 'red', 'obstacle')):
        facets = np.asarray(mesh.boundary)[rupperts.boundary_loops == loop_id]
        ax.add_collection(LineCollection(mesh.vertices[facets], colors=colour,
                                         linewidths=2.0))
        ax.plot([], [], color=colour, linewidth=2.0, label=f'{label} ({len(facets)} edges)')
    ax.legend(loc='upper right')
    return DemoResult([Figure(
        plotter,
        f'{len(mesh.elements)} triangles between the two outlines. The hole is absent from '
        'the mesh but present in its boundary, and every boundary edge knows which outline '
        'it came from -- which is what lets Dirichlet on the obstacle and Neumann on the '
        'wall be written separately.')])

def demo_douglas_peucker_svg(svg_file=DEFAULT_SVG_FILE, tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE,
                             interactive=False):
    """Simplify an SVG outline via Douglas-Peucker; --interactive opens a slider over the
    tolerance, with a button that saves the curve you settle on."""
    curve = get_curve_from_svg(svg_file)
    simplified = demo_douglas_peucker(curve, tolerance=tolerance, interactive=interactive)

    # The interactive path has already had its say on screen; this is the result either
    # way, and the only thing a saved gallery can show of it.
    plotter = Plotter(title='Douglas-Peucker simplification')
    ax = plotter.get_ax()
    ax.plot(curve[:, 0], curve[:, 1], color='gray', linewidth=1.0, label=f'original ({len(curve)} pts)')
    ax.plot(simplified[:, 0], simplified[:, 1], 'b-', label=f'simplified ({len(simplified)} pts)')
    return DemoResult([Figure(
        plotter,
        f'{len(curve)} outline points reduced to {len(simplified)}. Ruppert\'s cost is '
        'superlinear in what it is handed, so this is what makes triangulating it tractable.')])

def demo_rupperts_svg(svg_file=DEFAULT_SVG_FILE, tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE,
                      interactive=False, min_angle=20,
                      max_area_fraction=DEFAULT_MAX_AREA_FRACTION):
    """Triangulate every closed outline in an SVG with Ruppert's algorithm;
    --interactive opens a slider to preview simplification of the largest outline
    before meshing.  The tolerance used for meshing is always `tolerance`, applied
    per-loop by `read_svg_to_pslg`."""
    if interactive:
        curve = get_curve_from_svg(svg_file)
        demo_douglas_peucker(curve, tolerance=tolerance, interactive=True)
    pslg = read_svg_to_pslg(svg_file, tolerance=tolerance)
    return demo_rupperts(pslg, min_angle=min_angle,
                         max_area_fraction=max_area_fraction)


DEMOS = [
    Demo('uniform_mesh', demo_uniform_mesh, needs_mesh=False),
    Demo('mesh_plotting', demo_mesh_plotting),
    Demo('douglas_peucker', demo_douglas_peucker_svg, needs_mesh=False),
    # Both mesh to a size cap, which is what makes the figures worth looking at and
    # also most of their cost; the smoke run only needs the code paths. Loosen the cap
    # and nothing else -- simplifying the outline further is *not* reliably cheaper,
    # because it sharpens the corners refinement struggles with.
    Demo('rupperts', demo_rupperts_svg, needs_mesh=False,
         smoke_kwargs={'max_area_fraction': 0.05}),
    Demo('plate_with_hole', demo_plate_with_hole, needs_mesh=False,
         smoke_kwargs={'max_area_fraction': 0.05}),
]
