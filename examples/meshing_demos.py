"""Meshing demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run mesh_plotting
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from fem.plot.plotter import Plotter
from fem.mesh.ruppert import create_rect_mesh, RuppertsAlgorithm
from fem.mesh.svg import read_svg_to_list_of_path_points, douglas_peucker, PSLG

from demo_registry import Demo, DemoResult, Figure

# Resolved against the repo rather than the working directory: the input files ship
# with the project, so a demo should not depend on where it was launched from. Output
# paths stay relative, and so follow the caller's directory.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[1] / 'files' / 'california.svg')

# Simplification tolerance as a fraction of the curve's bounding-box extent, so one
# number suits any outline. Ruppert's cost still grows steeply in the point count it is
# handed -- ~1.4 s at this tolerance on the California outline, ~450 s on the raw
# 1700-point curve -- so simplifying first is what keeps the demo interactive, and is
# why the slider starts here rather than at zero.
DEFAULT_TOLERANCE = 0.005

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
                         tolerance=DEFAULT_TOLERANCE, interactive=False):
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

def rupperts_mesh(curve, min_angle=20):
    """Triangulate a closed curve via Ruppert's algorithm; returns (mesh, algorithm)."""
    pslg = PSLG(curve)
    pslg.add_bounding_box(buffer=0.2)
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle)
    return rupperts.run_algo(), rupperts

def demo_rupperts(curve, min_angle=20):
    """Triangulate a closed curve with Ruppert's algorithm and plot the result."""
    mesh, rupperts = rupperts_mesh(curve, min_angle=min_angle)

    plotter = Plotter(title='Triangulated mesh')
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    for seg in rupperts.segments:
        ax.plot(rupperts.vertices[seg, 0], rupperts.vertices[seg, 1], 'b-')
    return DemoResult([Figure(
        plotter,
        f"Ruppert's refinement of the outline (blue) into {len(mesh.elements)} triangles, "
        f'every angle at least {min_angle} degrees.')])

def demo_douglas_peucker_svg(svg_file=DEFAULT_SVG_FILE, tolerance=DEFAULT_TOLERANCE,
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

def demo_rupperts_svg(svg_file=DEFAULT_SVG_FILE, tolerance=DEFAULT_TOLERANCE,
                      interactive=False, min_angle=20):
    """Simplify an SVG outline then triangulate it with Ruppert's algorithm;
    --interactive lets you pick the simplification first."""
    curve = get_curve_from_svg(svg_file)
    curve_reduced = demo_douglas_peucker(curve, tolerance=tolerance, interactive=interactive)
    return demo_rupperts(curve_reduced, min_angle=min_angle)


DEMOS = [
    Demo('uniform_mesh', demo_uniform_mesh, needs_mesh=False),
    Demo('mesh_plotting', demo_mesh_plotting),
    Demo('douglas_peucker', demo_douglas_peucker_svg, needs_mesh=False),
    Demo('rupperts', demo_rupperts_svg, needs_mesh=False),
]
