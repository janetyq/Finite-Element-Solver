"""Meshing demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run mesh_plotting
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objects as go

from fem.plot.plotter import Plotter
from fem.mesh.generation import create_rect_mesh, RuppertsAlgorithm
from fem.mesh.svg import read_svg_to_list_of_path_points, douglas_peucker, PSLG

from demo_registry import Demo

DEFAULT_SVG_FILE = 'files/california.svg'

def demo_uniform_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40), save_file='files/mesh.json'):
    """Build a uniform rectangular mesh and save it to disk."""
    mesh = create_rect_mesh(corners, resolution=resolution)
    mesh.save(save_file)
    return mesh

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
    return [plotter, highlight_plotter]

def get_curve_from_svg(svg_file):
    output = read_svg_to_list_of_path_points(svg_file)
    curve = max(output, key=lambda x: len(x)) # get the longest path
    return np.array(curve)

def demo_douglas_peucker(curve, save_file='douglas_peucker_output.json'):
    fig, ax = plt.subplots()
    ax.plot(curve[:, 0], curve[:, 1], color='gray', alpha=0.5)
    plt.subplots_adjust(bottom=0.15)

    sampled_plot = plt.plot(curve[:, 0], curve[:, 1], 'b-')[0]
    d = max(np.max(curve, axis=0) - np.min(curve, axis=0))
    slider = Slider(plt.axes([0.15, 0.04, 0.6, 0.03]), 'Epsilon ', 0, d/20, valinit=0.0)
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

    # The input PSLG's boundary segments, overlaid in blue -- there's no shared
    # "ax" to draw onto under Plotly, so build the segment trace directly and
    # add it via add_trace instead.
    xs, ys, zs = [], [], []
    for i, j in rupperts.segments:
        xs += [rupperts.vertices[i, 0], rupperts.vertices[j, 0], None]
        ys += [rupperts.vertices[i, 1], rupperts.vertices[j, 1], None]
        zs += [0, 0, None]
    boundary_trace = go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='blue', width=3),
                                   showlegend=False, hoverinfo='skip')
    plotter.add_trace(boundary_trace)
    return plotter

def demo_douglas_peucker_svg(svg_file=DEFAULT_SVG_FILE):
    """Interactively simplify an SVG outline via Douglas-Peucker (drag epsilon, click Save)."""
    curve = get_curve_from_svg(svg_file)
    return demo_douglas_peucker(curve)

def demo_rupperts_svg(svg_file=DEFAULT_SVG_FILE, min_angle=20):
    """Simplify an SVG outline (interactively) then triangulate it with Ruppert's algorithm."""
    curve = get_curve_from_svg(svg_file)
    curve_reduced = demo_douglas_peucker(curve)
    return demo_rupperts(curve_reduced, min_angle=min_angle)


DEMOS = [
    Demo('uniform_mesh', demo_uniform_mesh, needs_mesh=False, returns_plotter=False),
    Demo('mesh_plotting', demo_mesh_plotting),
    Demo('douglas_peucker', demo_douglas_peucker_svg, needs_mesh=False, returns_plotter=False),
    Demo('rupperts', demo_rupperts_svg, needs_mesh=False),
]
