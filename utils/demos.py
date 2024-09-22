import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

sys.path.append('..')
from Mesh import *
from Plotter import *

from utils.meshing import *
from utils.svg_pslg import *
from utils.helper import *

def demo_uniform_mesh(corners=[[0, 0], [1, 1]], resolution=(40, 40), save_file='../files/mesh.json'):
    mesh = create_rect_mesh(corners, resolution=resolution)
    mesh.save(save_file)
    return mesh

def demo_mesh_plotting(mesh):
    plotter = Plotter(title='Mesh plot (color=x)')
    plotter.plot(mesh, mode='colored', values=mesh.vertices[mesh.elements].mean(axis=1)[:, 0])
    plotter.show()

    min_x = np.min(mesh.vertices[:, 0])
    mid_x = np.mean(mesh.vertices[:, 0])
    
    # Mesh plotting examples with color
    e_idxs = [e_idx for e_idx, element in enumerate(mesh.elements) if np.mean(mesh.vertices[element], axis=0)[0] > mid_x]
    v_idxs = [v_idx for v_idx, vert in enumerate(mesh.vertices) if vert[0] < min_x + 1e-3]

    plotter = Plotter(title='Highlighted plot')
    plotter.plot(mesh, mode='mesh')
    plotter.plot_highlights(mesh, [e_idxs], ['blue'], ['right blue elements'], mode='elements')
    plotter.plot_highlights(mesh, [v_idxs], ['red'], ['left red vertices'], mode='vertices')
    plotter.show()

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

def demo_rupperts(curve, min_angle=20):
    pslg = PSLG(curve)
    pslg.add_bounding_box(buffer=0.2)
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle)
    mesh = rupperts.run_algo()

    plotter = Plotter(title='Triangulated mesh')
    plotter.plot(mesh, mode='mesh')
    ax = plotter.get_ax()
    for seg in rupperts.segments:
        ax.plot(rupperts.vertices[seg, 0], rupperts.vertices[seg, 1], 'b-')
    plotter.show()

    return mesh


if __name__ == '__main__':

    # # Uniform Mesh and Plotting
    # mesh = demo_uniform_mesh(resolution=(40, 40), save_file='../files/uniform_mesh.json')
    # demo_mesh_plotting(mesh)

    # SVG to Mesh
    curve = get_curve_from_svg('../files/california.svg')
    
    curve_reduced = demo_douglas_peucker(curve)
    print(f"Original points: {len(curve)}, Final points: {len(curve_reduced)}")
    
    mesh = demo_rupperts(curve_reduced, min_angle=20)