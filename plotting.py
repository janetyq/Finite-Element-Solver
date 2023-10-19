import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# PLOTTING
def plot_mesh(points, faces, title=None):
    '''Plots flat 2d triangulated mesh'''
    if title:
        plt.title(title)
    plt.triplot(points[:,0], points[:,1], faces)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_colored_mesh(points, faces, u, title=None, contour=False, colorscale=None, show=True):
    '''Plots 2d triangulated mesh with colored faces  according to u'''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_triangulation = Triangulation(points[:, 0], points[:, 1], triangles=faces)
    ax.triplot(plot_triangulation, 'k-')  # Plot the mesh edges
    if colorscale:
        tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis', vmin=colorscale[0], vmax=colorscale[1])
    else:
        tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis')
    cbar = fig.colorbar(tripcolor, ax=ax, pad=0.1)

    if contour:
        contour_levels = np.linspace(min(u), max(u), 20)  # Adjust the number of levels as needed
        ax.tricontour(plot_triangulation, u, levels=contour_levels, colors='k', linestyles='solid')

    ax.ticklabel_format(useOffset=False)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    if show:
        plt.show()

def plot_surface_mesh(points, faces, u, title=None, contour=False, show=True):
    '''Plots 2d surface mesh in 3d with colored faces according to u'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_triangulation = Triangulation(points[:, 0], points[:, 1], triangles=faces)
    ax.plot_trisurf(plot_triangulation, u, cmap='viridis')
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    if show:
        plt.show()