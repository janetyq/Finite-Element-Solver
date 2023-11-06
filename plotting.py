import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# PLOTTING
def plot_mesh(points, faces, title=None, ax=None, show=True):
    '''Plots flat 2d triangulated mesh'''
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.triplot(points[:,0], points[:,1], faces)
    ax.set_aspect('equal')
    # ax.plot(0, 0, 'o', color='red')
    if show:
        plt.show()

def plot_colored_mesh(points, faces, u, title=None, contour=False, colorscale=None, ax=None, show=True, cbar_label=None):
    '''Plots 2d triangulated mesh with colored faces  according to u'''
    if ax is None:
        fig, ax = plt.subplots()
    plot_triangulation = Triangulation(points[:, 0], points[:, 1], triangles=faces)
    ax.triplot(plot_triangulation, 'k-')  # Plot the mesh edges
    if colorscale:
        tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis', vmin=colorscale[0], vmax=colorscale[1])
    else:
        min_u, max_u = min(u), max(u)
        if abs(min_u-max_u) < 1e-2:
            vmin = min_u - 1e-2
            vmax = max_u + 1e-2
        else:
            vmin = min_u
            vmax = max_u
        tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
    if cbar_label is not None:
        cbar.set_label(cbar_label, rotation=90)

    if contour:
        contour_levels = np.linspace(min(u), max(u), 20)  # Adjust the number of levels as needed
        ax.tricontour(plot_triangulation, u, levels=contour_levels, colors='k', linestyles='solid')

    # x_min, x_max = min(points[:, 0]), max(points[:, 0])
    # y_min, y_max = min(points[:, 1]), max(points[:, 1])
    # x_mid, y_mid = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
    # x_range, y_range = x_max - x_min, y_max - y_min
    # avg_range = (x_range + y_range) * 0.5
    # if x_range > y_range:
    #     ax.set_xlim(x_mid - x_range * 0.6, x_mid + x_range * 0.6)
    #     ax.set_ylim(y_mid - avg_range * 0.5, y_mid + avg_range * 0.5)
    # elif x_range < y_range:
    #     ax.set_xlim(x_mid - avg_range * 0.5, x_mid + avg_range * 0.5)
    #     ax.set_ylim(y_mid - y_range * 0.6, y_mid + y_range * 0.6)

    ax.ticklabel_format(useOffset=False)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    
    return ax

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

    return ax