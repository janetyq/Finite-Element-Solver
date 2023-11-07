import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# TODO:
# round t values in title

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

def plot_colored_mesh(points, faces, u, title=None, contour=False, colorscale=None, ax=None, show=True, cbar_label=''):
    '''Plots 2d triangulated mesh with colored faces  according to u'''
    if ax is None:
        fig, ax = plt.subplots()
    plot_triangulation = Triangulation(points[:, 0], points[:, 1], triangles=faces)
    tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis')
    cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
    cbar.set_label('', rotation=90)
    if contour:
        contour_levels = np.linspace(min(u), max(u), 20)  # Adjust the number of levels as needed
        ax.tricontour(plot_triangulation, u, levels=contour_levels, colors='k', linestyles='solid')

    ax.ticklabel_format(useOffset=False)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    
    return ax

def plot_surface_mesh(points, faces, u, title=None, ax=None, show=True):
    '''Plots 2d surface mesh in 3d with colored faces according to u'''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        # set ax projection as 3d
        # TODO: implement this
        pass
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

def plot_surface_mesh_animation(points, faces, t_values, u_values, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax settings
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_aspect('equal')
    
    plot_triangulation = Triangulation(points[:, 0], points[:, 1], triangles=faces)

    # plot first frame, TODO: not aligned
    ax.set_title(f'{title} t = 0')
    surf = ax.plot_trisurf(plot_triangulation, u_values[0], cmap='viridis')

    def update(frame):
        ax.clear()  # Clear the previous frame
        ax.set_title(f'{title} t = {t_values[frame]}')
        surf = ax.plot_trisurf(plot_triangulation, u_values[frame], cmap='viridis')
        return surf

    ani = FuncAnimation(fig, update, frames=range(len(t_values)), blit=False, repeat=True, interval=400)
    plt.show()

def plot_colored_mesh_animation(points, faces, t_values, u_values, title='', contour=False, cbar_label=''):
    '''Plots 2d triangulated mesh with colored faces according to u'''
    fig, ax = plt.subplots()

    # ax settings
    ax.set_aspect('equal')
    ax.ticklabel_format(useOffset=False)

    # plot first fram
    ax.set_title(f'{title} t = 0')
    plot_triangulation = Triangulation(points[:, 0], points[:, 1], triangles=faces)
    tripcolor = ax.tripcolor(plot_triangulation, u_values[0], cmap='viridis')
    cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.6)
    cbar.set_label(cbar_label, rotation=270)

    def update(frame):
        ax.clear()
        ax.set_title(f'{title} t = {t_values[frame]}')
        u = u_values[frame]
        tripcolor = ax.tripcolor(plot_triangulation, u, cmap='viridis')
        cbar.mappable.set_clim(vmin=u.min(), vmax=u.max())
        if contour:
            contour_levels = np.linspace(min(u), max(u), 20)  # Adjust the number of levels as needed
            ax.tricontour(plot_triangulation, u, levels=contour_levels, colors='k', linestyles='solid')
        return ax

    ani = FuncAnimation(fig, update, frames=range(len(u_values)), blit=False, repeat=True)

    plt.show()