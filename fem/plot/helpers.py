"""Low-level matplotlib drawing helpers used by the Plotter class: mesh, boundary,
highlights, colored fields, surfaces, arrows, colorbars, and boundary conditions.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.tri import Triangulation


def plot_mesh(ax, mesh, color='black', linewidth=0.2):
    ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements, color=color, linewidth=linewidth)


def plot_boundary(ax, mesh, color='black', linewidth=1.0):
    for seg in mesh.boundary:
        ax.plot(mesh.vertices[seg, 0], mesh.vertices[seg, 1], color=color, linewidth=linewidth)


def plot_highlight(ax, mesh, idxs_list, color_list, label_list, mode='vertices'):
    for idxs, color, label in zip(idxs_list, color_list, label_list):
        if mode == 'vertices':
            ax.scatter(mesh.vertices[idxs, 0], mesh.vertices[idxs, 1], color=color, s=5, label=label)
        elif mode == 'elements':
            first = True  # Handle the label only for the first element
            for e_idx in idxs:
                vertices = mesh.vertices[mesh.elements[e_idx]]
                ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=0.2, label=label if first else None)
                first = False


def setup_colorbar(ax, vlim, label=None):
    cmap = matplotlib.colormaps['viridis']  # Choose a colormap
    norm = plt.Normalize(vmin=vlim[0], vmax=vlim[1])  # Normalize values between vmin and vmax

    # Create a scalar mappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data for colorbar

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label)
    return cmap, norm


def plot_colored(ax, mesh, values, cbar_info=None):
    if cbar_info is None:
        cbar_info = setup_colorbar(ax, (min(values), max(values)), None)

    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.elements)
    ax.tripcolor(triangulation, values, cmap=cbar_info[0], norm=cbar_info[1])

    # TODO: check contour
    # if contour > 0:
    #     ax.tricontour(triangulation, values, levels=np.linspace(min(values), max(values), contour), colors='k', linestyles='solid')
    return cbar_info


def change_ax_to_ax3d(ax, fig, ax_shape, ax_idx):
    if hasattr(ax, 'get_zlim'):
        return ax
    ax.remove()
    n = ax_shape[0]*100 + ax_shape[1]*10 + ax_idx[0]*ax_shape[1] + ax_idx[1] + 1
    ax = fig.add_subplot(n, projection='3d')
    return ax


def plot_surface(ax, mesh, values):
    if values.shape == (len(mesh.vertices),):
        pass
    elif values.shape == (len(mesh.elements),):
        values = mesh.convert_element_values_to_vertex_values(values)
    else:
        raise ValueError(f'Invalid values shape: {values.shape}')
    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.elements)
    ax.plot_trisurf(triangulation, values, cmap='viridis')


def plot_arrows(ax, mesh, values):
    # TODO: colored arrows, hard to see scale currently
    element_vertices = np.mean(mesh.vertices[mesh.elements], axis=1)
    ax.quiver(element_vertices[:, 0], element_vertices[:, 1], values[:, 0], values[:, 1], alpha=0.5, scale=10)


def plot_bc(ax, mesh, bc):
    from fem.boundary import BCType

    plot_mesh(ax, mesh)
    # entries() resolves regions against this mesh without needing a dim, which
    # is all plotting needs -- no DOF numbering involved.
    for bc_type, idxs, values in bc.entries(mesh):
        points = mesh.vertices[idxs]
        if bc_type is BCType.DIRICHLET:
            ax.plot(points[:, 0], points[:, 1], 'ro')
        elif bc_type is BCType.NEUMANN:
            ax.quiver(points[:, 0], points[:, 1], values[:, 0], values[:, 1])
