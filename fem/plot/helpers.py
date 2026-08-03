"""Low-level matplotlib drawing helpers used by the Plotter class: mesh, boundary,
highlights, colored fields, surfaces, arrows, colorbars, and boundary conditions.
"""
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.tri import Triangulation


@dataclass(frozen=True)
class ColorbarInfo:
    """One panel's colour mapping, and the bar drawn beside it.

    `bar` is kept, not just the mapping it was built from, because the bar owns an
    axes of its own -- and that axes has to be resized to match the panel once the
    layout is settled (`Plotter._fit_colorbars`).
    """
    cmap: Colormap
    norm: Normalize
    bar: Colorbar


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
    norm = Normalize(vmin=vlim[0], vmax=vlim[1])  # Normalize values between vmin and vmax

    # Create a scalar mappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data for colorbar

    cbar = plt.colorbar(sm, ax=ax)
    if label is not None:
        cbar.set_label(label)
    return ColorbarInfo(cmap, norm, cbar)


def plot_colored(ax, mesh, values, cbar_info=None, label=None):
    if cbar_info is None:
        cbar_info = setup_colorbar(ax, (min(values), max(values)), label)

    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.elements)
    ax.tripcolor(triangulation, values, cmap=cbar_info.cmap, norm=cbar_info.norm)
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
        # A surface plot interpolates between nodes, so an element-constant field
        # has to be projected first. The projection is volume-weighted and lives
        # on the space, which is cheap to build -- nothing assembles until asked.
        from fem.space import FunctionSpace
        values = FunctionSpace(mesh).element_to_vertex(values)
    else:
        raise ValueError(f'Invalid values shape: {values.shape}')
    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], triangles=mesh.elements)
    ax.plot_trisurf(triangulation, values, cmap='viridis')


def plot_solid(ax, mesh, values, cbar_info=None):
    """Draw a 3D mesh as its boundary surface, coloured by `values`.

    Only the boundary facets are drawn -- the interior of a solid is not visible, and
    a tet mesh has several times more elements than surface triangles.

    `values=None` draws the surface plain, for showing a mesh rather than a field
    on it; there is nothing for a colorbar to say in that case, so there is none.

    This is the 3D path that needs no optional dependency: `fem.plot.tet` renders
    through PyVista, which is the better viewer but pulls in VTK, so the deployed
    gallery has never had a 3D solve on it.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    facets = np.asarray(mesh.boundary)
    if values is None or cbar_info is None:
        ax.add_collection3d(Poly3DCollection(
            mesh.vertices[facets], facecolor='#9fb8cd', edgecolor='black', linewidth=0.1))
        _fit_3d_limits(ax, mesh)
        return

    if values.shape == (len(mesh.elements),):
        # An element-constant field has no value at a shared node; project it, the
        # same volume-weighted way `plot_surface` does.
        from fem.space import FunctionSpace
        values = FunctionSpace(mesh).element_to_vertex(values)

    collection = Poly3DCollection(mesh.vertices[facets], cmap=cbar_info.cmap,
                                  norm=cbar_info.norm, edgecolor='black', linewidth=0.1)
    collection.set_array(values[facets].mean(axis=1))
    ax.add_collection3d(collection)
    _fit_3d_limits(ax, mesh)


def _fit_3d_limits(ax, mesh):
    """Frame a 3D mesh: `add_collection3d` does not autoscale, so the limits come from
    the mesh. Ticks are thinned too -- a thin direction seen in projection puts six
    labels in the space of two."""
    lower, upper = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.locator_params(nbins=4)


def plot_arrows(ax, mesh, values):
    # TODO: colored arrows, hard to see scale currently
    element_vertices = np.mean(mesh.vertices[mesh.elements], axis=1)
    ax.quiver(element_vertices[:, 0], element_vertices[:, 1], values[:, 0], values[:, 1], alpha=0.5, scale=10)


def plot_refinement(ax, mesh, classifications, linewidth=0.5):
    """Draw a refined mesh with red/green triangle fills and a wireframe overlay."""
    for e_idx, kind in enumerate(classifications):
        verts = mesh.vertices[mesh.elements[e_idx]]
        color = '#e06666' if kind == 'red' else '#93c47d'
        ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.45)
    plot_mesh(ax, mesh, color='black', linewidth=linewidth)


def plot_bc(ax, mesh, bc):
    from fem.boundary import BCType

    plot_mesh(ax, mesh)
    # entries() resolves regions against this mesh without needing a component count, which
    # is all plotting needs -- no DOF numbering involved.
    for bc_type, idxs, values in bc.entries(mesh):
        points = mesh.vertices[idxs]
        if bc_type is BCType.DIRICHLET:
            ax.plot(points[:, 0], points[:, 1], 'ro')
        elif bc_type is BCType.NEUMANN:
            ax.quiver(points[:, 0], points[:, 1], values[:, 0], values[:, 1])
