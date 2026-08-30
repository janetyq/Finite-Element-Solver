"""Low-level matplotlib drawing helpers used by the Plotter class: mesh, boundary,
highlights, colored fields, surfaces, arrows, and colorbars. Each draws a `PanelView`
(`fem.plot.tessellation`), the triangulation and field a panel shows on the true
geometry, so nothing here reads a `FunctionSpace` or an element type. Boundary
conditions are a picture with a vocabulary of their own and live in `fem.plot.bc`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D

from fem.mesh.mesh import unique_rows
from fem.plot.tessellation import PanelView, panel_view

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh


@dataclass(frozen=True)
class ColorbarInfo:
    """One panel's colour mapping, and the bar drawn beside it.

    `bar` is kept, not just the mapping it was built from, because the bar owns an
    axes of its own, and that axes has to be resized to match the panel once the
    layout is settled (`Plotter._fit_colorbars`). It is `None` when the colouring is
    drawn without a bar (`colorbar=False`).
    """
    cmap: Colormap
    norm: Normalize
    bar: Colorbar | None


def _as_view(target: Mesh | PanelView) -> PanelView:
    """A bare mesh drawn directly gets the plain P1 view."""
    return target if isinstance(target, PanelView) else panel_view(target)


def _triangulation(view: PanelView) -> Triangulation:
    return Triangulation(view.points[:, 0], view.points[:, 1], triangles=view.triangles)


def plot_mesh(ax, target: Mesh | PanelView, color='black', linewidth=0.2):
    """The mesh wireframe. On a curved view, interior edges stay straight while the
    boundary edges bow to follow their true curve."""
    view = _as_view(target)
    mesh = view.mesh
    if not view.curved:
        ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements,
                   color=color, linewidth=linewidth)
        return
    from matplotlib.collections import LineCollection
    # Only boundary edges curve, so draw the straight interior edges directly and leave
    # the boundary to the curved polylines: a straight chord over the curve would double
    # the domain edge.
    boundary_keys = {tuple(sorted((int(f[0]), int(f[1])))) for f in mesh.boundary}
    interior = np.array(
        [e for e in mesh.edges if (int(e[0]), int(e[1])) not in boundary_keys])
    if len(interior):
        ax.add_collection(
            LineCollection(list(mesh.vertices[interior]), colors=color, linewidths=linewidth))
    plot_boundary(ax, view, color=color, linewidth=linewidth)


def plot_boundary(ax, target: Mesh | PanelView, color='black', linewidth=1.0):
    """The domain outline, as the view's polylines: along the true curve where the
    geometry has one, else the facet chords."""
    view = _as_view(target)
    if view.boundary is None:
        return
    for line in view.boundary:
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=linewidth)


def setup_colorbar(ax, vlim, label=None, cmap_name='viridis', log_scale=False, colorbar=True):
    cmap = matplotlib.colormaps[cmap_name]
    if log_scale:
        vmin = max(vlim[0], 1e-10)  # floor to avoid log(0)
        norm = LogNorm(vmin=vmin, vmax=vlim[1])
    else:
        norm = Normalize(vmin=vlim[0], vmax=vlim[1])

    # `colorbar=False` keeps the mapping but draws no bar, for a colouring whose scale is
    # arbitrary or qualitative (a mode shape, where only the pattern is physical).
    if not colorbar:
        return ColorbarInfo(cmap, norm, None)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy data for colorbar
    cbar = plt.colorbar(sm, ax=ax)
    if label is not None:
        cbar.set_label(label)
    return ColorbarInfo(cmap, norm, cbar)


def plot_colored(ax, view: PanelView, cbar_info=None, label=None, cmap_name='viridis',
                 log_scale=False, colorbar=True, contour=None):
    """Colour the view's triangulation by its field, per point or per face.

    Returns the collection so an animation can recolour it in place across frames
    rather than clearing the axes and rebuilding it; its array is per-face, which
    `PanelView.face_values` matches for either a per-point or a per-element field.
    """
    values = view._require_values()
    if cbar_info is None:
        cbar_info = setup_colorbar(ax, (values.min(), values.max()), label, cmap_name,
                                   log_scale, colorbar)
    collection = ax.tripcolor(_triangulation(view), values, cmap=cbar_info.cmap,
                              norm=cbar_info.norm)
    if contour:
        # Isolines over the flat colouring: the level sets of the field (a potential's
        # equipotentials, say). tricontour needs a continuous per-point field, so an
        # element-constant one is projected to the points first. `levels=contour` lets
        # matplotlib choose that many "nice" values, which stays legible on a skewed
        # field where an even split would bunch them.
        ax.tricontour(_triangulation(view), view.point_values, levels=contour,
                      colors='black', linewidths=0.5, alpha=0.5)
    return cbar_info, collection


def change_ax_to_ax3d(ax, fig, ax_shape, ax_idx):
    if isinstance(ax, Axes3D):
        return ax
    ax.remove()
    n = ax_shape[0]*100 + ax_shape[1]*10 + ax_idx[0]*ax_shape[1] + ax_idx[1] + 1
    ax = fig.add_subplot(n, projection='3d')
    return ax


def plot_surface(ax, view: PanelView, clim=None):
    """Lift the view's field over its 2D triangulation into the z direction.

    `clim` fixes both the colour mapping and the z axis, so a grid of surfaces can be
    compared: left to autoscale, each panel is drawn to its own height, and a wave
    losing amplitude looks exactly like one that is not.

    A surface interpolates between points, so a per-element field is projected to the
    points first (`PanelView.point_values`); on a P2 or curved view the surface is lifted
    over the element tessellation, so it shows the within-element curvature.
    """
    vmin, vmax = clim if clim is not None else (None, None)
    ax.plot_trisurf(_triangulation(view), view.point_values, cmap='viridis',
                    vmin=vmin, vmax=vmax)
    if clim is not None:
        ax.set_zlim(*clim)


def plot_solid(ax, view: PanelView, cbar_info=None):
    """Draw a 3D mesh as its boundary surface, coloured by the view's field.

    Only the boundary facets are drawn: the interior of a solid is not visible, and
    a tet mesh has several times more elements than surface triangles.

    A view with no field draws the surface plain, for showing a mesh rather than a
    field on it; there is nothing for a colorbar to say in that case, so there is none.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    facets = view.points[view.triangles]
    if view.values is None or cbar_info is None:
        ax.add_collection3d(Poly3DCollection(
            facets, facecolor='#9fb8cd', edgecolor='black', linewidth=0.1))
        _fit_3d_limits(ax, view.mesh)
        return None

    # Returned so an animation can recolour the boundary surface in place: the facets
    # are fixed across frames, only the field on them changes.
    collection = Poly3DCollection(facets, cmap=cbar_info.cmap, norm=cbar_info.norm,
                                  edgecolor='black', linewidth=0.1)
    collection.set_array(view.face_values)
    ax.add_collection3d(collection)
    _fit_3d_limits(ax, view.mesh)
    return collection


def _fit_3d_limits(ax, mesh):
    """Frame a 3D mesh: `add_collection3d` does not autoscale, so the limits come from
    the mesh. Ticks are thinned too: a thin direction seen in projection puts six
    labels in the space of two."""
    lower, upper = mesh.bounds
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.locator_params(nbins=4)


# Arrows a quiver panel draws, at most. A vector field is read from the pattern the
# arrows make, and past this many they overlap into a grey mat that hides it, so this
# is a property of the picture, not of the mesh, and holds as the mesh is refined.
MAX_ARROWS = 700


def _spread_sample(points, target):
    """Indices of up to `target` points, spread evenly over the area they cover.

    Bins on a regular grid and takes one point per occupied bin, rather than every
    n-th point: element numbering follows the meshing order, so a stride through it
    samples in bands on a structured mesh and clumps on a generated one.
    """
    if len(points) <= target:
        return np.arange(len(points))

    lower, upper = points.min(axis=0), points.max(axis=0)
    span = np.where(upper > lower, upper - lower, 1.0)
    # A grid of about `target` cells, shaped like the domain.
    side = max(1, int(np.sqrt(target * span[0] / span[1])))
    cells = np.floor((points - lower) / span * [side, max(1, target // side)]).astype(int)
    _, first = unique_rows(cells, return_index=True)
    return first


def plot_arrows(ax, view: PanelView, values, max_arrows=MAX_ARROWS):
    """A vector field as arrows: at the view's nodes for a per-node field (a recovered
    flux, on the deformed configuration if the view is warped), else one arrow per
    element at its centroid."""
    # TODO: colored arrows, hard to see scale currently
    values = np.asarray(values)
    positions = view.nodes if values.shape[0] == len(view.nodes) else view.mesh.centroids
    keep = _spread_sample(positions, max_arrows)
    ax.quiver(positions[keep, 0], positions[keep, 1],
              values[keep, 0], values[keep, 1], alpha=0.5, scale=10)


def plot_refinement(ax, mesh, classifications, linewidth=0.5):
    """Draw a refined mesh with red/green triangle fills and a wireframe overlay."""
    for e_idx, kind in enumerate(classifications):
        verts = mesh.vertices[mesh.elements[e_idx]]
        color = '#e06666' if kind == 'red' else '#93c47d'
        ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=0.45)
    plot_mesh(ax, mesh, color='black', linewidth=linewidth)
