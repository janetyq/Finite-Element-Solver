"""Low-level matplotlib drawing helpers used by the Plotter class: mesh, boundary,
highlights, colored fields, surfaces, arrows, and colorbars. Boundary conditions are a
picture with a vocabulary of their own and live in `fem.plot.bc`.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.tri import Triangulation

if TYPE_CHECKING:
    from fem.space import FunctionSpace


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


def _tessellates_field(space):
    """Whether `space`'s element carries a within-element field to sub-sample (P2+)."""
    return space is not None and getattr(space.element_type, 'SHAPE_DEGREE', 1) > 1


def _curved_boundary(space, mesh):
    """Whether the boundary bends: a curved element over a mesh with curves."""
    return (space is not None
            and getattr(space.element_type, 'GEOMETRY_DEGREE', 1) > 1
            and mesh.boundary_curves is not None)


def plot_mesh(ax, mesh, color='black', linewidth=0.2,
              space: 'FunctionSpace | None' = None, subdivisions=3):
    """The mesh wireframe. With a curved `space`, interior edges stay straight while the
    boundary edges bow to follow their true curve."""
    if _curved_boundary(space, mesh):
        assert space is not None
        _plot_curved_wireframe(ax, mesh, space, subdivisions, color, linewidth)
    else:
        ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements,
                   color=color, linewidth=linewidth)


def _plot_curved_wireframe(ax, mesh, space: 'FunctionSpace', subdivisions, color, linewidth):
    from matplotlib.collections import LineCollection
    # Only boundary edges curve, so draw the straight interior edges directly and leave
    # the boundary to the curved polylines: a straight chord over the curve would double
    # the domain edge.
    boundary_keys = {tuple(sorted((int(f[0]), int(f[1])))) for f in mesh.boundary}
    interior = np.array(
        [e for e in mesh.edges if (int(e[0]), int(e[1])) not in boundary_keys])
    if len(interior):
        ax.add_collection(
            LineCollection(mesh.vertices[interior], colors=color, linewidths=linewidth))
    for line in space.boundary_polylines(subdivisions):
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=linewidth)


def plot_boundary(ax, mesh, color='black', linewidth=1.0,
                  space: 'FunctionSpace | None' = None, subdivisions=3):
    """The domain outline. On a mesh carrying analytic curves it follows them: through the
    curved element map when a `space` is given, else by sampling the curve directly (the
    most faithful outline available for a mesh-only figure)."""
    if mesh.boundary_curves is None:
        for seg in mesh.boundary:
            ax.plot(mesh.vertices[seg, 0], mesh.vertices[seg, 1], color=color, linewidth=linewidth)
        return
    if _curved_boundary(space, mesh):
        assert space is not None
        for line in space.boundary_polylines(subdivisions):
            ax.plot(line[:, 0], line[:, 1], color=color, linewidth=linewidth)
        return
    ts = np.linspace(0.0, 1.0, subdivisions + 1)[:, None]
    for facet, curve in zip(mesh.boundary, mesh.boundary_curves):
        a, b = mesh.vertices[facet[0]], mesh.vertices[facet[1]]
        if curve is None:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=linewidth)
        else:
            pts = np.asarray(curve.project(a + ts * (b - a)))
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth)


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


def plot_colored(ax, mesh, values, cbar_info=None, label=None, cmap_name='viridis', log_scale=False,
                 colorbar=True, contour=None, space: 'FunctionSpace | None' = None, subdivisions=3,
                 warp=None):
    if cbar_info is None:
        cbar_info = setup_colorbar(ax, (min(values), max(values)), label, cmap_name, log_scale, colorbar)

    # With a P2 space and a per-node field, draw on a fine tessellation of each element,
    # so a quadratic field shows its within-element curvature and a curved boundary its
    # true shape, rather than one flat triangle per element. Otherwise the P1 path: the
    # mesh's own triangles, coloured per-vertex or per-element.
    tess = None
    if _tessellates_field(space):
        assert space is not None
        if np.asarray(values).shape[0] == space.n_nodes:
            # `warp` (a nodal displacement) tessellates the deformed configuration, so a
            # P2 field draws on the warped shape rather than the reference one.
            deformed = None if warp is None else space.node_coords + np.asarray(warp)
            tess = space.tessellation(subdivisions, node_coords=deformed)
    if tess is not None:
        triangulation = Triangulation(tess.points[:, 0], tess.points[:, 1],
                                      triangles=tess.triangles)
        field = tess.interpolate(values)
    else:
        triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                                      triangles=mesh.elements)
        field = values
    # The collection is returned so an animation can recolour it in place across frames
    # rather than clearing the axes and rebuilding it; its array is per-face, which
    # `face_values` below matches for either a per-vertex or a per-element field.
    collection = ax.tripcolor(triangulation, field, cmap=cbar_info.cmap, norm=cbar_info.norm)

    if contour:
        # Isolines over the flat colouring: the level sets of the field (a potential's
        # equipotentials, say). tricontour needs a continuous per-vertex field, so an
        # element-constant one is projected to the vertices first. `levels=contour`
        # lets matplotlib choose that many "nice" values, which stays legible on a
        # skewed field where an even split would bunch them. On the tessellation the
        # field is already per-point, so it is used as is.
        nodal = np.asarray(field)
        if tess is None and nodal.shape == (len(mesh.elements),):
            from fem.space import FunctionSpace
            nodal = FunctionSpace(mesh).recover_nodal(nodal)
        ax.tricontour(triangulation, nodal, levels=contour, colors='black',
                      linewidths=0.5, alpha=0.5)
    return cbar_info, collection


def face_values(mesh, values):
    """The per-face array a flat-shaded 2D `tripcolor` carries.

    A per-element field is already one value per triangle; a per-vertex field is
    averaged over each triangle's corners, as `tripcolor` does internally for flat
    shading. This lets an animation update the collection's array
    frame to frame instead of rebuilding it.
    """
    values = np.asarray(values)
    if values.shape == (len(mesh.elements),):
        return values
    return values[np.asarray(mesh.elements)].mean(axis=1)


def solid_face_values(mesh, values):
    """The per-facet array the boundary-surface `Poly3DCollection` carries.

    A 3D solid is drawn as its boundary facets, coloured by the field averaged over
    each facet's vertices. An element-constant field is projected to the vertices
    first, the same volume-weighted way `plot_surface` does.
    """
    values = np.asarray(values)
    if values.shape == (len(mesh.elements),):
        from fem.space import FunctionSpace
        values = FunctionSpace(mesh).recover_nodal(values)
    return values[np.asarray(mesh.boundary)].mean(axis=1)


def change_ax_to_ax3d(ax, fig, ax_shape, ax_idx):
    if hasattr(ax, 'get_zlim'):
        return ax
    ax.remove()
    n = ax_shape[0]*100 + ax_shape[1]*10 + ax_idx[0]*ax_shape[1] + ax_idx[1] + 1
    ax = fig.add_subplot(n, projection='3d')
    return ax


def plot_surface(ax, mesh, values, clim=None, space: 'FunctionSpace | None' = None,
                 subdivisions=3, warp=None):
    """Lift `values` over a 2D mesh into the z direction.

    `clim` fixes both the colour mapping and the z axis, so a grid of surfaces can be
    compared: left to autoscale, each panel is drawn to its own height, and a wave
    losing amplitude looks exactly like one that is not.

    With a P2 (or curved) `space` and a per-node field, the surface is lifted over the
    element tessellation, so it shows the within-element curvature rather than one flat
    facet per element. `warp` tessellates the deformed configuration (see `plot_colored`).
    """
    if (space is not None and _tessellates_field(space)
            and np.asarray(values).shape[0] == space.n_nodes):
        deformed = None if warp is None else space.node_coords + np.asarray(warp)
        tess = space.tessellation(subdivisions, node_coords=deformed)
        triangulation = Triangulation(tess.points[:, 0], tess.points[:, 1],
                                      triangles=tess.triangles)
        values = tess.interpolate(values)
    else:
        if values.shape == (len(mesh.vertices),):
            pass
        elif values.shape == (len(mesh.elements),):
            # A surface plot interpolates between nodes, so an element-constant field
            # has to be projected first. The projection is volume-weighted and lives
            # on the space, which is cheap to build; nothing assembles until asked.
            from fem.space import FunctionSpace
            values = FunctionSpace(mesh).recover_nodal(values)
        else:
            raise ValueError(f'Invalid values shape: {values.shape}')
        triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                                      triangles=mesh.elements)
    vmin, vmax = clim if clim is not None else (None, None)
    ax.plot_trisurf(triangulation, values, cmap='viridis', vmin=vmin, vmax=vmax)
    if clim is not None:
        ax.set_zlim(*clim)


def plot_solid(ax, mesh, values, cbar_info=None):
    """Draw a 3D mesh as its boundary surface, coloured by `values`.

    Only the boundary facets are drawn: the interior of a solid is not visible, and
    a tet mesh has several times more elements than surface triangles.

    `values=None` draws the surface plain, for showing a mesh rather than a field
    on it; there is nothing for a colorbar to say in that case, so there is none.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    facets = np.asarray(mesh.boundary)
    if values is None or cbar_info is None:
        ax.add_collection3d(Poly3DCollection(
            mesh.vertices[facets], facecolor='#9fb8cd', edgecolor='black', linewidth=0.1))
        _fit_3d_limits(ax, mesh)
        return None

    # Returned so an animation can recolour the boundary surface in place: the facets
    # are fixed across frames, only the field on them changes.
    collection = Poly3DCollection(mesh.vertices[facets], cmap=cbar_info.cmap,
                                  norm=cbar_info.norm, edgecolor='black', linewidth=0.1)
    collection.set_array(solid_face_values(mesh, values))
    ax.add_collection3d(collection)
    _fit_3d_limits(ax, mesh)
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
    _, first = np.unique(cells, axis=0, return_index=True)
    return first


def plot_arrows(ax, mesh, values, max_arrows=MAX_ARROWS,
                space: 'FunctionSpace | None' = None, warp=None):
    # TODO: colored arrows, hard to see scale currently
    values = np.asarray(values)
    if space is not None and values.shape[0] == space.n_nodes:
        # A per-node vector field (a recovered flux) drawn at the nodes, optionally on the
        # deformed configuration, rather than one arrow per element at its centroid.
        positions = space.node_coords if warp is None else space.node_coords + np.asarray(warp)
    else:
        positions = np.mean(mesh.vertices[mesh.elements], axis=1)   # per-element, at centroids
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
