"""Display sampling: a space tessellated into flat triangles on its true geometry, and
the `PanelView` a panel draws.

matplotlib draws straight-sided triangles with one value per vertex or per face. A P1
mesh already is that. A P2 or curved space is not: its field varies within an element
and its boundary nodes sit on a curve, so each element is sampled into `subdivisions**2`
sub-triangles through its own shape functions (`tessellate`). A display tessellation
only: it adds no error to the solve, it just controls how faithfully the computed
geometry and field are drawn.

`panel_view` packages whichever case applies into one `PanelView`, so the drawing helpers
in `fem.plot.helpers` see points, triangles, values, and boundary polylines, and never a
`FunctionSpace`, an element type, or a `Solution`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fem.field import NodalField
from fem.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh
    from fem.post.solution import Solution
    from fem.space import FunctionSpace

__all__ = ['PanelView', 'PlotTessellation', 'boundary_polylines', 'panel_view', 'tessellate']


def _reference_subtriangulation(subdivisions: int) -> tuple[FloatArray, IntArray]:
    '''Uniform split of the reference triangle into `subdivisions**2` sub-triangles.

    Returns `(points, triangles)`: the barycentric lattice `{(i/k, j/k): i + j <= k}`
    in reference `(xi, eta)` coordinates, and a structured triangulation of it. Sampling
    a P2 element's shape functions at this lattice traces the element's true (curved)
    image; the triangles tessellate it into the flat pieces matplotlib can draw.
    '''
    if subdivisions < 1:
        raise ValueError(f'subdivisions must be at least 1, got {subdivisions}')
    k = subdivisions
    index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    for j in range(k + 1):
        for i in range(k - j + 1):
            index[(i, j)] = len(points)
            points.append((i / k, j / k))
    triangles: list[list[int]] = []
    for j in range(k):
        for i in range(k - j):
            triangles.append([index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]])
            if i < k - j - 1:   # the downward triangle filling the gap above
                triangles.append(
                    [index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]])
    return np.array(points, dtype=float), np.array(triangles, dtype=int)


@dataclass(frozen=True)
class PlotTessellation:
    '''A curved element space sampled into flat sub-triangles for display.

    `points`/`triangles` are a fine straight-sided triangulation whose vertices sit on
    the true element geometry, so matplotlib draws a curved boundary as a chord chain
    fine enough to read as smooth. `interpolate` samples a per-node field at the same
    points, so a P2 field shows its within-element curvature instead of being flattened
    to one triangle per element.
    '''
    points: FloatArray            # (n_el * n_sub, spatial)
    triangles: IntArray           # (n_el * n_ref_tris, 3) into `points`
    _sample: FloatArray           # (n_sub, N) shape functions at the reference sub-points
    _element_nodes: IntArray      # (n_el, N) global node index per element

    def interpolate(self, nodal: FloatArray) -> FloatArray:
        '''Sample a per-node field at the tessellation points, aligned with `points`.

        `nodal` is one value (or component vector) per space node; the result is the
        field evaluated at every sub-point through the same shape functions the geometry
        used, so a quadratic field is drawn quadratically within each element.
        '''
        vals = np.asarray(nodal)[self._element_nodes]              # (n_el, N[, comp])
        sampled = np.einsum('sn,en...->es...', self._sample, vals)  # (n_el, n_sub[, comp])
        return sampled.reshape(-1, *sampled.shape[2:])


def tessellate(space: FunctionSpace, subdivisions: int = 3,
               node_coords: FloatArray | None = None) -> PlotTessellation:
    '''Sample every element of `space` into `subdivisions**2` flat sub-triangles on its
    true geometry, for a plot that follows a curved boundary and a P2 field.

    Each element's shape functions are evaluated at a reference sub-lattice and mapped
    through the element's own nodes, the same geometry map `Element.geometry` integrates
    over, so boundary sub-points land on the true curve and interior ones stay flat.

    `node_coords` overrides the node positions the sub-points are mapped through,
    `(n_nodes, spatial)` like `space.node_coords`. Passing the deformed positions (node
    coords plus a nodal displacement) tessellates the deformed configuration, so a P2
    field can be drawn on the warped shape rather than only the reference one.
    '''
    coords_of = space.node_coords if node_coords is None else np.asarray(node_coords)
    ref_points, ref_triangles = _reference_subtriangulation(subdivisions)
    sample = space.element_type.shape_values(ref_points)       # (n_sub, N)
    element_nodes = np.asarray(space.element_nodes)
    coords = coords_of[element_nodes]                          # (n_el, N, spatial)
    points = np.einsum('sn,end->esd', sample, coords)          # (n_el, n_sub, spatial)
    n_el, n_sub = points.shape[0], points.shape[1]
    points = points.reshape(n_el * n_sub, -1)
    offsets = (np.arange(n_el) * n_sub)[:, None, None]
    triangles = (ref_triangles[None] + offsets).reshape(-1, 3)
    return PlotTessellation(points, triangles, sample, element_nodes)


def boundary_polylines(space: FunctionSpace, subdivisions: int = 3,
                       node_coords: FloatArray | None = None) -> FloatArray:
    '''`(n_facets, subdivisions + 1, spatial)` boundary facets on their true curve.

    Each boundary facet sampled through the boundary element's geometry map, so a curved
    facet draws as a smooth polyline; a straight facet comes back as a straight sampled
    line. `node_coords` overrides the node positions, as in `tessellate`.
    '''
    coords_of = space.node_coords if node_coords is None else np.asarray(node_coords)
    xi = np.linspace(0.0, 1.0, subdivisions + 1)[:, None]     # (k+1, 1)
    sample = space.boundary_type.shape_values(xi)             # (k+1, boundary_N)
    facet_coords = coords_of[space.boundary_nodes]            # (n_facets, boundary_N, spatial)
    return np.einsum('sn,end->esd', sample, facet_coords)


def _mesh_boundary_polylines(mesh: Mesh, subdivisions: int,
                             vertices: FloatArray | None = None) -> FloatArray:
    '''The outline of a 2D mesh: each facet sampled along its analytic curve when the
    mesh carries one (the most faithful outline a mesh-only figure has), else its chord.
    `vertices` overrides the vertex positions (a warped outline), in which case the
    reference curves no longer apply and the chords are used.'''
    ts = np.linspace(0.0, 1.0, subdivisions + 1)[:, None]
    facets = np.asarray(mesh.boundary)
    coords = mesh.vertices if vertices is None else vertices
    a, b = coords[facets[:, 0]], coords[facets[:, 1]]
    lines = a[:, None, :] + ts[None] * (b - a)[:, None, :]      # (n_facets, k+1, dim)
    if vertices is None and mesh.boundary_curves is not None:
        for i, curve in enumerate(mesh.boundary_curves):
            if curve is not None:
                lines[i] = np.asarray(curve.project(lines[i]))
    return lines


@dataclass(frozen=True)
class PanelView:
    '''What one panel draws, on the true geometry.

    `points`/`triangles` are the straight-sided triangulation matplotlib gets: the mesh
    itself on the P1 path, the element tessellation for a P2 or curved space, and the
    boundary facets of a 3D solid. `values` is the field as given (per point or per
    triangle); `point_values` and `face_values` convert either way for the artists that
    need one or the other. `nodes` is where a per-node quantity (an arrow) sits, and
    `boundary` the outline as polylines, curved where the geometry is. `mesh` is kept for
    the wireframe and the 3D frame.
    '''
    mesh: Mesh
    nodes: FloatArray                       # (n_nodes, dim), warped if asked
    points: FloatArray                      # (n_points, dim)
    triangles: IntArray                     # (n_tri, 3) into points
    values: FloatArray | None
    boundary: FloatArray | None             # (n_facets, k+1, dim); None for a 3D solid
    curved: bool                            # the boundary bends: draw it as polylines
    _to_points: Callable[[FloatArray], FloatArray] = field(repr=False)

    @property
    def is_3d(self) -> bool:
        return self.points.shape[1] == 3

    def with_values(self, values: FloatArray | None) -> PanelView:
        '''The same geometry carrying another field (an animation's next frame).'''
        values = None if values is None else np.asarray(values)
        return PanelView(self.mesh, self.nodes, self.points, self.triangles, values,
                         self.boundary, self.curved, self._to_points)

    @property
    def per_face(self) -> bool:
        '''Whether `values` is one value per triangle rather than per point.'''
        values = self._require_values()
        n_faces = len(self.mesh.elements)
        return len(values) == n_faces and len(values) != len(self.points)

    @property
    def point_values(self) -> FloatArray:
        '''The field at every point of the triangulation, for a surface, a contour, or a
        shaded facet: a per-element field is volume-averaged onto the nodes first.'''
        values = self._require_values()
        return self._to_points(values) if self.per_face else values

    @property
    def face_values(self) -> FloatArray:
        '''One value per drawn triangle, the array a flat-shaded collection carries: a
        per-point field is averaged over each triangle's corners, as `tripcolor` does.'''
        values = self._require_values()
        if self.per_face and len(values) == len(self.triangles):
            return values
        # A per-point field, or a per-element one on a 3D solid whose drawn triangles
        # are its boundary facets: average the point values over each triangle.
        return self.point_values[self.triangles].mean(axis=1)

    def _require_values(self) -> FloatArray:
        if self.values is None:
            raise ValueError('this panel needs a field to draw; none was given')
        values = self.values
        if len(values) not in (len(self.points), len(self.mesh.elements)):
            raise ValueError(
                f'Invalid values shape: {values.shape}; expected one value per point '
                f'({len(self.points)}) or per element ({len(self.mesh.elements)})'
            )
        return values


def panel_view(
    target: Mesh | Solution | NodalField,
    values: FloatArray | Sequence[float] | NodalField | None = None,
    *,
    space: FunctionSpace | None = None,
    warp: FloatArray | bool | None = None,
    subdivisions: int = 3,
) -> PanelView:
    '''Build the `PanelView` a panel draws for `values` on `target`.

    `target` is a `Mesh`, a `Solution`, or a `NodalField`. A field or solution supplies
    its mesh and the space that numbers it, so a P2 or curved solve renders faithfully
    with nothing else passed (an explicit `space` still wins); `values` given as a
    `NodalField` draws its nodal values on its own space. Three cases:

    - A bare mesh, or a P1 space: the mesh's own triangles, the field per vertex or per
      element.
    - A P2 or curved space with a per-node field: each element is tessellated into
      `subdivisions**2` sub-triangles on its true geometry and the field is interpolated
      onto them. A curved space also traces the boundary through the element map even
      when only the mesh is drawn.
    - `warp`: a nodal displacement `(n_nodes, spatial)` that moves the nodes before
      tessellating, so a field draws on the deformed shape; `True` deforms by the
      solution's own displacement field and needs a `Solution` as the target.
    '''
    from fem.post.solution import Solution
    from fem.space import FunctionSpace

    if isinstance(values, NodalField):
        space = space if space is not None else values.space
        values = values.nodal_values
    if isinstance(target, (Solution, NodalField)):
        mesh = target.mesh
        space = space if space is not None else target.space
        if warp is True:
            if not isinstance(target, NodalField):
                raise ValueError('warp=True needs a field carrying a displacement')
            warp = target.dofs.reshape(-1, target.n_components)
    else:
        mesh = target
        if warp is True:
            raise ValueError('warp=True needs a field as the first argument, not a mesh')
    warp = None if warp is False else warp
    values = None if values is None else np.asarray(values)

    curved = (space is not None and space.element_type.GEOMETRY_DEGREE > 1
              and mesh.boundary_curves is not None)
    per_node = (space is not None and values is not None
                and values.shape[0] == space.n_nodes)
    tessellates = space is not None and space.element_type.SHAPE_DEGREE > 1 and per_node

    if space is not None and (tessellates or curved):
        nodes = space.node_coords if warp is None else space.node_coords + np.asarray(warp)
    else:
        nodes = mesh.vertices if warp is None else mesh.vertices + np.asarray(warp)

    boundary = None
    if tessellates:
        assert space is not None and values is not None
        tess = tessellate(space, subdivisions, node_coords=nodes)
        points, triangles = tess.points, tess.triangles
        values = tess.interpolate(values)
    else:
        points = nodes
        triangles = np.asarray(mesh.boundary if mesh.spatial_dim == 3 else mesh.elements)
    if mesh.spatial_dim == 2:
        if curved and space is not None:
            # Through the (possibly warped) element map, so the outline bends with it.
            boundary = boundary_polylines(space, subdivisions, node_coords=nodes)
        elif warp is None:
            boundary = _mesh_boundary_polylines(mesh, subdivisions)
        else:
            # P2 nodes are the vertices followed by the edge midpoints, so the leading
            # rows of the warped nodes are the warped vertices.
            boundary = _mesh_boundary_polylines(mesh, subdivisions,
                                                vertices=nodes[:mesh.n_vertices])

    def to_points(per_element: FloatArray) -> FloatArray:
        # Recovery onto the vertices of the drawn triangulation: the mesh's P1 space.
        from fem.post.recovery import recover_nodal
        return recover_nodal(FunctionSpace(mesh), per_element)

    return PanelView(mesh, nodes, points, triangles, values, boundary, curved, to_points)
