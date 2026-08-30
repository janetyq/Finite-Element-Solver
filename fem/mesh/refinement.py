"""Red-green triangle refinement.

`RedGreenRefiner` owns a parent/child tree recording how each triangle was produced,
so that when a green child is later marked for refinement its parent is recovered
and re-refined red, preserving mesh quality across rounds. Callers see only
``refine(idxs) -> Mesh``.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum, auto
from typing import TypeVar

import numpy as np

from fem.mesh.curves import Curve
from fem.mesh.mesh import Edge, Mesh
from fem.typing import Vertices

logger = logging.getLogger(__name__)

class _Status(Enum):
    RED_CHILD = auto()
    RED_PARENT = auto()
    GREEN_CHILD = auto()
    GREEN_PARENT = auto()
    GONE = auto()


class _Triangle:
    """A node in the red-green refinement tree."""

    __slots__ = ('vertex_idxs', 'status', 'parent', 'children', 'idx')

    def __init__(
        self,
        vertex_idxs: list[int],
        parent: _Triangle | None = None,
        status: _Status = _Status.RED_CHILD,
    ) -> None:
        self.vertex_idxs = vertex_idxs
        self.status = status
        self.parent = parent
        self.children: list[_Triangle] = []
        self.idx: int = -1

    def __repr__(self) -> str:
        return (
            f'_Triangle(verts={self.vertex_idxs}, status={self.status.name}, '
            f'parent={self.parent is not None}, children={len(self.children)})'
        )


T = TypeVar('T')


def _carry_onto_halves(table: dict[Edge, T], edge: list[int], mid_idx: int) -> None:
    """Whatever `table` holds for `edge`, held for both halves it was split into."""
    value = table.get(_edge_key(edge[0], edge[1]))
    if value is not None:
        table[_edge_key(edge[0], mid_idx)] = value
        table[_edge_key(mid_idx, edge[1])] = value


def _edge_key(v0: int, v1: int) -> Edge:
    return (v0, v1) if v0 < v1 else (v1, v0)


def _tri_edges(tri: _Triangle) -> list[Edge]:
    v = tri.vertex_idxs
    return [_edge_key(v[0], v[1]), _edge_key(v[1], v[2]), _edge_key(v[0], v[2])]


class RedGreenRefiner:
    """Persistent red-green refinement session over a triangle mesh.

    Wraps a mesh and maintains an internal hierarchy so that successive calls
    to `refine` can roll back green closures when needed. The working arrays
    are private copies; the input mesh is never mutated.

    Internal arrays (vertices, triangles, boundary) grow monotonically and are
    never compacted. Dead triangles are tombstoned via `_Status.GONE`, not
    removed, so every index structure stays valid across rounds.

    Every per-edge lookup is a dict keyed by the sorted vertex pair, and new vertices
    are accumulated in a list and stacked once at emission, so a round costs time
    linear in the triangles it touches rather than in the mesh.
    """

    def __init__(self, mesh: Mesh) -> None:
        n_nodes = mesh.elements.shape[1]
        if n_nodes != 3:
            raise NotImplementedError(
                f'red-green refinement is defined for triangles (3-node elements), '
                f'got {n_nodes}-node elements'
            )
        self._source_mesh: Mesh = mesh
        # The source vertices, then the midpoints created so far in index order; they
        # are stacked into one array only when a mesh is emitted.
        self._source_vertices: Vertices = mesh.vertices
        self._new_vertices: list[Vertices] = []
        # Boundary facets keyed by sorted edge, holding the facet's own orientation.
        # Insertion-ordered, so the emitted facets are the source facets in order with
        # each split facet replaced by its halves at the end.
        self._boundary: dict[Edge, tuple[int, int]] = {
            _edge_key(int(a), int(b)): (int(a), int(b)) for a, b in mesh.boundary}

        self._triangles: list[_Triangle] = []
        self._edge_to_tris: dict[Edge, set[int]] = {}
        self._edge_midpoints: dict[Edge, int] = {}
        # Curve each boundary edge lies on, so a new boundary vertex lands on the true
        # curve rather than the chord midpoint, and its two halves inherit the curve.
        # Empty for a straight-sided mesh, leaving midpoints exactly where they were.
        self._edge_curve: dict[Edge, Curve] = {}
        if mesh.boundary_curves is not None:
            for facet, curve in zip(mesh.boundary, mesh.boundary_curves):
                if curve is not None:
                    self._edge_curve[_edge_key(int(facet[0]), int(facet[1]))] = curve
        # Likewise the outline each boundary edge came from, so its halves keep the tag.
        self._edge_tag: dict[Edge, int] = {}
        if mesh.boundary_tags is not None:
            for facet, tag in zip(mesh.boundary, mesh.boundary_tags):
                self._edge_tag[_edge_key(int(facet[0]), int(facet[1]))] = int(tag)
        for element in mesh.elements:
            self._append_triangle(_Triangle(list(element)))

        self._tri_index_map: dict[int, int] = {
            idx: idx for idx in range(len(self._triangles))
        }
        self._pending: set[int] = set()

    def leaf_classifications(self) -> list[str]:
        """Return ``'red'`` or ``'green'`` for each leaf triangle.

        The order matches the elements array of the most recently emitted mesh.
        """
        classifications: list[str] = []
        for tri in self._triangles:
            if tri.status is _Status.RED_CHILD:
                classifications.append('red')
            elif tri.status is _Status.GREEN_CHILD:
                classifications.append('green')
        return classifications

    def refine(self, element_idxs: Sequence[int]) -> Mesh:
        """Refine the given elements and return the updated mesh.

        ``element_idxs`` are indices into the most recently emitted mesh (or the
        original mesh, on the first call).
        """
        # Every triangle still queued is known up front, so a neighbour that is about
        # to be refined red is not first closed green and then rolled back.
        self._pending = {self._tri_index_map[e_idx] for e_idx in element_idxs}
        while self._pending:
            self._refine_single(self._pending.pop())
        return self._emit_mesh()

    # -- internal: triangle list management ---------------------------------

    def _append_triangle(self, tri: _Triangle) -> int:
        tri.idx = len(self._triangles)
        self._triangles.append(tri)
        for edge in _tri_edges(tri):
            self._edge_to_tris.setdefault(edge, set()).add(tri.idx)
        return tri.idx

    def _mark_gone(self, tri: _Triangle) -> None:
        tri.status = _Status.GONE
        for edge in _tri_edges(tri):
            s = self._edge_to_tris.get(edge)
            if s is not None:
                s.discard(tri.idx)

    # -- internal: dispatch -------------------------------------------------

    def _refine_single(self, tri_idx: int) -> None:
        tri = self._triangles[tri_idx]
        if tri.status is _Status.RED_PARENT:
            return
        elif tri.status is _Status.RED_CHILD:
            self._refine_red(tri_idx)
        elif tri.status in (_Status.GREEN_PARENT, _Status.GREEN_CHILD):
            parent_idx = self._rollback_green(tri_idx)
            self._refine_red(parent_idx)
        elif tri.status is _Status.GONE:
            pass

    # -- internal: red / green / rollback -----------------------------------

    def _refine_red(self, tri_idx: int) -> list[int]:
        tri = self._triangles[tri_idx]
        new_point_idxs: list[int] = []
        for i in range(3):
            v0 = tri.vertex_idxs[i]
            v1 = tri.vertex_idxs[(i + 1) % 3]
            edge = [v0, v1]
            mid_idx = self._get_or_create_midpoint(v0, v1)
            new_point_idxs.append(mid_idx)
            self._update_boundary(edge, mid_idx)

        new_tris = [
            _Triangle(
                [tri.vertex_idxs[0], new_point_idxs[0], new_point_idxs[2]],
                parent=tri,
            ),
            _Triangle(
                [tri.vertex_idxs[1], new_point_idxs[1], new_point_idxs[0]],
                parent=tri,
            ),
            _Triangle(
                [tri.vertex_idxs[2], new_point_idxs[2], new_point_idxs[1]],
                parent=tri,
            ),
            _Triangle(
                [new_point_idxs[0], new_point_idxs[1], new_point_idxs[2]],
                parent=tri,
            ),
        ]
        tri.children = new_tris
        new_tri_idxs = [self._append_triangle(t) for t in new_tris]
        tri.status = _Status.RED_PARENT

        for i in range(3):
            edge = [tri.vertex_idxs[i], tri.vertex_idxs[(i + 1) % 3]]
            shared_idx = self._find_shared_triangle(edge, exclude={tri_idx})
            if shared_idx is None:
                continue
            shared = self._triangles[shared_idx]
            if shared.status is _Status.RED_PARENT:
                continue
            elif shared_idx in self._pending:
                continue
            elif shared.status is _Status.RED_CHILD:
                self._refine_green(shared_idx, edge, new_point_idxs[i])
            elif shared.status is _Status.GREEN_PARENT:
                parent_idx = self._rollback_green(shared_idx)
                self._refine_red(parent_idx)
            elif shared.status is _Status.GREEN_CHILD:
                parent_idx = self._rollback_green(shared_idx)
                child_idxs = self._refine_red(parent_idx)
                for new_idx in child_idxs:
                    child = self._triangles[new_idx]
                    if edge[0] in child.vertex_idxs and edge[1] in child.vertex_idxs:
                        self._refine_green(new_idx, edge, new_point_idxs[i])
                        break

        return new_tri_idxs

    def _rollback_green(self, tri_idx: int) -> int:
        tri = self._triangles[tri_idx]
        parent = tri if tri.children else tri.parent
        assert parent is not None

        parent.status = _Status.RED_PARENT
        for child in parent.children:
            self._mark_gone(child)
        parent.children = []
        return parent.idx

    def _refine_green(
        self,
        tri_idx: int,
        edge: list[int],
        mid_idx: int,
    ) -> None:
        tri = self._triangles[tri_idx]
        tri.status = _Status.GREEN_PARENT

        opposite = [v for v in tri.vertex_idxs if v not in edge][0]
        g1 = _Triangle(
            [edge[0], opposite, mid_idx],
            parent=tri,
            status=_Status.GREEN_CHILD,
        )
        g2 = _Triangle(
            [edge[1], opposite, mid_idx],
            parent=tri,
            status=_Status.GREEN_CHILD,
        )
        tri.children = [g1, g2]
        self._append_triangle(g1)
        self._append_triangle(g2)
        self._update_boundary(edge, mid_idx)

    # -- internal: O(1) lookups ---------------------------------------------

    def _find_shared_triangle(
        self,
        edge: list[int],
        exclude: set[int] | None = None,
    ) -> int | None:
        key = _edge_key(edge[0], edge[1])
        for idx in self._edge_to_tris.get(key, ()):
            if exclude is not None and idx in exclude:
                continue
            return idx
        return None

    def _vertex(self, idx: int) -> Vertices:
        n_source = len(self._source_vertices)
        return self._source_vertices[idx] if idx < n_source else self._new_vertices[idx - n_source]

    def _get_or_create_midpoint(self, v0: int, v1: int) -> int:
        key = _edge_key(v0, v1)
        mid = self._edge_midpoints.get(key)
        if mid is not None:
            return mid
        midpoint = (self._vertex(v0) + self._vertex(v1)) / 2
        curve = self._edge_curve.get(key)
        if curve is not None:
            midpoint = np.asarray(curve.project(midpoint))
        mid = len(self._source_vertices) + len(self._new_vertices)
        self._new_vertices.append(midpoint)
        self._edge_midpoints[key] = mid
        return mid

    # -- internal: boundary bookkeeping -------------------------------------

    def _update_boundary(self, edge: list[int], mid_idx: int) -> None:
        facet = self._boundary.pop(_edge_key(edge[0], edge[1]), None)
        if facet is None:
            return
        a, b = facet
        self._boundary[_edge_key(a, mid_idx)] = (a, mid_idx)
        self._boundary[_edge_key(mid_idx, b)] = (mid_idx, b)
        # The two halves lie on whatever curve and outline the split boundary edge did,
        # so a facet keeps following them however many times it is bisected.
        _carry_onto_halves(self._edge_curve, edge, mid_idx)
        _carry_onto_halves(self._edge_tag, edge, mid_idx)

    # -- internal: mesh emission --------------------------------------------

    def _emit_mesh(self) -> Mesh:
        """Build a new mesh from the current leaf triangles.

        Compaction (vertex renumbering) is applied only to the output mesh.
        Internal arrays keep their original indices so that edge and midpoint
        dicts stay valid across rounds.
        """
        self._tri_index_map = {}
        elements: list[list[int]] = []
        for tri_idx, tri in enumerate(self._triangles):
            if tri.status not in (_Status.RED_CHILD, _Status.GREEN_CHILD):
                continue
            elements.append(tri.vertex_idxs)
            self._tri_index_map[len(elements) - 1] = tri_idx
        elements_arr = np.array(elements)

        all_vertices = np.vstack([self._source_vertices, *self._new_vertices])
        used_idxs = np.unique(elements_arr)
        vertices = all_vertices[used_idxs]
        # Compaction: each old vertex index maps to its position in the sorted
        # used set, which searchsorted returns directly. Boundary facets are edges
        # of the emitted elements, so every boundary index is in used_idxs.
        remapped_elements = np.searchsorted(used_idxs, elements_arr)
        facets = list(self._boundary.values())
        remapped_boundary = np.searchsorted(used_idxs, np.array(facets))

        # Curves keyed by the original (uncompacted) endpoints, in the same facet order
        # as `_boundary`, so they align with the remapped boundary rows.
        boundary_curves = None
        if self._edge_curve:
            boundary_curves = [self._edge_curve.get(key) for key in self._boundary]

        boundary_tags = None
        if self._source_mesh.boundary_tags is not None:
            boundary_tags = np.array(
                [self._edge_tag.get(key, -1) for key in self._boundary], dtype=int)

        self._source_mesh = self._source_mesh.with_topology(
            vertices, remapped_elements, remapped_boundary, boundary_curves, boundary_tags,
        )
        return self._source_mesh

