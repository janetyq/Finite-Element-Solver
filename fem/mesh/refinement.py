"""Red-green triangle refinement.

The session object (`RedGreenRefiner`) owns a parent/child tree that tracks how
each triangle was produced.  This tree is what allows green-closure rollback:
when a green child is later marked for refinement, its parent can be recovered
and re-refined red, preserving mesh quality across successive rounds.

The tree is private state -- callers see only ``refine(idxs) -> Mesh``.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from fem.mesh.mesh import Edge, Mesh
from fem.typing import Vertices

logger = logging.getLogger(__name__)

_M = TypeVar('_M', bound=Mesh)


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


def _edge_key(v0: int, v1: int) -> Edge:
    return (v0, v1) if v0 < v1 else (v1, v0)


def _tri_edges(tri: _Triangle) -> list[Edge]:
    v = tri.vertex_idxs
    return [_edge_key(v[0], v[1]), _edge_key(v[1], v[2]), _edge_key(v[0], v[2])]


class RedGreenRefiner(Generic[_M]):
    """Persistent red-green refinement session over a triangle mesh.

    Wraps a mesh and maintains an internal hierarchy so that successive calls
    to `refine` can roll back green closures when needed.  The working arrays
    are private copies -- the input mesh is never mutated.

    Internal arrays (vertices, triangles, boundary) grow monotonically and are
    never compacted.  Dead triangles are tombstoned via `_Status.GONE`, not
    removed, so every index structure stays valid across rounds.
    """

    def __init__(self, mesh: _M) -> None:
        n_nodes = mesh.elements.shape[1]
        if n_nodes != 3:
            raise NotImplementedError(
                f'red-green refinement is defined for triangles (3-node elements), '
                f'got {n_nodes}-node elements'
            )
        self._source_mesh: _M = mesh
        self._vertices: Vertices = mesh.vertices.copy()
        self._boundary: list[list[int]] = [list(edge) for edge in mesh.boundary]

        self._triangles: list[_Triangle] = []
        self._edge_to_tris: dict[Edge, set[int]] = {}
        self._edge_midpoints: dict[Edge, int] = {}
        for element in mesh.elements:
            self._append_triangle(_Triangle(list(element)))

        self._tri_index_map: dict[int, int] = {
            idx: idx for idx in range(len(self._triangles))
        }

    def refine(self, element_idxs: Sequence[int]) -> _M:
        """Refine the given elements and return the updated mesh.

        ``element_idxs`` are indices into the most recently emitted mesh (or the
        original mesh, on the first call).
        """
        for e_idx in element_idxs:
            self._refine_single(self._tri_index_map[e_idx])
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

    def _get_or_create_midpoint(self, v0: int, v1: int) -> int:
        key = _edge_key(v0, v1)
        mid = self._edge_midpoints.get(key)
        if mid is not None:
            return mid
        midpoint = (self._vertices[v0] + self._vertices[v1]) / 2
        self._vertices = np.vstack((self._vertices, midpoint))
        mid = len(self._vertices) - 1
        self._edge_midpoints[key] = mid
        return mid

    # -- internal: boundary bookkeeping -------------------------------------

    def _update_boundary(self, edge: list[int], mid_idx: int) -> None:
        if edge in self._boundary:
            self._boundary.remove(edge)
            self._boundary.extend([[edge[0], mid_idx], [mid_idx, edge[1]]])
        elif edge[::-1] in self._boundary:
            self._boundary.remove(edge[::-1])
            self._boundary.extend([[edge[1], mid_idx], [mid_idx, edge[0]]])

    # -- internal: mesh emission --------------------------------------------

    def _emit_mesh(self) -> _M:
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

        used_idxs = sorted(set(elements_arr.ravel()))
        index_mapping = {old: new for new, old in enumerate(used_idxs)}

        vertices = self._vertices[used_idxs]
        remapped_elements = np.vectorize(index_mapping.get)(elements_arr)
        remapped_boundary = np.vectorize(index_mapping.get)(np.array(self._boundary))

        self._source_mesh = self._source_mesh.with_topology(
            vertices, remapped_elements, remapped_boundary,
        )
        return self._source_mesh

    # -- debug plotting -----------------------------------------------------

    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        edge: list[int] | None = None,
        main_idx: int | None = None,
        green_idx: int | None = None,
        red_idx: int | None = None,
        triangle_idxs: Sequence[int] | None = None,
    ) -> Axes:
        """Draw the refinement state for debugging."""
        from fem.plot.helpers import plot_mesh
        from fem.plot.plotter import Plotter

        if ax is None:
            ax = Plotter(title=title).get_ax()
        plot_mesh(ax, self._source_mesh, linewidth=3)
        if edge is not None:
            edge_verts = self._vertices[edge]
            ax.plot(edge_verts[:, 0], edge_verts[:, 1], linewidth=3, color='blue')
        if main_idx is not None:
            center = np.mean(self._vertices[self._triangles[main_idx].vertex_idxs], axis=0)
            ax.scatter(center[0], center[1], color='blue')
        if green_idx is not None:
            center = np.mean(self._vertices[self._triangles[green_idx].vertex_idxs], axis=0)
            ax.scatter(center[0], center[1], color='green')
        if red_idx is not None:
            center = np.mean(self._vertices[self._triangles[red_idx].vertex_idxs], axis=0)
            ax.scatter(center[0], center[1], color='red')
        if triangle_idxs is not None:
            for idx in triangle_idxs:
                center = np.mean(self._vertices[self._triangles[idx].vertex_idxs], axis=0)
                ax.scatter(center[0], center[1], color='black')

        leaf_tris = [
            t for t in self._triangles
            if t.status in (_Status.RED_CHILD, _Status.GREEN_CHILD)
        ]
        plotting_mesh = Mesh(
            self._vertices,
            [t.vertex_idxs for t in leaf_tris],
            self._boundary,
        )
        plot_mesh(ax, plotting_mesh, color='cyan', linewidth=1)
        return ax
