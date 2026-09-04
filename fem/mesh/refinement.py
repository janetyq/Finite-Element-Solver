"""Red-green triangle refinement.

`RedGreenRefiner` refines a triangle mesh round by round: an element marked for
refinement is split red (into four similar triangles at its edge midpoints), and a
neighbour that thereby gains one split edge is closed green (into two, from the
midpoint to the opposite corner) so the mesh stays conforming. A neighbour that gains
two split edges is promoted to red. Green triangles are provisional: when a later round
asks for one, or splits one of its edges, its parent is restored and refined red
instead, so green closures never stack and the angles stay bounded.

Each round is a handful of numpy passes over the whole leaf mesh: mark the edges of the
requested elements, iterate the closure (promotions and green rollbacks) to a fixed
point, place one midpoint per marked edge, and emit the children. The state carried
between rounds is the leaf mesh itself plus, for each green leaf, the parent it came
from and the edge that was split, which is all a rollback needs; no tree of every
triangle ever made is kept. Callers see only ``refine(idxs) -> Mesh``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fem.mesh.curves import Curve
from fem.mesh.mesh import Mesh
from fem.typing import BoolArray, Elements, IntArray, Vertices

# Local edge i of a triangle [v0, v1, v2] joins corners i and i + 1: (0, 1), (1, 2), (2, 0).
_EDGE_CORNERS = np.array([[0, 1], [1, 2], [2, 0]])


class RedGreenRefiner:
    """Persistent red-green refinement session over a triangle mesh.

    Wraps a mesh and keeps the leaf mesh between calls, so successive `refine` rounds
    can roll a green closure back to its parent before refining it red. The input mesh
    is never mutated; every round returns a new `Mesh` built through
    `Mesh.with_topology`, so boundary curves and tags carry onto the split facets.

    Vertex indices are stable across rounds (a midpoint, once placed, keeps its index)
    and the elements of the returned mesh are exactly the refiner's leaves, so the
    indices `refine` takes are positions in the mesh it last returned.
    """

    def __init__(self, mesh: Mesh) -> None:
        n_nodes = mesh.elements.shape[1]
        if n_nodes != 3:
            raise NotImplementedError(
                f'red-green refinement is defined for triangles (3-node elements), '
                f'got {n_nodes}-node elements'
            )
        self._source_mesh: Mesh = mesh
        self._vertices: Vertices = np.array(mesh.vertices, dtype=float)
        self._elements: Elements = np.array(mesh.elements, dtype=int)
        n = len(self._elements)
        # The green record: for a green leaf, the corners of the parent it halves, the
        # endpoints of the edge that was split, the midpoint on it, and the position of
        # its sibling. -1 throughout for a red leaf.
        self._is_green: BoolArray = np.zeros(n, dtype=bool)
        self._green_parent: IntArray = np.full((n, 3), -1, dtype=int)
        self._green_edge: IntArray = np.full((n, 2), -1, dtype=int)
        self._green_mid: IntArray = np.full(n, -1, dtype=int)
        self._green_sibling: IntArray = np.full(n, -1, dtype=int)
        # Boundary facets in their own orientation, with the curve and tag of each.
        self._boundary: IntArray = np.array(mesh.boundary, dtype=int).reshape(-1, 2)
        self._curves: list[Curve | None] | None = (
            list(mesh.boundary_curves) if mesh.boundary_curves is not None else None)
        self._tags: IntArray | None = (
            np.array(mesh.boundary_tags, dtype=int) if mesh.boundary_tags is not None else None)

    def leaf_classifications(self) -> list[str]:
        """``'red'`` or ``'green'`` for each leaf triangle, in the order of the elements
        of the most recently emitted mesh."""
        return ['green' if g else 'red' for g in self._is_green]

    def refine(self, element_idxs: Sequence[int]) -> Mesh:
        """Refine the given elements and return the updated mesh.

        ``element_idxs`` are indices into the most recently emitted mesh (or the
        original mesh, on the first call).
        """
        requested = np.zeros(len(self._elements), dtype=bool)
        requested[np.asarray(list(element_idxs), dtype=int)] = True

        # Edge keys are `lo * stride + hi`; the midpoints placed this round get indices
        # past the current vertices, so the stride leaves room for one per edge.
        stride = len(self._vertices) + 3 * len(self._elements) + 1
        elements, is_green, parent, split_edge, split_mid, sibling = (
            self._elements, self._is_green, self._green_parent, self._green_edge,
            self._green_mid, self._green_sibling)
        marked = np.zeros(0, dtype=np.int64)          # sorted unique keys of split edges
        known_keys = np.zeros(0, dtype=np.int64)      # split edges whose midpoint already exists
        known_mids = np.zeros(0, dtype=int)

        # The closure: a requested element marks its three edges; an element with two
        # marked edges is promoted and marks its third; a green leaf with any marked edge
        # is rolled back to its parent, which is refined red. Each pass is over the whole
        # working set, and the passes stop when nothing changes.
        while True:
            keys = self._edge_keys(elements, stride)                     # (n, 3)
            # A requested green leaf marks nothing itself: it is rolled back and its
            # parent marks the parent's edges, whose halves the pair's own edges are.
            marked = np.union1d(marked, keys[requested & ~is_green].ravel())
            count = np.isin(keys, marked).sum(axis=1)
            promote = (count >= 2) & ~requested & ~is_green
            rollback = is_green & (requested | (count >= 1))
            if not promote.any() and not rollback.any():
                break
            requested = requested | promote
            if rollback.any():
                # Both siblings of a rolled-back pair go, whichever was touched.
                gone = rollback | np.isin(np.arange(len(elements)), sibling[rollback])
                gone_green = gone & is_green
                # One parent per pair: take it from the sibling with the lower position.
                first_of_pair = gone_green & (sibling > np.arange(len(elements)))
                parents = parent[first_of_pair]
                known_keys = np.concatenate([known_keys, self._pair_keys(split_edge[first_of_pair], stride)])
                known_mids = np.concatenate([known_mids, split_mid[first_of_pair]])
                keep = ~gone
                n_parents = len(parents)
                elements = np.vstack([elements[keep], parents])
                requested = np.concatenate([requested[keep], np.ones(n_parents, dtype=bool)])
                is_green = np.concatenate([is_green[keep], np.zeros(n_parents, dtype=bool)])
                # The survivors' sibling positions shift with the compaction.
                new_position = np.cumsum(keep) - 1
                sib = sibling[keep]
                sib = np.where(sib >= 0, new_position[np.maximum(sib, 0)], -1)
                sibling = np.concatenate([sib, np.full(n_parents, -1, dtype=int)])
                parent = np.vstack([parent[keep], np.full((n_parents, 3), -1, dtype=int)])
                split_edge = np.vstack([split_edge[keep], np.full((n_parents, 2), -1, dtype=int)])
                split_mid = np.concatenate([split_mid[keep], np.full(n_parents, -1, dtype=int)])

        keys = self._edge_keys(elements, stride)
        is_marked = np.isin(keys, marked)                                # (n, 3)
        assert not np.any(is_marked.sum(axis=1) == 2), 'closure left an element with two split edges'

        # One midpoint per marked edge that still exists: a rolled-back parent's split
        # edge already has its vertex, every other marked edge gets a new one.
        split_keys = np.unique(keys[is_marked])
        mid_of_key = self._place_midpoints(split_keys, known_keys, known_mids, stride)
        self._elements, self._is_green, self._green_parent, self._green_edge, self._green_mid,             self._green_sibling = elements, is_green, parent, split_edge, split_mid, sibling

        # Two emissions. The first turns the working set into its children. A red child of
        # a rolled-back parent can then carry a split edge of its own: half of the
        # parent's old split edge, marked by the neighbour across it, which is refining
        # red this round. The second emission closes those children green; nothing it
        # emits can have a split edge (a green child's outer edges are full edges of a
        # parent that was not promoted), so it is the last.
        for _ in range(2):
            count, mids = self._split_edges(self._elements, split_keys, mid_of_key, stride)
            if not count.any():
                break
            self._emit(count, mids)
        assert not self._split_edges(self._elements, split_keys, mid_of_key, stride)[0].any()

        self._split_boundary(split_keys, mid_of_key, stride)
        return self._emit_mesh()

    @classmethod
    def _split_edges(cls, elements: Elements, split_keys: IntArray, mid_of_key: IntArray,
                     stride: int) -> tuple[IntArray, IntArray]:
        '''Per element, how many of its edges are split and the midpoint on each
        (-1 on an unsplit edge).'''
        keys = cls._edge_keys(elements, stride)
        is_split = np.isin(keys, split_keys)
        mids = np.full(keys.shape, -1, dtype=int)
        mids[is_split] = mid_of_key[np.searchsorted(split_keys, keys[is_split])]
        return is_split.sum(axis=1), mids

    # -- keys ----------------------------------------------------------------

    @staticmethod
    def _pair_keys(pairs: IntArray, stride: int) -> IntArray:
        """One integer per (a, b) row, the same for either orientation."""
        lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        return lo * stride + hi

    @classmethod
    def _edge_keys(cls, elements: Elements, stride: int) -> IntArray:
        """(n_elements, 3) keys of the local edges (0, 1), (1, 2), (2, 0)."""
        pairs = elements[:, _EDGE_CORNERS]                               # (n, 3, 2)
        return cls._pair_keys(pairs.reshape(-1, 2), stride).reshape(len(elements), 3)

    # -- midpoints -----------------------------------------------------------

    def _place_midpoints(self, split_keys: IntArray, known_keys: IntArray,
                         known_mids: IntArray, stride: int) -> IntArray:
        """The vertex index of the midpoint of each of `split_keys` (sorted), creating
        the vertices that do not exist yet; a boundary midpoint on a curve is projected
        onto it."""
        mid_of_key = np.full(len(split_keys), -1, dtype=int)
        if len(known_keys):
            found = np.isin(split_keys, known_keys)
            order = np.argsort(known_keys)
            mid_of_key[found] = known_mids[order][np.searchsorted(known_keys[order], split_keys[found])]
        new = mid_of_key < 0
        new_keys = split_keys[new]
        a, b = new_keys // stride, new_keys % stride
        points = 0.5 * (self._vertices[a] + self._vertices[b])
        if self._curves is not None and len(new_keys):
            facet_keys = self._pair_keys(self._boundary, stride)
            # Each curve projects the midpoints of all its split facets in one call.
            for curve in {id(c): c for c in self._curves if c is not None}.values():
                on_curve = facet_keys[[c is curve for c in self._curves]]
                which = np.flatnonzero(np.isin(new_keys, on_curve))
                if len(which):
                    points[which] = curve.project(points[which])
        mid_of_key[new] = len(self._vertices) + np.arange(len(new_keys))
        self._vertices = np.vstack([self._vertices, points])
        return mid_of_key

    # -- emission ------------------------------------------------------------

    def _emit(self, count: IntArray, mids: IntArray) -> None:
        """Replace the leaves by their children, each element in its own place: four
        red children for three split edges, two green for one, itself for none."""
        elements, is_green, parent, split_edge, split_mid, sibling = (
            self._elements, self._is_green, self._green_parent, self._green_edge,
            self._green_mid, self._green_sibling)
        n = len(elements)
        assert not np.any(count == 2)
        v = elements
        m = mids
        red = count == 3
        green = count == 1
        keep = count == 0

        # Red: [v0, m01, m20], [v1, m12, m01], [v2, m20, m12], [m01, m12, m20], where mi is
        # the midpoint of local edge i.
        r = np.flatnonzero(red)
        red_children = np.stack([
            np.stack([v[r, 0], m[r, 0], m[r, 2]], axis=1),
            np.stack([v[r, 1], m[r, 1], m[r, 0]], axis=1),
            np.stack([v[r, 2], m[r, 2], m[r, 1]], axis=1),
            np.stack([m[r, 0], m[r, 1], m[r, 2]], axis=1),
        ], axis=1).reshape(-1, 3)                                        # (4 n_red, 3)

        # Green: the split edge i joins corners i and i + 1; the children are
        # [corner i, opposite, mid] and [corner i + 1, opposite, mid].
        g = np.flatnonzero(green)
        i = np.argmax(mids[g] >= 0, axis=1)
        c0, c1, opp = i, (i + 1) % 3, (i + 2) % 3
        e0, e1, o = v[g, c0], v[g, c1], v[g, opp]
        gm = m[g, i]
        green_children = np.stack([
            np.stack([e0, o, gm], axis=1),
            np.stack([e1, o, gm], axis=1),
        ], axis=1).reshape(-1, 3)                                        # (2 n_green, 3)
        green_parent = np.repeat(v[g], 2, axis=0)
        green_edge = np.repeat(np.stack([e0, e1], axis=1), 2, axis=0)
        green_mid = np.repeat(gm, 2)

        # Every child lands where its parent was: a stable sort on the parent's position.
        k = np.flatnonzero(keep)
        source = np.concatenate([k, np.repeat(r, 4), np.repeat(g, 2)])
        order = np.argsort(source, kind='stable')
        new_elements = np.vstack([elements[k], red_children, green_children])[order]
        n_keep, n_red = len(k), 4 * len(r)
        new_is_green = np.concatenate([
            is_green[k], np.zeros(n_red, dtype=bool), np.ones(2 * len(g), dtype=bool)])[order]
        new_parent = np.vstack([parent[k], np.full((n_red, 3), -1, dtype=int), green_parent])[order]
        new_edge = np.vstack([split_edge[k], np.full((n_red, 2), -1, dtype=int), green_edge])[order]
        new_mid = np.concatenate([split_mid[k], np.full(n_red, -1, dtype=int), green_mid])[order]

        # Siblings: a surviving green leaf keeps its sibling, which also survived (a split
        # edge on either would have rolled both back); a new green pair sits at
        # consecutive slots, and both are read through the sort's slot -> position map.
        position = np.empty(len(order), dtype=int)
        position[order] = np.arange(len(order))
        slot_of_working = np.full(n, -1, dtype=int)
        slot_of_working[k] = np.arange(n_keep)
        new_sibling = np.full(len(order), -1, dtype=int)
        paired = k[sibling[k] >= 0]
        new_sibling[position[slot_of_working[paired]]] = position[slot_of_working[sibling[paired]]]
        first = n_keep + n_red + 2 * np.arange(len(g))
        new_sibling[position[first]] = position[first + 1]
        new_sibling[position[first + 1]] = position[first]

        self._elements = new_elements
        self._is_green = new_is_green
        self._green_parent = new_parent
        self._green_edge = new_edge
        self._green_mid = new_mid
        self._green_sibling = new_sibling

    def _emit_mesh(self) -> Mesh:
        curves = None if self._curves is None else list(self._curves)
        tags = None if self._tags is None else self._tags.copy()
        self._source_mesh = self._source_mesh.with_topology(
            self._vertices.copy(), self._elements.copy(), self._boundary.copy(), curves, tags)
        return self._source_mesh

    # -- boundary ------------------------------------------------------------

    def _split_boundary(self, split_keys: IntArray, mid_of_key: IntArray, stride: int) -> None:
        """Replace each boundary facet on a split edge by its halves, in place and in
        the facet's own orientation; both halves keep the facet's curve and tag."""
        if not len(self._boundary):
            return
        facet_keys = self._pair_keys(self._boundary, stride)
        split = np.isin(facet_keys, split_keys)
        if not split.any():
            return
        mids = mid_of_key[np.searchsorted(split_keys, facet_keys[split])]
        a, b = self._boundary[split, 0], self._boundary[split, 1]
        halves = np.stack([np.stack([a, mids], axis=1), np.stack([mids, b], axis=1)], axis=1)  # (s, 2, 2)
        # Each facet becomes two rows where it stood.
        repeat = np.where(split, 2, 1)
        expanded = np.repeat(self._boundary, repeat, axis=0)
        starts = np.cumsum(repeat) - repeat
        expanded[starts[split]] = halves[:, 0]
        expanded[starts[split] + 1] = halves[:, 1]
        self._boundary = expanded
        if self._curves is not None:
            self._curves = [c for c, r in zip(self._curves, repeat) for _ in range(r)]
        if self._tags is not None:
            self._tags = np.repeat(self._tags, repeat)
