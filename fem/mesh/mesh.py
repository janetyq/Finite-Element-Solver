"""The mesh: vertices, elements, boundary facets, and the topology and geometry
queries built on them.

`refined`, `save`, and `load` import `fem.mesh.refinement` and `fem.post.io` lazily: they are
methods on the mesh for convenience, and both modules sit above it.
"""
import itertools
import math
from collections.abc import Sequence
from functools import cached_property

import numpy as np

from fem.mesh.curves import Curve
from fem.typing import Elements, FloatArray, IntArray, Vertices

Edge = tuple[int, int]

# Node counts of the linear simplices a Mesh holds: a line (2), a triangle (3),
# a tet (4). Higher-node (quadratic) elements are the FunctionSpace's concern
# (it adds midside DOFs on top of a P1 Mesh), not the geometry's.
_SIMPLEX_NODE_COUNTS = (2, 3, 4)

_ELEMENT_NAMES = {1: 'line', 2: 'triangle', 3: 'tet'}


def frozen_array(array: np.ndarray) -> np.ndarray:
    '''`array`, made read-only in place: the arrays of an immutable object.'''
    array.setflags(write=False)
    return array


def boundary_facets(elements: Elements) -> Elements:
    '''The facets belonging to exactly one element, as sorted vertex-index rows.

    A facet is the codimension-1 face of an element (an edge of a triangle, a face of
    a tet). Every element's facets are listed, sorted within the row, and grouped with
    one `np.unique`; those seen once are the boundary. The facets are unoriented, which
    is all the boundary mass matrix and the region resolution need.
    '''
    elements = np.asarray(elements, dtype=int)
    n_nodes = elements.shape[1]
    if len(elements) == 0:
        return np.zeros((0, n_nodes - 1), dtype=int)
    # Dropping each node in turn gives the n_nodes facets of a simplex.
    keep = np.array([[j for j in range(n_nodes) if j != i] for i in range(n_nodes)])
    facets = np.sort(elements[:, keep].reshape(-1, n_nodes - 1), axis=1)
    unique, counts = np.unique(facets, axis=0, return_counts=True)
    return unique[counts == 1]


def triangle_angles(triangle: FloatArray) -> FloatArray:
    '''The three interior angles, in degrees, angle `i` at vertex `i`.

    Takes a single `(3, d)` triangle and returns `(3,)`, or a stacked `(..., 3, d)`
    array of triangles and returns `(..., 3)`. Mesh refinement tests every element on
    every pass, so the batched form is the one that keeps that loop off Python.
    '''
    points = np.asarray(triangle, dtype=float)
    # Side i is opposite vertex i.
    sides = np.linalg.norm(np.roll(points, -1, axis=-2) - np.roll(points, 1, axis=-2), axis=-1)
    a, b, c = sides[..., 0], sides[..., 1], sides[..., 2]
    # Law of cosines. Clipped because a degenerate triangle can put the ratio a
    # hair outside [-1, 1], and a NaN angle would compare false against any bound,
    # accepting the sliver as good.
    cosines = np.stack([
        (b**2 + c**2 - a**2) / (2 * b * c),
        (c**2 + a**2 - b**2) / (2 * c * a),
        (a**2 + b**2 - c**2) / (2 * a * b),
    ], axis=-1)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def triangle_min_angle(triangle: FloatArray) -> FloatArray:
    '''The smallest interior angle, in degrees. Batches like `triangle_angles`, one
    angle per triangle.'''
    return triangle_angles(triangle).min(axis=-1)


def _edge_node_pairs(n_nodes: int) -> IntArray:
    '''Local node-index pairs spanning the edges of one linear simplex.

    Every pair of nodes: 1 for a line, 3 for a triangle, 6 for a tet. Used to
    lift per-element connectivity into batched (n_elements, n_pairs, ...) form.
    '''
    return np.array(list(itertools.combinations(range(n_nodes), 2)))


class Mesh:
    '''A linear simplex mesh: vertices, the elements over them, and the boundary facets.

    `boundary` is derived from `elements` when not given: a facet is on the boundary
    when exactly one element has it. Pass it only to fix the facet order, as a mesh
    reloaded from a file does so that `boundary_curves` and `boundary_tags` stay
    aligned with its facets.

    `boundary_tags` is one integer per boundary facet, or None: which outline the facet
    came from. Ruppert's tags by the `PSLG` loop id, so the hole in a plate is the
    facets tagged 1, and refinement carries a tag onto a facet's halves, so a condition
    written as `on_tag(1)` resolves on any mesh in the hierarchy.

    The arrays are read-only. Every derived table below is cached, and a mesh is shared
    by the spaces, solutions, and refiners built on it, so a change in place would leave
    all of them stale; `displaced`, `refined`, and `with_topology` build a new mesh.
    '''

    def __init__(
        self,
        vertices: Vertices | Sequence[Sequence[float]],
        elements: Elements | Sequence[Sequence[int]],
        boundary: Elements | Sequence[Sequence[int]] | None = None,
        boundary_curves: Sequence[Curve | None] | None = None,
        boundary_tags: IntArray | Sequence[int] | None = None,
    ) -> None:
        self._vertices: Vertices = frozen_array(np.array(vertices, dtype=float))
        self._elements: Elements = frozen_array(np.array(elements, dtype=int))
        self._validate_elements()
        if boundary is None:
            boundary = boundary_facets(self._elements)
        self._boundary: Elements = frozen_array(np.array(boundary, dtype=int))
        self._validate_boundary()
        self._boundary_idxs: IntArray | None = None
        # Optional analytic curve each boundary facet lies on (or None), aligned with
        # `boundary` rows. None (the default) is a fully straight-sided mesh; a curved
        # (isoparametric) space reads these to put its boundary nodes on the true curve.
        self.boundary_curves: tuple[Curve | None, ...] | None = self._per_facet(
            'boundary_curves', tuple(boundary_curves) if boundary_curves is not None else None)
        self.boundary_tags: IntArray | None = self._per_facet(
            'boundary_tags',
            frozen_array(np.array(boundary_tags, dtype=int)) if boundary_tags is not None else None)

    # -- the arrays --------------------------------------------------------------------

    @property
    def vertices(self) -> Vertices:
        '''(n_vertices, spatial_dim) coordinates.'''
        return self._vertices

    @property
    def elements(self) -> Elements:
        '''(n_elements, n_nodes) vertex indices per element.'''
        return self._elements

    @property
    def boundary(self) -> Elements:
        '''(n_facets, n_nodes - 1) vertex indices per boundary facet.'''
        return self._boundary

    def _validate_elements(self) -> None:
        '''Reject malformed topology at the source with a named error.

        Without this a wrong-rank or out-of-range array survives the constructor
        and fails much later inside `ElementGeometry` or a scatter, with an
        opaque shape error far from the call that introduced it. `Mesh` is the
        entry point for user data (`Mesh.load`, hand-built meshes), so this is
        where a clear message pays off.
        '''
        if self._vertices.ndim != 2:
            raise ValueError(
                'vertices must be a 2D (n_vertices, spatial_dim) array, '
                f'got shape {self._vertices.shape}'
            )
        if self._elements.ndim != 2:
            raise ValueError(
                'elements must be a 2D (n_elements, n_nodes) array, '
                f'got shape {self._elements.shape}'
            )
        n_nodes = self._elements.shape[1]
        if n_nodes not in _SIMPLEX_NODE_COUNTS:
            raise NotImplementedError(
                'elements must be linear simplices with 2, 3, or 4 nodes '
                f'(a line, triangle, or tet), got {n_nodes}-node elements'
            )
        self._check_indices_in_range(self._elements, self.n_vertices, 'element')

    def _validate_boundary(self) -> None:
        if not self._boundary.size:
            return
        n_nodes = self._elements.shape[1]
        if self._boundary.ndim != 2:
            raise ValueError(
                'boundary must be a 2D (n_facets, n_nodes) array, '
                f'got shape {self._boundary.shape}'
            )
        if self._boundary.shape[1] != n_nodes - 1:
            raise ValueError(
                f'a boundary facet of a {n_nodes}-node element has '
                f'{n_nodes - 1} nodes, got {self._boundary.shape[1]}'
            )
        self._check_indices_in_range(self._boundary, self.n_vertices, 'boundary')

    def _per_facet(self, name, values):
        '''`values`, checked to have one entry per boundary facet (or be None).'''
        if values is not None and len(values) != len(self._boundary):
            raise ValueError(
                f'{name} has {len(values)} entries but the mesh has '
                f'{len(self._boundary)} boundary facets'
            )
        return values

    @staticmethod
    def _check_indices_in_range(indices: IntArray, n_vertices: int, name: str) -> None:
        if not indices.size:
            return
        lo, hi = int(indices.min()), int(indices.max())
        if lo < 0 or hi >= n_vertices:
            raise ValueError(
                f'{name} node indices must be in [0, {n_vertices}), '
                f'got range [{lo}, {hi}]'
            )

    # -- sizes and extent --------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    @property
    def spatial_dim(self) -> int:
        '''Dimension of the space the nodes live in.

        Distinct from `element_dim`: a triangle mesh embedded in 3D has spatial_dim 3
        but element_dim 2.
        '''
        return int(self._vertices.shape[1])

    @property
    def element_dim(self) -> int:
        '''Dimension of the elements themselves: 1 for lines, 2 for triangles, 3 for tets.'''
        return int(self._elements.shape[1]) - 1

    @cached_property
    def bounds(self) -> tuple[FloatArray, FloatArray]:
        '''The axis-aligned extent, `(lower, upper)`, each of length `spatial_dim`.'''
        return self._vertices.min(axis=0), self._vertices.max(axis=0)

    @property
    def boundary_idxs(self) -> IntArray:
        '''The unique vertex indices on the boundary, ascending.'''
        # A plain property over a cached value, rather than `cached_property`, so it
        # satisfies the `NodeGeometry` protocol's read-only property as pyright sees it.
        if self._boundary_idxs is None:
            self._boundary_idxs = frozen_array(np.unique(self._boundary.ravel()))
        return self._boundary_idxs

    # -- element geometry --------------------------------------------------------------

    @cached_property
    def centroids(self) -> FloatArray:
        '''(n_elements, spatial_dim) element centroids.'''
        return self._vertices[self._elements].mean(axis=1)

    @cached_property
    def element_measures(self) -> FloatArray:
        '''Length, area, or volume of each element, by `element_dim`.

        The Gram determinant of the edge vectors from one corner, rooted and divided
        by d!, which is one formula for every dimension and holds for an element
        embedded in a higher space (a triangle in 3D) as well.
        '''
        corners = self._vertices[self._elements]                       # (n_el, n_nodes, dim)
        edges = corners[:, 1:] - corners[:, :1]                        # (n_el, d, dim)
        gram = np.einsum('eid,ejd->eij', edges, edges)                 # (n_el, d, d)
        return np.sqrt(np.abs(np.linalg.det(gram))) / math.factorial(self.element_dim)

    @property
    def measure(self) -> float:
        '''Total length, area, or volume of the mesh.'''
        return float(self.element_measures.sum())

    @property
    def area(self) -> float:
        '''`measure`, named for a triangle mesh.'''
        if self.element_dim != 2:
            raise ValueError(
                f'area is for a triangle mesh; this one has {_ELEMENT_NAMES[self.element_dim]}s')
        return self.measure

    @cached_property
    def element_diameters(self) -> FloatArray:
        '''Maximum edge length per element: the h_K in error estimates.'''
        pairs = _edge_node_pairs(self._elements.shape[1])
        corners = self._vertices[self._elements]                        # (n_el, n_nodes, dim)
        edge_vecs = corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]]  # (n_el, n_pairs, dim)
        return np.linalg.norm(edge_vecs, axis=2).max(axis=1)

    @cached_property
    def min_angle(self) -> float:
        '''The smallest interior angle of any triangle, in degrees: the quality a
        Delaunay refinement was asked to guarantee, and what red-green refinement
        does not.'''
        if self.element_dim != 2:
            raise ValueError('min_angle is for a triangle mesh')
        return float(triangle_min_angle(self._vertices[self._elements]).min())

    # -- new meshes from this one ------------------------------------------------------

    def locate(self, points: Vertices, tol: float = 1e-9) -> tuple[IntArray, FloatArray]:
        '''The element containing each of `points`, and the point's reference coordinates
        in it: `(elements (n_points,), reference (n_points, element_dim))`.

        The reference coordinates are the barycentric weights of corners 1.. (corner 0
        is the origin), the affine map every straight-sided element shares, so a P2
        field evaluates through them as well. Candidates are the elements nearest by
        centroid, falling back to every element for a point none of them holds; a
        point outside the mesh (past `tol` in barycentric terms, shared edges and
        corners included) is an error. A point on a shared facet belongs to whichever
        of its elements is tested first.
        '''
        from scipy.spatial import KDTree

        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != self.spatial_dim:
            raise ValueError(
                f'points are {points.shape[1]}-dimensional, the mesh is {self.spatial_dim}')
        corners = self._vertices[self._elements[:, :self.element_dim + 1]]   # (n_el, d+1, d)
        # T[e, i, r] = corner_{r+1} - corner_0, so T lambda = p - corner_0.
        edges = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        inverse = np.linalg.inv(edges)                                       # (n_el, d, d)

        def reference_in(candidates: IntArray, p: FloatArray) -> FloatArray:
            '''(n_candidates, d) reference coordinates of `p` in each candidate element.'''
            return np.einsum('eri,ei->er', inverse[candidates], p - corners[candidates, 0])

        def inside(reference: FloatArray) -> np.ndarray:
            return (reference >= -tol).all(axis=1) & (reference.sum(axis=1) <= 1 + tol)

        k = min(self.n_elements, 2 * (self.element_dim + 1) ** 2)
        _, nearest = KDTree(self.centroids).query(points, k=k)
        nearest = np.atleast_2d(nearest).reshape(len(points), -1)
        elements = np.full(len(points), -1)
        reference = np.zeros((len(points), self.element_dim))
        for i, p in enumerate(points):
            for candidates in (nearest[i], np.arange(self.n_elements)):
                ref = reference_in(candidates, p)
                hits = np.flatnonzero(inside(ref))
                if len(hits):
                    elements[i] = candidates[hits[0]]
                    reference[i] = ref[hits[0]]
                    break
            else:
                raise ValueError(f'point {p} lies outside the mesh')
        return elements, reference

    def displaced(self, displacement: FloatArray, scale: float = 1.0) -> 'Mesh':
        '''The mesh with every vertex moved by `scale * displacement`.

        `displacement` is per vertex, `(n_vertices, spatial_dim)`, or a flat vector in
        that order. A longer vector (a P2 DOF vector, whose edge nodes follow the
        vertices) is read for its leading vertex entries, so the warp is the field's P1
        restriction. The topology is unchanged, so the facets keep their curves.
        '''
        displacement = np.asarray(displacement, dtype=float)
        if displacement.ndim == 1:
            displacement = displacement.reshape(-1, self.spatial_dim)
        if len(displacement) < self.n_vertices:
            raise ValueError(
                f'displacement covers {len(displacement)} vertices, '
                f'the mesh has {self.n_vertices}')
        vertices = self._vertices + scale * displacement[:self.n_vertices]
        return Mesh(vertices, self._elements, self._boundary, self.boundary_curves,
                    self.boundary_tags)

    def refined(self, element_idxs: Sequence[int] | None = None) -> 'Mesh':
        '''Red-green refinement of the given elements, or of every element when None.'''
        from fem.mesh.refinement import RedGreenRefiner
        idxs = range(self.n_elements) if element_idxs is None else element_idxs
        return RedGreenRefiner(self).refine([int(i) for i in idxs])

    def with_topology(
        self,
        vertices: Vertices,
        elements: Elements,
        boundary: Elements,
        boundary_curves: Sequence[Curve | None] | None = None,
        boundary_tags: IntArray | None = None,
    ) -> 'Mesh':
        '''A new mesh over the given topology.

        The seam remeshers build through, so that refinement and coarsening name
        what they are doing rather than reaching for the constructor. A remesher that
        keeps its boundary on the same outlines passes the (remapped) `boundary_curves`
        and `boundary_tags`; omitting them yields a straight-sided, untagged mesh, since
        the new facets are otherwise unrelated to the original facets.
        '''
        return Mesh(vertices, elements, boundary, boundary_curves, boundary_tags)

    # -- files -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        from fem.post.io import save_mesh
        save_mesh(self, path)

    @classmethod
    def load(cls, path: str) -> 'Mesh':
        from fem.post.io import load_mesh
        return load_mesh(path)

    def __repr__(self) -> str:
        return (f'Mesh({self.n_vertices} vertices, {self.n_elements} '
                f'{_ELEMENT_NAMES[self.element_dim]}s, {len(self._boundary)} boundary facets, '
                f'{self.spatial_dim}D)')

    # -- connectivity ------------------------------------------------------------------

    @cached_property
    def _edge_table(self) -> tuple[IntArray, IntArray]:
        '''Unique edges and the elements meeting at each, grouped in one pass.

        Returns `(edges, edge_elements)`. `edges` is the (n_edges, 2) array of
        sorted vertex pairs; `edge_elements` is the (n_edges, 2) element indices
        sharing each edge, with -1 in the second slot of a boundary edge (one
        element). The whole connectivity is grouped with one `np.unique` and an
        `argsort`, rather than a Python loop per element.

        Lazy, like its connectivity siblings: a P1 solve with no refinement or
        estimation never reads `edges`/`edge_elements`, so the build is skipped.
        '''
        node_pairs = _edge_node_pairs(self._elements.shape[1])
        n_pairs = len(node_pairs)
        edge_rows = np.sort(self._elements[:, node_pairs].reshape(-1, 2), axis=1)
        owners = np.repeat(np.arange(self.n_elements), n_pairs)
        edges, inverse = np.unique(edge_rows, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

        # Sorting the inverse lines each edge's rows up contiguously; the counts
        # (1 for a boundary edge, 2 for an interior one) say where each starts.
        owners_by_edge = owners[np.argsort(inverse, kind='stable')]
        counts = np.bincount(inverse, minlength=len(edges))
        starts = np.zeros(len(edges), dtype=int)
        starts[1:] = np.cumsum(counts)[:-1]

        edge_elements = np.full((len(edges), 2), -1, dtype=int)
        edge_elements[:, 0] = owners_by_edge[starts]
        interior = counts >= 2
        edge_elements[interior, 1] = owners_by_edge[starts[interior] + 1]
        return edges, edge_elements

    @cached_property
    def edge_to_elements(self) -> dict[Edge, list[int]]:
        '''Map each sorted edge to the indices of elements that contain it.

        Interior edges map to exactly two elements; boundary edges to one.
        '''
        edges, edge_elements = self._edge_table
        return {
            (int(v0), int(v1)): [int(e) for e in elems if e >= 0]
            for (v0, v1), elems in zip(edges, edge_elements)
        }

    @cached_property
    def edge_elements(self) -> IntArray:
        '''(n_edges, 2) element indices meeting at each edge in `edges`.

        The second column is -1 where the edge has a single element (a boundary
        edge), so `edge_elements[:, 1] >= 0` masks the interior edges. This is
        the batched form the residual estimator jumps the flux across.
        '''
        return self._edge_table[1]

    @cached_property
    def element_neighbours(self) -> list[list[int]]:
        '''For each element, the indices of elements sharing at least one edge.'''
        neighbours: list[set[int]] = [set() for _ in range(self.n_elements)]
        for elements in self.edge_to_elements.values():
            if len(elements) == 2:
                a, b = elements
                neighbours[a].add(b)
                neighbours[b].add(a)
        return [sorted(s) for s in neighbours]

    @cached_property
    def edges(self) -> IntArray:
        '''Every edge in the mesh, as sorted (v0, v1) index pairs.

        For a linear simplex the edge set is every pair of its nodes:
        1 pair for a line, 3 for a triangle, 6 for a tet. That makes this
        dimension-general without a per-shape table; it holds only for linear
        simplices, which the constructor guarantees (quadratic elements carry
        midside nodes, so pairing every node would invent edges that don't
        exist).
        '''
        return self._edge_table[0]
