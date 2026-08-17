import itertools
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


def _edge_node_pairs(n_nodes: int) -> IntArray:
    '''Local node-index pairs spanning the edges of one linear simplex.

    Every pair of nodes: 1 for a line, 3 for a triangle, 6 for a tet. Used to
    lift per-element connectivity into batched (n_elements, n_pairs, ...) form.
    '''
    return np.array(list(itertools.combinations(range(n_nodes), 2)))


class Mesh:
    def __init__(
        self,
        vertices: Vertices | Sequence[Sequence[float]],
        elements: Elements | Sequence[Sequence[int]],
        boundary: Elements | Sequence[Sequence[int]],
        boundary_curves: Sequence[Curve | None] | None = None,
        cell_tags: IntArray | Sequence[int] | None = None,
        facet_tags: IntArray | Sequence[int] | None = None,
        tag_names: dict[int, str] | None = None,
    ) -> None:
        self.vertices: Vertices = np.array(vertices)
        self.elements: Elements = np.array(elements)  # vertex indices per element
        self.boundary: Elements = np.array(boundary)  # vertex indices per facet
        self._validate()
        self.boundary_idxs: IntArray = np.unique(self.boundary.ravel())
        # Optional analytic curve each boundary facet lies on (or None), aligned with
        # `boundary` rows. None (the default) is a fully straight-sided mesh; a curved
        # (isoparametric) space reads these to put its boundary nodes on the true curve.
        self.boundary_curves: list[Curve | None] | None = (
            list(boundary_curves) if boundary_curves is not None else None
        )
        if self.boundary_curves is not None and len(self.boundary_curves) != len(self.boundary):
            raise ValueError(
                f'boundary_curves has {len(self.boundary_curves)} entries but the mesh '
                f'has {len(self.boundary)} boundary facets'
            )
        # Optional integer physical-group tags per element and per boundary facet, with
        # `tag_names` naming the ids (e.g. 3 -> "inlet"). These carry the named regions a
        # standard mesh format (Gmsh) records; `on_tag` turns a facet tag into a region a
        # boundary condition can use. None means untagged, the fully geometric default.
        self.cell_tags: IntArray | None = (
            np.asarray(cell_tags, dtype=int) if cell_tags is not None else None
        )
        self.facet_tags: IntArray | None = (
            np.asarray(facet_tags, dtype=int) if facet_tags is not None else None
        )
        self.tag_names: dict[int, str] = dict(tag_names) if tag_names is not None else {}
        self._validate_tags()

    def _validate(self) -> None:
        '''Reject malformed topology at the source with a named error.

        Without this a wrong-rank or out-of-range array survives the constructor
        and fails much later inside `ElementGeometry` or a scatter, with an
        opaque shape error far from the call that introduced it. `Mesh` is the
        entry point for user data (`Mesh.load`, hand-built meshes), so this is
        where a clear message pays off.
        '''
        if self.vertices.ndim != 2:
            raise ValueError(
                'vertices must be a 2D (n_vertices, spatial_dim) array, '
                f'got shape {self.vertices.shape}'
            )
        if self.elements.ndim != 2:
            raise ValueError(
                'elements must be a 2D (n_elements, n_nodes) array, '
                f'got shape {self.elements.shape}'
            )
        n_nodes = self.elements.shape[1]
        if n_nodes not in _SIMPLEX_NODE_COUNTS:
            raise NotImplementedError(
                'elements must be linear simplices with 2, 3, or 4 nodes '
                f'(a line, triangle, or tet), got {n_nodes}-node elements'
            )
        n_vertices = len(self.vertices)
        self._check_indices_in_range(self.elements, n_vertices, 'element')
        if self.boundary.size:
            if self.boundary.ndim != 2:
                raise ValueError(
                    'boundary must be a 2D (n_facets, n_nodes) array, '
                    f'got shape {self.boundary.shape}'
                )
            if self.boundary.shape[1] != n_nodes - 1:
                raise ValueError(
                    f'a boundary facet of a {n_nodes}-node element has '
                    f'{n_nodes - 1} nodes, got {self.boundary.shape[1]}'
                )
            self._check_indices_in_range(self.boundary, n_vertices, 'boundary')

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

    def _validate_tags(self) -> None:
        '''Reject tag arrays that do not line up with the elements or facets they tag.'''
        if self.cell_tags is not None and len(self.cell_tags) != len(self.elements):
            raise ValueError(
                f'cell_tags has {len(self.cell_tags)} entries but the mesh has '
                f'{len(self.elements)} elements'
            )
        if self.facet_tags is not None and len(self.facet_tags) != len(self.boundary):
            raise ValueError(
                f'facet_tags has {len(self.facet_tags)} entries but the mesh has '
                f'{len(self.boundary)} boundary facets'
            )

    @property
    def spatial_dim(self) -> int:
        '''Dimension of the space the nodes live in.

        Distinct from an element's `reference_dim`: a triangle mesh embedded in
        3D has spatial_dim 3 but reference_dim 2. The two coincide only when the
        elements fill their ambient space, which is why one number has served
        for both so far.
        '''
        return int(self.vertices.shape[1])

    def save(self, path: str = 'test_mesh.json') -> None:
        '''Write the mesh. `.json` is the native format; any other suffix
        (`.vtu`, `.msh`, `.obj`, ...) is written through meshio.'''
        from fem.io import save_mesh
        save_mesh(self, path)

    @classmethod
    def load(cls, path: str = 'test_mesh.json') -> 'Mesh':
        '''Read a mesh. `.json` is the native format; any other suffix
        (`.msh`, `.vtu`, ...) is read through meshio.'''
        from fem.io import load_mesh
        return load_mesh(path)

    def __repr__(self) -> str:
        return f'Mesh(vertices={self.vertices}, elements={self.elements}, boundary={self.boundary})'

    def with_topology(
        self,
        vertices: Vertices,
        elements: Elements,
        boundary: Elements,
        boundary_curves: Sequence[Curve | None] | None = None,
    ) -> 'Mesh':
        '''A new mesh over the given topology.

        The seam remeshers build through, so that refinement and coarsening name
        what they are doing rather than reaching for the constructor. A remesher that
        keeps its boundary on the same curves passes the (remapped) `boundary_curves`;
        omitting them yields a straight-sided mesh, since the new facets are otherwise
        unrelated to the old ones' curves.
        '''
        return Mesh(vertices, elements, boundary, boundary_curves)

    def copy(self) -> 'Mesh':
        # Same topology, so the per-facet curve association and the tags carry
        # unchanged. `with_topology` builds a different topology and so does not.
        curves = list(self.boundary_curves) if self.boundary_curves is not None else None
        return Mesh(
            self.vertices.copy(), self.elements.copy(), self.boundary.copy(), curves,
            cell_tags=None if self.cell_tags is None else self.cell_tags.copy(),
            facet_tags=None if self.facet_tags is None else self.facet_tags.copy(),
            tag_names=dict(self.tag_names),
        )

    # -- physical-group tags -------------------------------------------------

    def _resolve_tag(self, tag: int | str, tags: IntArray | None, kind: str) -> int:
        '''The integer id for `tag` (an id or a name), checked to actually tag a `kind`.'''
        if tags is None:
            raise ValueError(f'mesh has no {kind} tags')
        if isinstance(tag, str):
            by_name = {name: tid for tid, name in self.tag_names.items()}
            if tag not in by_name:
                raise ValueError(f'unknown tag name {tag!r}; known names: {sorted(by_name)}')
            tag = by_name[tag]
        tag = int(tag)
        if tag not in np.unique(tags):
            named = f' ({self.tag_names[tag]!r})' if tag in self.tag_names else ''
            raise ValueError(f'no {kind} carries tag {tag}{named}')
        return tag

    def facets_with_tag(self, tag: int | str) -> IntArray:
        '''Indices into `boundary` of the facets carrying `tag` (an id or a name).'''
        return np.flatnonzero(self.facet_tags == self._resolve_tag(tag, self.facet_tags, 'facet'))

    def cells_with_tag(self, tag: int | str) -> IntArray:
        '''Indices into `elements` of the cells carrying `tag` (an id or a name).'''
        return np.flatnonzero(self.cell_tags == self._resolve_tag(tag, self.cell_tags, 'cell'))

    def on_tag(self, tag: int | str, atol: float | None = None):
        '''A geometric region selecting the boundary nodes on the facets tagged `tag`.

        `tag` is a physical-group id or its name (e.g. `"inlet"`). The region tests
        whether a node lies on one of those facets, so it picks up the P2 edge-midpoint
        nodes on them and survives refinement, the same as the coordinate regions in
        `fem.regions`. Use it wherever a region is expected:
        `bc.add("dirichlet", mesh.on_tag("inlet"), 0.0)`.
        '''
        from fem.geometry import nodes_on_facets
        from fem.regions import DEFAULT_ATOL

        tol = DEFAULT_ATOL if atol is None else atol
        facet_coords = self.vertices[self.boundary[self.facets_with_tag(tag)]]
        return lambda points: nodes_on_facets(points, facet_coords, tol)

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
        node_pairs = _edge_node_pairs(self.elements.shape[1])
        n_pairs = len(node_pairs)
        edge_rows = np.sort(self.elements[:, node_pairs].reshape(-1, 2), axis=1)
        owners = np.repeat(np.arange(len(self.elements)), n_pairs)
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
    def element_diameters(self) -> FloatArray:
        '''Maximum edge length per element: the h_K in error estimates.'''
        pairs = _edge_node_pairs(self.elements.shape[1])
        corners = self.vertices[self.elements]                        # (n_el, n_nodes, dim)
        edge_vecs = corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]]  # (n_el, n_pairs, dim)
        return np.linalg.norm(edge_vecs, axis=2).max(axis=1)

    @cached_property
    def element_neighbours(self) -> list[list[int]]:
        '''For each element, the indices of elements sharing at least one edge.'''
        neighbours: list[set[int]] = [set() for _ in range(len(self.elements))]
        for elements in self.edge_to_elements.values():
            if len(elements) == 2:
                a, b = elements
                neighbours[a].add(b)
                neighbours[b].add(a)
        return [sorted(s) for s in neighbours]

    @cached_property
    def edges(self) -> IntArray:
        '''Every edge in the mesh, as sorted (v0, v1) index pairs.

        For a linear simplex the edge set is exactly every pair of its nodes:
        1 pair for a line, 3 for a triangle, 6 for a tet. That makes this
        dimension-general without a per-shape table; it holds only for linear
        simplices, which the constructor guarantees (quadratic elements carry
        midside nodes, so pairing every node would invent edges that don't
        exist).
        '''
        return self._edge_table[0]

