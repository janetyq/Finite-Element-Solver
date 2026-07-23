import itertools
from collections.abc import Sequence
from functools import cached_property

import numpy as np

from fem.typing import ElementField, Elements, IntArray, VertexField, Vertices

Edge = tuple[int, int]

class Mesh:
    def __init__(
        self,
        vertices: Vertices | Sequence[Sequence[float]],
        elements: Elements | Sequence[Sequence[int]],
        boundary: Elements | Sequence[Sequence[int]],
    ) -> None:
        # TODO: assert the correct dimensions
        self.vertices: Vertices = np.array(vertices)
        self.elements: Elements = np.array(elements) # list of indices of vertices
        self.boundary: Elements = np.array(boundary) # list of indices of vertices
        self.boundary_idxs: IntArray = np.array(list(set(self.boundary.ravel())))
        self.edges: IntArray = self._get_all_edges()

    @property
    def spatial_dim(self) -> int:
        '''Dimension of the space the nodes live in.

        Distinct from an element's `reference_dim`: a triangle mesh embedded in
        3D has spatial_dim 3 but reference_dim 2. The two coincide only when the
        elements fill their ambient space, which is why one number has served
        for both so far.
        '''
        return int(self.vertices.shape[1])

    def convert_vertex_values_to_element_values(self, vertex_values: VertexField) -> ElementField:
        assert len(vertex_values) == len(self.vertices)
        element_values = np.zeros(len(self.elements))
        for e_idx, element in enumerate(self.elements):
            element_values[e_idx] = np.mean([vertex_values[v_idx] for v_idx in element])
        return element_values

    def convert_element_values_to_vertex_values(self, element_values: ElementField) -> VertexField:
        assert len(element_values) == len(self.elements)
        vertex_values = np.zeros(len(self.vertices))
        for e_idx, element in enumerate(self.elements):
            for v_idx in element:
                vertex_values[v_idx] = element_values[e_idx]
        return vertex_values

    # TODO: Save and load to better formats - off, obj
    def save(self, path: str = 'test_mesh.json') -> None:
        from fem.io import save_mesh
        save_mesh(self, path)

    @classmethod
    def load(cls, path: str = 'test_mesh.json') -> 'Mesh':
        from fem.io import load_mesh
        return load_mesh(path)

    def __repr__(self) -> str:
        return f'Mesh(vertices={self.vertices}, elements={self.elements}, boundary={self.boundary})'

    def with_topology(
        self,
        vertices: Vertices,
        elements: Elements,
        boundary: Elements,
    ) -> 'Mesh':
        '''A new mesh over the given topology.

        The seam remeshers build through, so that refinement and coarsening name
        what they are doing rather than reaching for the constructor.
        '''
        return Mesh(vertices, elements, boundary)

    def copy(self) -> 'Mesh':
        return self.with_topology(
            self.vertices.copy(), self.elements.copy(), self.boundary.copy()
        )

    @cached_property
    def edge_to_elements(self) -> dict[Edge, list[int]]:
        '''Map each sorted edge to the indices of elements that contain it.

        Interior edges map to exactly two elements; boundary edges to one.
        '''
        mapping: dict[Edge, list[int]] = {}
        for e_idx, element in enumerate(self.elements):
            for pair in itertools.combinations(sorted(element), 2):
                edge: Edge = pair  # type: ignore[assignment]
                mapping.setdefault(edge, []).append(e_idx)
        return mapping

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

    def _get_all_edges(self) -> IntArray:
        '''Every edge in the mesh, as sorted (v0, v1) index pairs.

        For a linear simplex the edge set is exactly every pair of its nodes:
        1 pair for a line, 3 for a triangle, 6 for a tet. That makes this
        dimension-general without a per-shape table -- but it holds *only* for
        linear simplices, hence the guard (quadratic elements carry midside
        nodes, so pairing every node would invent edges that don't exist).
        '''
        n_nodes = self.elements.shape[1]
        if n_nodes not in (2, 3, 4):
            raise NotImplementedError(
                f'edge extraction is only defined for linear simplices, '
                f'got {n_nodes}-node elements'
            )
        all_edges = {
            tuple(sorted(pair))
            for element in self.elements
            for pair in itertools.combinations(element, 2)
        }
        return np.array(sorted(all_edges))

