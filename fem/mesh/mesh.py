import itertools
from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np

from fem.plot.plotter import Plotter, PlotMode
from fem.typing import ElementField, Elements, IntArray, VertexField, Vertices

MeshT = TypeVar('MeshT', bound='Mesh')


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

    def plot(self) -> None:
        plotter = Plotter(title='Mesh plot')
        plotter.plot(self, mode=PlotMode.MESH)
        plotter.show()

    # TODO: Save and load to better formats - off, obj
    def save(self, path: str = 'test_mesh.json') -> None:
        from fem.io import save_mesh
        save_mesh(self, path)

    @classmethod
    def load(cls: type[MeshT], path: str = 'test_mesh.json') -> MeshT:
        from fem.io import load_mesh
        # load_mesh instantiates `cls` itself, but its signature can only promise
        # a Mesh back.
        return cast(MeshT, load_mesh(path, cls=cls))

    def __repr__(self) -> str:
        return f'Mesh(vertices={self.vertices}, elements={self.elements}, boundary={self.boundary})'

    def with_topology(
        self: MeshT,
        vertices: Vertices,
        elements: Elements,
        boundary: Elements,
    ) -> MeshT:
        '''A new mesh of this same concrete type over the given topology.

        Remeshers (refinement, coarsening) have to hand back something the caller
        can keep using: an FEMesh must come back an FEMesh, carrying its element
        type and assembled matrices, rather than silently degrading to a bare Mesh.
        '''
        return type(self)(vertices, elements, boundary)

    def copy(self: MeshT) -> MeshT:
        return self.with_topology(
            self.vertices.copy(), self.elements.copy(), self.boundary.copy()
        )

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

