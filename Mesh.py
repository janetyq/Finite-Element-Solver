import json
import numpy as np
from matplotlib.tri import Triangulation

from utils.helper import *

class Mesh:
    def __init__(self, vertices, elements, boundary):
        # TODO: assert the correct dimensions, type-hinting
        self.vertices = np.array(vertices)
        self.elements = np.array(elements) # list of indices of vertices (num_elements, 3)
        self.boundary = np.array(boundary) # list of indices of vertices (num_boundary, 2)
        self.boundary_idxs = list(set(self.boundary.ravel()))
        self.edges = self._get_all_edges()

    def convert_vertex_values_to_element_values(self, vertex_values):
        assert len(vertex_values) == len(self.vertices)
        element_values = np.zeros(len(self.elements))
        for e_idx, element in enumerate(self.elements):
            element_values[e_idx] = np.mean([vertex_values[v_idx] for v_idx in element])
        return element_values

    def convert_element_values_to_vertex_values(self, element_values):
        assert len(element_values) == len(self.elements)
        vertex_values = np.zeros(len(self.vertices))
        for e_idx, element in enumerate(self.elements):
            for v_idx in element:
                vertex_values[v_idx] = element_values[e_idx]
        return vertex_values

    def plot(self):
        plotter = Plotter(title='Mesh plot')
        plotter.plot(self, mode='mesh')
        plotter.show()

    # TODO: Save and load to better formats - off, obj
    def save(self, path='test_mesh.json'):
        with open(path, 'w') as f:
            vertices = self.vertices.tolist()
            elements = self.elements.tolist()
            boundary = self.boundary.tolist()
            json.dump({'vertices': vertices, 'elements': elements, 'boundary': boundary}, f)
        print(f'Saved mesh to {path}')

    @classmethod
    def load(cls, path='test_mesh.json'):
        with open(path, 'r') as f:
            data = json.load(f)
            return cls(data['vertices'], data['elements'], data['boundary'])

    def __repr__(self):
        return f'Mesh(vertices={self.vertices}, elements={self.elements}, boundary={self.boundary})'

    def copy(self):
        return Mesh(self.vertices.copy(), self.elements.copy(), self.boundary.copy())

    def _get_all_edges(self):
        all_edges = set()
        for element in self.elements:
            for i in range(3):
                edge = [element[i], element[(i+1)%3]]
                all_edges.add(tuple(sorted(edge)))
        all_edges = np.array(list(all_edges))
        return all_edges

