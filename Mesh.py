import json
import numpy as np
from matplotlib.tri import Triangulation

from utils.helper import *

class Mesh:
    def __init__(self, vertices, elements, boundary):
        self.vertices = np.array(vertices)
        self.elements = np.array(elements) # list of indices of vertices (num_elements, 3)
        self.boundary = np.array(boundary) # list of indices of vertices (num_boundary, 2)
        self.boundary_idxs = list(set(self.boundary.ravel()))
        self.edges = self._get_all_edges()

    def plot(self, values=None):
        fig, ax = plt.subplots()
        triangulation = Triangulation(self.vertices[:, 0], self.vertices[:, 1], self.elements)
        if values is not None:
            tripcolor = ax.tripcolor(triangulation, values)
            fig.colorbar(tripcolor)
        ax.triplot(triangulation, 'k-', linewidth=0.5)
        ax.set_aspect('equal')
        plt.show()

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

def create_rect_mesh(corners, resolution):
    x_range = np.linspace(corners[0][0], corners[1][0], resolution[0])
    y_range = np.linspace(corners[0][1], corners[1][1], resolution[1])

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*resolution[0] + i

    for i in range(resolution[0]-1):
        for j in range(resolution[1]-1):
            if (i + j) % 2 == 0:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            else:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i, j+1)])
                elements.append([get_index(i+1, j), get_index(i+1, j+1), get_index(i, j+1)])

    boundary = get_boundary_from_vertices_elements(vertices, elements)
    mesh = Mesh(vertices, elements, boundary)

    return mesh

def create_approx_mesh(outline, approx_triangles=100):
    dx = np.sqrt(2 * calculate_polygon_area(outline) / approx_triangles)
    x_min, x_max = np.min(outline[:, 0]), np.max(outline[:, 0])
    y_min, y_max = np.min(outline[:, 1]), np.max(outline[:, 1])
    x_range = np.arange(x_min, x_max, dx)
    y_range = np.arange(y_min, y_max, dx)
    x_range += (x_max - x_range[-1])/2
    y_range += (y_max - y_range[-1])/2

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*len(x_range) + i

    # first mesh everything
    for i, x in enumerate(x_range[:-1]):
        for j, y in enumerate(y_range[:-1]):
            elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
            elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            
    # second remove elements with centers outside of outline
    removed_elements = []
    for element in elements:
        center = np.mean(vertices[element], axis=0)
        offcenters = [(center + vertices[i])/2 for i in element]
        for offcenter in offcenters:
            if not point_in_polygon(offcenter, outline):
                removed_elements.append(element)
                break
    for element in removed_elements:
        elements.remove(element)

    # remove unnecessary vertices
    used_v_idxs = np.unique(np.array(elements).flatten())
    # map old indices to new indices
    v_idx_map = {old: new for new, old in enumerate(used_v_idxs)}
    vertices = vertices[used_v_idxs]
    elements = [[v_idx_map[e_idx] for e_idx in element] for element in elements]
    boundary = get_boundary_from_vertices_elements(vertices, elements)
    mesh = Mesh(vertices, elements, boundary)

    fig, ax = plt.subplots()
    ax.plot(outline[:, 0], outline[:, 1], 'r-')
    Plotter(mesh, fig=fig, ax=ax, options={'title': 'Approximate mesh'}).plot_mesh(mode='wireframe')

    return mesh


if __name__ == '__main__':
    # create meshes
    # for resolution in [(20, 20), (40, 40), (80, 40), (100, 20)]:
    #     corners = [[0, 0], [resolution[0] // min(resolution), resolution[1] // min(resolution)]]
    #     mesh = create_rect_mesh(corners, resolution=resolution)
    #     mesh.save(f'meshes/{resolution[0]}x{resolution[1]}.json')
    #     mesh.plot()

    from Plotter import *
    corners = [[0, 0], [1, 1]]
    resolution = (40, 40)
    mesh = create_rect_mesh(corners, resolution=resolution)
    mesh.save(f'meshes/{resolution[0]}x{resolution[1]}.json')
    mesh.plot(np.random.rand(len(mesh.elements)))

    # outline = np.array([[0, 0], [1, 2], [3, 2], [2, 0], [0, 0]])
    # mesh = create_approx_mesh(outline, approx_triangles=500)
    # mesh.save('meshes/approx_mesh.pkl')

    # Mesh plotting examples with color
    element_list = []
    for e_idx, element in enumerate(mesh.elements):
        center = np.mean(mesh.vertices[element], axis=0)
        if center[0] > corners[1][0]/2:
            element_list.append(e_idx)
    color_elements = [('blue', element_list, 'right blue elements')]
    vert_list = []
    for vert_idx, vert in enumerate(mesh.vertices):
        if vert[0] < 1e-3:
            vert_list.append(vert_idx)
    color_vertices = [('red', vert_list, 'left red vertices')]
    Plotter(mesh, options={'title': 'Labeled mesh plot'}).plot_mesh(mode='wireframe', color_elements=color_elements, color_vertices=color_vertices)

    values = []
    for e_idx, element in enumerate(mesh.elements):
        x, y = np.mean(mesh.vertices[element], axis=0)
        value = abs(((y-0.5) - 0.3*np.sin(10*x)))
        values.append(value)
    Plotter(mesh, options={'title': 'Colorful value plot'}).plot_values(values, mode='colored')
    