import pickle
import numpy as np
from utils.helper import *

from Plotter import *
from Elements import *

class Mesh:
    '''
    2D triangular mesh
    '''
    def __init__(self, vertices, elements, boundary):
        self.vertices = np.array(vertices)
        self.elements = np.array(elements)
        self.boundary = np.array(boundary)

        self.element_type = LinearTriangleElement
        self.areas = np.array([calculate_triangle_area(self.vertices[element]) for element in self.elements])
        self.shape_functions = self._get_all_shape_functions()

        self.boundary_idxs = list(set(self.boundary.ravel()))
        self.edges = self._get_all_edges()
        self.element_neighbors = self._calculate_element_neighbors(self.elements)

    # TODO: Save and load to better formats - off, obj
    @classmethod
    def load(cls, path='test_mesh.pkl'):
        with open(path, 'rb') as f:
            return cls(*pickle.load(f))

    def save(self, path='test_mesh.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.get_info(), f)
        print(f'Saved mesh to {path}')

    def get_info(self):
        return self.vertices, self.elements, self.boundary

    def __repr__(self):
        return f'Mesh(vertices={self.vertices}, elements={self.elements}, boundary={self.boundary})'

    def copy(self):
        return Mesh(self.vertices.copy(), self.elements.copy(), self.boundary.copy())

    def plot(self, fig=None, ax=None, options={}):
        return Plotter(self, fig=fig, ax=ax, options=options).plot_mesh()

    # METRICS
    def calculate_total_value(self, u):
        if len(u) == len(self.elements):       # u defined on elements
            return sum([self.areas[e_idx] * u[e_idx] for e_idx in range(len(self.elements))])
        elif len(u) == len(self.vertices):    # u defined on vertices
            return sum([self.areas[e_idx] * np.mean(u[self.elements[e_idx]]) for e_idx in range(len(self.elements))])

    def calculate_mean_value(self, u):
        return self.calculate_total_value(u) / sum(self.areas)

    def calculate_element_gradient(self, e_idx, u_element): # TODO: args suck, some code repetitive
        shape_gradient = self.shape_functions[e_idx].gradient
        return shape_gradient.T @ u_element

    def calculate_gradient(self, u): # TODO: works, but need to understand 1D vs 2D use in dirichlet energy
        gradient = []
        for e_idx, elt in enumerate(self.elements):
            gradient.append(self.calculate_element_gradient(e_idx, u[elt]))
        return np.array(gradient)

    def calculate_dirichlet_energy(self, u):
        u_gradient = self.calculate_gradient(u)
        return sum([self.areas[e_idx] * calculate_dot(u_gradient[e_idx], u_gradient[e_idx]) 
                    for e_idx in range(len(self.elements))])

    def calculate_energy(self, u, dudt):
        dirichlet_energy = self.calculate_dirichlet_energy(u)
        kinetic_energy = self.calculate_total_value(dudt**2)
        return dirichlet_energy + kinetic_energy
        
    def _calculate_element_neighbors(self, elements):
        element_neighbors = {e_idx: [] for e_idx in range(len(elements))}
        for e_idx, element in enumerate(elements):
            for neighbor_idx in element:
                element_neighbors[e_idx].extend(np.where(elements == neighbor_idx)[0])
            # keep only elements that appear twice
            element_neighbors[e_idx] = list(set([v_idx for v_idx in element_neighbors[e_idx] if element_neighbors[e_idx].count(v_idx) == 2]))
        return element_neighbors

    def _get_all_edges(self):
        all_edges = set()
        for element in self.elements:
            for i in range(3):
                edge = [element[i], element[(i+1)%3]]
                all_edges.add(tuple(sorted(edge)))
        all_edges = np.array(list(all_edges))
        return all_edges

    def _get_all_shape_functions(self):
        '''
        Calculates shape function for element e_idx
        N(x, y) = a + b*x + c*y
        '''
        shape_functions = []
        for e_idx, element in enumerate(self.elements):
            shape_functions.append(LinearTriangleElement(self.vertices[element]))
        return shape_functions

    def get_edges_in_idxs(self, vertices_idxs, exclude_corners=False):
        in_edges = []
        for edge in self.edges:
            if edge[0] in vertices_idxs and edge[1] in vertices_idxs:
                if exclude_corners:
                    x1, y1 = self.vertices[edge[0]]
                    x2, y2 = self.vertices[edge[1]]
                    if (x1 - x2) != 0 and (y1 - y2) != 0:
                        continue
                in_edges.append(edge)
        return in_edges

    def get_boundary_idxs_in_rect(self, rect):
        x_min, y_min, x_max, y_max = rect
        in_boundary_idxs = []
        for v_idx in self.boundary_idxs:
            x, y = self.vertices[v_idx]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                in_boundary_idxs.append(v_idx)
        return in_boundary_idxs

    @classmethod
    def create_approx_mesh(cls, outline, approx_triangles=100):
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

    @classmethod
    def create_rect_mesh(cls, corners, resolution):
        x_range = np.linspace(corners[0][0], corners[1][0], resolution[0])
        y_range = np.linspace(corners[0][1], corners[1][1], resolution[1])

        vertices = np.array([[x, y] for y in y_range for x in x_range])
        elements = []

        def get_index(i, j):
            return j*resolution[0] + i

        for i in range(resolution[0]-1):
            for j in range(resolution[1]-1):
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])

        boundary = get_boundary_from_vertices_elements(vertices, elements)
        mesh = Mesh(vertices, elements, boundary)

        # fig, ax = plt.subplots()
        # ax.plot([corners[0][0], corners[1][0], corners[1][0], corners[0][0], corners[0][0]], 
        #         [corners[0][1], corners[0][1], corners[1][1], corners[1][1], corners[0][1]], 'r-')
        # Plotter(mesh, fig=fig, ax=ax, options={'title': 'Rectangular mesh'}).plot_mesh(mode='wireframe')

        return mesh


if __name__ == '__main__':
    corners = [[0, 0], [1, 1]]
    resolution = (5, 5)
    mesh = Mesh.create_rect_mesh(corners, resolution=resolution)
    mesh.save(f'meshes/{resolution[0]}x{resolution[1]}.pkl')

    # outline = np.array([[0, 0], [1, 2], [3, 2], [2, 0], [0, 0]])
    # mesh = Mesh.create_approx_mesh(outline, approx_triangles=500)
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

    # Neighbors plotting example
    element_neighbors = mesh.calculate_element_neighbors(mesh.elements)
    fig, ax = plt.subplots()
    for e_idx, neighbors in element_neighbors.items():
        if e_idx % 2 == 0 or e_idx % 3 == 0:
            continue
        center = np.mean(mesh.vertices[mesh.elements[e_idx]], axis=0)
        random_color = np.random.rand(3)
        for neighbor_idx in neighbors:
            neighbor_center = np.mean(mesh.vertices[mesh.elements[neighbor_idx]], axis=0)
            ax.plot([center[0], neighbor_center[0]], [center[1], neighbor_center[1]], color=random_color)
    Plotter(mesh, fig=fig, ax=ax, options={'title': 'Face neighbors'}).plot_mesh(mode='wireframe')
    