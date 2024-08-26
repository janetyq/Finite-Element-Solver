from Mesh import *
from Plotter import *
from Elements import *

class FEMesh(Mesh):
    '''
    2D triangular mesh
    '''
    def __init__(self, vertices, elements, boundary, element_type=LinearTriangleElement):
        Mesh.__init__(self, vertices, elements, boundary)

        self.element_type = element_type
        self.element_objs = self._get_element_objs()

        self.areas = [element.area for element in self.element_objs]

    def plot(self, fig=None, ax=None, options={}):
        return Plotter(self, fig=fig, ax=ax, options=options).plot_mesh()

    # METRICS
    def calculate_total_value(self, u):
        if len(u) == len(self.elements):       # u defined on elements
            return sum([self.areas[e_idx] * u[e_idx] for e_idx in range(len(self.elements))])
        elif len(u) == len(self.vertices):    # u defined on vertices
            return sum([self.areas[e_idx] * np.mean(u[self.elements[e_idx]]) for e_idx in range(len(self.elements))])

    def calculate_mean_value(self, u):
        areas = [self.areas[e_idx] for e_idx in range(len(self.elements))]
        return self.calculate_total_value(u) / sum(areas)

    def calculate_element_gradient(self, e_idx, u_element): # TODO: args suck, some code repetitive
        shape_gradient = self.element_objs[e_idx].gradient
        return shape_gradient.T @ u_element

    def calculate_gradient(self, u): # TODO: works, but need to understand 1D vs 2D use in dirichlet energy
        gradient = []
        for e_idx, elt in enumerate(self.elements):
            gradient.append(self.calculate_element_gradient(e_idx, u[elt]))
        return np.array(gradient)

    def calculate_dirichlet_energy(self, u):
        u_gradient = self.calculate_gradient(u)
        squared_gradient_norm = np.einsum('ij,ij->i', u_gradient, u_gradient)
        return sum([self.areas[e_idx] * squared_gradient_norm[e_idx] for e_idx in range(len(self.elements))])

    def calculate_energy(self, u, dudt):
        dirichlet_energy = self.calculate_dirichlet_energy(u)
        kinetic_energy = self.calculate_total_value(dudt**2)
        return dirichlet_energy + kinetic_energy

    def _get_element_objs(self):
        element_objs = []
        for e_idx, element in enumerate(self.elements):
            element_objs.append(self.element_type(self.vertices[element]))
        return element_objs

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
        