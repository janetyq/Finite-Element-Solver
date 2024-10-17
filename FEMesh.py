from Mesh import *
from Plotter import *
from Elements import *

class FEMesh(Mesh):
    '''
    Built on top of Mesh class
    Adds functionality for FEM calculations (ie. element areas, gradients, integrals)
    '''
    def __init__(self, vertices, elements, boundary, element_type=LinearTriangleElement):
        Mesh.__init__(self, vertices, elements, boundary)

        self.element_type = element_type
        self.boundary_type = element_type.SUB_TYPE
        self.element_objs = [self.element_type(self.vertices[element]) for element in self.elements]
        self.boundary_objs = [self.boundary_type(self.vertices[boundary]) for boundary in self.boundary]

        self.prepare_matrices()

    def prepare_matrices(self, dim=1, **kwargs):
        # Can be called to reprepare matrices for different dim
        self.dim = dim
        self.M = self.assemble_matrix('mass', 'element', dim, **kwargs)
        self.M_b = self.assemble_matrix('mass', 'boundary', dim, **kwargs)
        self.K = self.assemble_matrix('stiffness', 'element', dim, **kwargs)
        if dim == 1:
            self.K_b = self.assemble_matrix('stiffness', 'boundary', dim, **kwargs)
        # TODO: not assembling K_b for dim=2

    def assemble_matrix(self, matrix_type_name, element_type_name, dim=1, **kwargs):
        # TODO: term "element" is overloaded here, and its a bit hacky

        if element_type_name == 'element':
            elements, element_objs = self.elements, self.element_objs
        elif element_type_name == 'boundary':
            elements, element_objs = self.boundary, self.boundary_objs

        matrix_calculators = {
            'mass': lambda e_idx: element_objs[e_idx].calculate_mass_matrix(dim, idx=e_idx, **kwargs),
            'stiffness': lambda e_idx: element_objs[e_idx].calculate_stiffness_matrix(dim, idx=e_idx, **kwargs),
        }

        N = len(self.vertices)
        A = np.zeros((dim * N, dim * N))
        for e_idx, element in enumerate(elements):
            idxs = np.array([dim*element + i for i in range(dim)]).T.flatten()
            element_matrix = matrix_calculators[matrix_type_name](e_idx)
            A[np.ix_(idxs, idxs)] += element_matrix
        return A

    # METRICS
    def calculate_total_value(self, u):
        if len(u) == len(self.elements):       # u defined on elements
            return sum([self.element_objs[e_idx].volume * u[e_idx] for e_idx in range(len(self.elements))])
        elif len(u) == len(self.vertices):    # u defined on vertices
            return sum([self.element_objs[e_idx].volume * np.mean(u[self.elements[e_idx]]) for e_idx in range(len(self.elements))])

    def calculate_mean_value(self, u):
        return self.calculate_total_value(u) / sum([element.volume for element in self.element_objs])

    def calculate_gradient(self, u): # TODO: works, but need to understand 1D vs 2D use in dirichlet energy
        gradient = []
        for e_idx, element_obj in enumerate(self.element_objs):
            # u_elt = u[np.array([2*self.elements[e_idx], 2*self.elements[e_idx]+1]).T.flatten()]
            u_elt = u[self.elements[e_idx]]
            gradient.append(element_obj.grad_phi.T @ u_elt)
        return np.array(gradient)

    def calculate_dirichlet_energy(self, u):
        u_gradient = self.calculate_gradient(u)
        squared_gradient_norm = np.einsum('ij,ij->i', u_gradient, u_gradient)
        return sum([self.element_objs[e_idx].volume * squared_gradient_norm[e_idx] for e_idx in range(len(self.elements))])

    def calculate_energy(self, u, dudt):
        dirichlet_energy = self.calculate_dirichlet_energy(u)
        kinetic_energy = self.calculate_total_value(dudt**2)
        return dirichlet_energy + kinetic_energy

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

    def copy(self):
        return FEMesh(self.vertices.copy(), self.elements.copy(), self.boundary.copy(), element_type=self.element_type)