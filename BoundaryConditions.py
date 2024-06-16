import numpy as np
from utils.matrices import *

# supports dirichlet, neumann, and mixed boundary conditions
# add boundary conditions with list of indices and values at indices

class BoundaryConditions:
    def __init__(self, mesh):
        self.mesh = mesh

        self.neumann = {}
        self.dirichlet = {}
        # self.convective = {}

    def add(self, bc_type, indices, values):
        if bc_type == 'dirichlet':
            if len(indices) == len(values):
                for idx, value in zip(indices, values):
                    self.dirichlet[idx] = value
            else:
                for idx in indices:
                    self.dirichlet[idx] = values
        elif bc_type == 'neumann':
            if len(indices) == len(values):
                for idx, value in zip(indices, values):
                    self.neumann[idx] = value
            else:
                for idx in indices:
                    self.neumann[idx] = values
        elif bc_type == 'stress':
            for idx in indices:
                self.neumann[idx] = np.array(values)/2
        else:
            raise ValueError(f'bc_type {bc_type} not recognized')
    
    def check(self):
        # check that max one BC per node
        # check that BC defined only on boundary
        pass

    def do(self, N, dim=1):
        # dirichlet boundary conditions
        self.fixed_idxs = list(self.dirichlet.keys())
        self.fixed_values = list(self.dirichlet.values())
        self.free_idxs = list(set(range(N)) - set(self.fixed_idxs))

        if dim == 2:
            self.fixed_idxs = list(np.array([[2*idx, 2*idx+1] for idx in self.fixed_idxs]).flatten())
            self.free_idxs = list(np.array([[2*idx, 2*idx+1] for idx in self.free_idxs]).flatten())
            self.fixed_values = list(np.array(self.fixed_values).flatten())
        
        # neumann boundary conditions
        def neumann_func(point):
            point_idx = np.where(np.all(self.mesh.points == point, axis=1))[0][0]
            if point_idx not in self.neumann:
                return 0 if dim == 1 else [0] * dim
            else:
                return self.neumann[point_idx]
        self.neumann_load = assemble_vector(self.mesh.points, self.mesh.boundary, calculate_element_boundary_load_vector, neumann_func, dim=dim)
        

