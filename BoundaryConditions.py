import numpy as np
import matplotlib.pyplot as plt
from Plotter import *

# supports dirichlet, neumann, and mixed boundary conditions
# add boundary conditions with list of indices and values at indices

class BoundaryConditions:
    def __init__(self, mesh):
        self.mesh = mesh

        self.neumann = {}
        self.dirichlet = {}
        self.force = {}

    def add(self, bc_type, indices, values):
        values = np.array(values)
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
                    self.neumann[idx] = value # TODO: check support for list of values
            else:
                for idx in indices: # stress
                    self.neumann[idx] = values
        else:
            raise ValueError(f'bc_type {bc_type} not recognized')

    def add_force(self, load_func):
        assert len(self.force) == 0, 'load already defined'
        for idx in range(len(self.mesh.points)):
            self.force[idx] = np.array(load_func(self.mesh.points[idx]))

    def check(self):
        # TODO:
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

        self.neumann_load = []
        self.force_load = []
        for idx, point in enumerate(self.mesh.points):
            self.neumann_load.append(self.neumann[idx] if idx in self.neumann else np.zeros(dim))
            self.force_load.append(self.force[idx] if idx in self.force else np.zeros(dim))
        self.neumann_load = np.array(self.neumann_load)
        self.force_load = np.array(self.force_load)  

    def plot(self, values=None): # TODO: what is values, body force?
        return Plotter(self.mesh).plot_bc(self, values)      
