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
                for v_idx, value in zip(indices, values):
                    self.dirichlet[v_idx] = value
            else:
                for v_idx in indices:
                    self.dirichlet[v_idx] = values
        elif bc_type == 'neumann':
            if len(indices) == len(values):
                for v_idx, value in zip(indices, values):
                    self.neumann[v_idx] = value # TODO: check support for list of values
            else:
                for v_idx in indices: # for elastics problems: neumann value = stress
                    self.neumann[v_idx] = values
        else:
            raise ValueError(f'bc_type {bc_type} not recognized')

    def add_force(self, load_func):
        assert len(self.force) == 0, 'load already defined'
        for v_idx in range(len(self.mesh.vertices)):
            self.force[v_idx] = np.array(load_func(self.mesh.vertices[v_idx]))

    def check(self):
        # TODO:
        # check that max one BC per node
        # check that BC defined only on boundary
        pass

    def do(self, N, dim):
        # dirichlet boundary conditions
        self.fixed_idxs = [dim*v_idx + d  for v_idx in self.dirichlet.keys() for d in range(dim)]
        self.fixed_values = list(np.array(list(self.dirichlet.values())).flatten())
        self.free_idxs = [dim*v_idx + d  for v_idx in list(set(range(N)) - set(self.dirichlet.keys())) for d in range(dim)]

        self.neumann_load = []
        self.force_load = []
        for v_idx in range(len(self.mesh.vertices)):
            self.neumann_load.append(self.neumann[v_idx] if v_idx in self.neumann else np.zeros(dim))
            self.force_load.append(self.force[v_idx] if v_idx in self.force else np.zeros(dim))
        self.neumann_load = np.array(self.neumann_load)
        self.force_load = np.array(self.force_load)  

    def plot(self):
        plotter = Plotter(title='Boundary conditions')
        plotter.plot(self.mesh, bc=self, mode='bc')
        plotter.show()