from enum import Enum

import numpy as np
from fem.plot.plotter import Plotter, PlotMode

# supports dirichlet, neumann, and mixed boundary conditions
# add boundary conditions with list of indices and values at indices


class BCType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class BoundaryConditions:
    def __init__(self, mesh):
        self.mesh = mesh

        self.neumann = {}
        self.dirichlet = {}
        self.force = {}

    def add(self, bc_type, indices, values):
        bc_type = BCType(bc_type)  # accepts BCType or its value; unknown raises ValueError
        values = np.array(values)
        target = self.dirichlet if bc_type is BCType.DIRICHLET else self.neumann
        if len(indices) == len(values):
            for v_idx, value in zip(indices, values):
                target[v_idx] = value
        else:
            for v_idx in indices:  # for elastic problems: neumann value = stress
                target[v_idx] = values

    def add_force(self, load_func):
        assert len(self.force) == 0, 'load already defined'
        for v_idx in range(len(self.mesh.vertices)):
            self.force[v_idx] = np.array(load_func(self.mesh.vertices[v_idx]))

    def check(self):
        '''Validate boundary conditions, catching two otherwise-silent footguns:

        1. Dirichlet/Neumann conditions must live on boundary vertices.
        2. No vertex may carry both a Dirichlet and a Neumann condition (a fixed
           node ignores its Neumann load, so the pairing is ambiguous).
        '''
        boundary = set(self.mesh.boundary_idxs)
        interior = (set(self.dirichlet) | set(self.neumann)) - boundary
        if interior:
            raise ValueError(f'Boundary conditions on non-boundary vertices: {sorted(interior)}')
        overlap = set(self.dirichlet) & set(self.neumann)
        if overlap:
            raise ValueError(f'Vertices carry both Dirichlet and Neumann conditions: {sorted(overlap)}')

    def do(self, N, dim):
        self.check()
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
        plotter.plot(self.mesh, bc=self, mode=PlotMode.BC)
        plotter.show()