import numpy as np

# supports dirichlet, neumann, and mixed boundary conditions

class BoundaryConditions:
    def __init__(self, mesh):
        self.points = mesh.points
        self.boundary = mesh.boundary
        self.boundary_idxs = list(set(mesh.boundary.ravel()))

        self.neumann = []
        self.dirichlet = []

    def add(self, bc_type, indices, values):
        assert len(indices) == len(values)
        if bc_type == 'dirichlet':
            self.dirichlet.append((np.array(indices), values))
        elif bc_type == 'neumann':
            self.neumann.append((np.array(indices), values))
        else:
            raise ValueError(f'bc_type {bc_type} not recognized')
    
    def check(self):
        all_indices = set()

        # check that max one BC per node
        for indices, values in self.dirichlet:
            assert set(indices).isdisjoint(all_indices)
            all_indices.update(indices)

        for indices, values in self.neumann:
            assert set(indices).isdisjoint(all_indices)
            all_indices.update(indices)

        # check that BC defined only on boundary
        assert set(all_indices).issubset(set(self.boundary_idxs))

        

