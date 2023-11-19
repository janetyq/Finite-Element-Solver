import numpy as np

class BoundaryConditions:
    def __init__(self, points, boundary, boundary_idxs):
        self.points = points
        self.boundary = boundary
        self.boundary_idxs = list(set(boundary.ravel()))

        self.convenient_indices = self.calculate_convenient_indices()

        self.neumann_bc = None
        self.dirichlet_bc = None
        self.robin_bc = None
    
    def set_dirichlet_bc(self, indices, values):
        '''
        indices: list of node indices
        values: list of values at each node
        (can take any indices)
        '''
        self.dirichlet_bc = np.array(indices), values
    
    def set_neumann_bc(self, indices, values, default=[0, 0]):
        '''
        indices: list of node indices
        values: list of values at each node
        (only takes boundary indices)
        '''
        mask = np.where(np.any(np.isin(self.boundary, indices), axis=1))
        neumann_boundary = self.boundary[mask]
        neumann_values = []
        for boundary in neumann_boundary:
            value = []
            for b in boundary:
                if b in indices:
                    value.append(values[indices.index(b)])
                else:
                    value.append(default.copy())
            neumann_values.append(np.array(value))
        self.neumann_bc = neumann_boundary, neumann_values

    def set_dirichlet_bc_func(self, func):
        '''
        func: function that takes in a point and returns a displacement
        (only acts on boundary indices)
        '''
        dirichlet_values = []
        for idx in self.boundary_idxs:
            dirichlet_values.append(func(self.points[idx]))
        
        self.dirichlet_bc = np.array(self.boundary_idxs), dirichlet_values

    def set_neumann_bc_func(self, func):
        '''
        func: function that takes in a point and returns a force
        (only acts on boundary indices)
        '''
        neumann_boundary = []
        neumann_values = []
        for boundary in self.boundary:
            neumann_boundary.append(boundary)
            value = np.apply_along_axis(func, axis=1, arr=self.points[boundary]) # 2x2 array, 2 nodes 2 directions, #TODO: inefficient
            neumann_values.append(value)
        self.neumann_bc = neumann_boundary, neumann_values

    def set_robin_bc_func(self, func):
        robin_indices = []
        robin_values = []
        for idx in self.boundary_idxs:
            robin_indices.append(idx)
            robin_values.append(func(self.points[idx]))
        self.robin_bc = robin_indices, robin_values

    def calculate_convenient_indices(self):
        return ConvenientIndices(self.points, self.boundary, self.boundary_idxs)

class ConvenientIndices:
    def __init__(self, points, boundary, boundary_idxs):
        w, h = np.max(points[:, 0]), np.max(points[:, 1])
        
        self.inner_idxs = list(set(range(len(points))) - set(boundary_idxs))
        self.top_idxs = [idx for idx in boundary_idxs if points[idx][1] > h-1e-6]
        self.bottom_idxs = [idx for idx in boundary_idxs if points[idx][1] < 1e-6]
        self.left_idxs = [idx for idx in boundary_idxs if points[idx][0] < 1e-6]
        self.right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
        self.top_boundary = [boundary for boundary in boundary if np.all(np.isin(boundary, self.top_idxs))]
        self.bottom_boundary = [boundary for boundary in boundary if np.all(np.isin(boundary, self.bottom_idxs))]
        self.left_boundary = [boundary for boundary in boundary if np.all(np.isin(boundary, self.left_idxs))]
        self.right_boundary = [boundary for boundary in boundary if np.all(np.isin(boundary, self.right_idxs))]

    