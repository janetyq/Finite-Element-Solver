import numpy as np
from utils.matrices import *

# Base solver for scalar PDEs, like Poisson, Heat, Wave
# TODO: generalize to vector PDEs

class BaseSolverResult:
    def __init__(self, u_values):
        self.u_values = u_values

class BaseSolver:
    def __init__(self, points, faces, boundary, matrices=None):
        self.points = points
        self.faces = faces
        self.boundary = boundary

        self.N = len(points)
        self.boundary_idxs = list(set(boundary.ravel()))
        self.inner_idxs = list(set(range(self.N)) - set(self.boundary_idxs))
        self.result = None

        if matrices is None:
            self.initialize_matrices()
        else:
            self.M, self.K = matrices
    
    def initialize_matrices(self):
        self.M = assemble_matrix(self.points, self.faces, calculate_element_mass_matrix)
        self.K = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix)
