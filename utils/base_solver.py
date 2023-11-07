import numpy as np
from utils.matrices import *

class BaseSolverResult:
    def __init__(self, u_values):
        self.u_values = u_values

class BaseSolver:
    def __init__(self, points, faces, boundary):
        self.points = points
        self.faces = faces
        self.boundary = boundary

        self.initialize_matrices()
        self.N = len(points)
        self.boundary_idxs = list(set(boundary.ravel()))
        self.inner_idxs = list(set(range(self.N)) - set(self.boundary_idxs))

        self.result = None
    
    def initialize_matrices(self):
        self.A = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix)
        self.M = assemble_matrix(self.points, self.faces, calculate_element_mass_matrix)
