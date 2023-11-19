import numpy as np
from utils.matrices import *
from utils.mesh import *

import sys
sys.path.append('../')
from refinement import *

# Base solver for scalar PDEs, like Poisson, Heat, Wave
# TODO: generalize to vector PDEs

class BaseSolverResult:
    def __init__(self, u_values):
        self.u_values = u_values

class BaseSolver:
    def __init__(self, mesh, refinement_mesh=None):
        self.update_attributes(mesh)

        self.refinement_mesh = RefinementMesh(mesh)

    def update_attributes(self, mesh):
        self.mesh = mesh
        self.points = mesh.points
        self.faces = mesh.faces
        self.boundary = mesh.boundary

        self.N = len(self.points)
        self.boundary_idxs = list(set(self.boundary.ravel()))
        self.inner_idxs = list(set(range(self.N)) - set(self.boundary_idxs))
        self.result = None

        self.M = assemble_matrix(self.points, self.faces, calculate_element_mass_matrix)
        self.K = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix)

    def add_bc(self, bc): # TODO: where is this used
        self.bc = bc

    def adaptive_refinement(self, factor):
        face_residuals = self.calculate_face_residuals()
        min_residual = min(face_residuals)
        refine_idxs = []
        for face_idx, residual in enumerate(face_residuals):
            if residual > factor * min_residual:
                refine_idxs.append(face_idx)

        print('refining', len(refine_idxs), 'faces')
        self.refinement_mesh.refine_triangles(refine_idxs)
        self.update_attributes(self.refinement_mesh.get_mesh())
        
    # # implemented in specific solvers
    # @abstractmethod
    # def solve(self):
    #     pass

    # @abstractmethod
    # def plot_result(self):
    #     pass
