import numpy as np
from utils.matrices import *
from utils.mesh import *
from refinement import *
from utils.half_edge import *

class BaseSolverResult:
    def __init__(self, u_values):
        self.u_values = u_values

class BaseSolver:
    def __init__(self, mesh, dim=1):
        self.dim = dim
        self.update_mesh(mesh)
        self.boundary_conditions = None
        self.load_function = None
        self.result = None

        self.refinement_mesh = RefinementMesh(mesh)

    def update_mesh(self, mesh):
        self.mesh = mesh.copy()
        self.he_mesh = HalfEdgeMesh(self.mesh) # for efficient neighbor queries - for topopt smoothing

        self.points = self.mesh.points
        self.faces = self.mesh.faces
        self.boundary = self.mesh.boundary

        self.N = len(self.points)
        self.boundary_idxs = list(set(self.boundary.ravel()))

        self.M = assemble_matrix(self.points, self.faces, calculate_element_mass_matrix, dim=self.dim)
        if self.dim == 1: # TODO: find a way to generalize to linear elastics too
            self.K = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix, dim=self.dim)

    def initialize(self, boundary_conditions, load_function=None, u_initial=None): # TODO: move u_initial?
        boundary_conditions.check()
        self.boundary_conditions = boundary_conditions
        self.load_function = load_function

        # load vector
        if load_function is not None:
            self.b = assemble_vector(self.points, self.faces, calculate_element_load_vector, load_function, dim=self.dim)
        else:
            self.b = np.zeros(self.dim*self.N)

        # neumann boundary conditions
        self.r = np.zeros(self.dim*self.N)
        for neumann_idxs, neumann_values in self.boundary_conditions.neumann:
            def neumann_func(point):
                point_idx = np.where(np.all(self.points == point, axis=1))[0][0]
                if point_idx not in neumann_idxs:
                    return 0 if self.dim == 1 else [0] * self.dim
                else:
                    return neumann_values[np.where(neumann_idxs == point_idx)[0][0]]
            self.r += assemble_vector(self.points, self.boundary, calculate_element_boundary_load_vector, neumann_func, dim=self.dim)

        # dirichlet boundary conditions
        fixed_idxs, fixed_displacements = [], []
        for dirichlet_idxs, dirichlet_values in self.boundary_conditions.dirichlet:
            fixed_idxs.extend(dirichlet_idxs)
            fixed_displacements.extend(dirichlet_values)
        fixed_idxs = np.array(fixed_idxs)

        self.fixed = np.array([self.dim*fixed_idxs + i for i in range(self.dim)], dtype=np.int32).T.flatten()
        self.free = list(set(range(self.dim*self.N)) - set(self.fixed))
        self.fixed_values = np.array(fixed_displacements).flatten()

        self.u = np.zeros(self.dim * self.N) if u_initial is None else u_initial
        self.u[self.fixed] = self.fixed_values


    def solve(self):
        raise NotImplementedError

    def adaptive_refinement(self, max_triangles=1000, max_iters=20):
        while len(self.faces) < max_triangles or max_iters == 0:
            self.initialize(self.boundary_conditions, self.load_function)
            print('num faces', len(self.faces))
            self.solve()
            face_residuals = self.calculate_face_residuals()
            max_residual = max(face_residuals)
            refine_idxs = []
            for face_idx, residual in enumerate(face_residuals):
                if residual > 0.9 * max_residual:
                    refine_idxs.append(face_idx)

            print('refining', len(refine_idxs), 'faces')
            self.refinement_mesh.refine_triangles(refine_idxs)
            self.update_mesh(self.refinement_mesh.get_mesh())
            max_iters -= 1
        
        self.initialize(self.boundary_conditions, self.load_function)
        self.solve()
