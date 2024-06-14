import numpy as np
from utils.matrices import *
from utils.refinement import *
from BoundaryConditions import *
from Mesh import *
from Solution import *

class Equation:
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters
        self.dim = 2 if name == "linear_elastic" else 1

class Solver:
    def __init__(self, mesh, equation, boundary_conditions=None, load_function=None):
        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions(mesh)
        self.boundary_conditions.do(mesh.points.shape[0], dim=self.equation.dim)
        self.load_function = load_function if load_function is not None else lambda x: 0
        self.solution = Solution(mesh)

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.solution = Solution(mesh)

    def set_bc(self, bc):
        self.boundary_conditions = bc
        self.boundary_conditions.do(self.mesh.points.shape[0], dim=self.equation.dim)

    def preprocess(self):
        # do boundary conditions, in the future, do more matrix building
        self.M = assemble_matrix(self.mesh.points, self.mesh.faces, calculate_element_mass_matrix, dim=self.equation.dim)
        if self.equation.dim == 1:
            self.K = assemble_matrix(self.mesh.points, self.mesh.faces, calculate_element_stiffness_matrix, dim=self.equation.dim)
            # self.K = assemble_matrix(self.mesh.points, self.mesh.faces, calculate_element_stiffness_matrix, dim=self.equation.dim, func=self.equation.parameters['func'])
        self.b = assemble_vector(self.mesh.points, self.mesh.faces, calculate_element_load_vector, self.load_function, dim=self.equation.dim)
        self.r = self.boundary_conditions.neumann_load
        
        # create solution vector
        self.u = np.zeros(self.mesh.points.shape[0] * self.equation.dim)
        self.free = self.boundary_conditions.free_idxs
        self.fixed = self.boundary_conditions.fixed_idxs
        self.u[self.fixed] = self.boundary_conditions.fixed_values

    def solve(self):
        self.preprocess()
        self.solution.reset()
        
        equation_solvers = {
            "projection": self._solve_projection,
            "poisson": self._solve_poisson,
            "heat": self._solve_heat,
            "wave": self._solve_wave,
            "linear_elastic": self._solve_linear_elastic,
        }

        solver_method = equation_solvers.get(self.equation.name)
        if solver_method:
            solver_method()
        else:
            raise ValueError(f"Unknown equation name: {self.equation.name}")

    # # residuals
    # def calculate_residuals(self):
    #     # apriori and aposteriori error estimation

    #     equation_residuals = {
    #         "projection": self._calculate_projection_residuals,
    #         "poisson": self._calculate_poisson_residuals,
    #         "heat": self._calculate_heat_residuals,
    #         "wave": self._calculate_wave_residuals,
    #         "linear_elastic": self._calculate_linear_elastic_residuals,
    #     }

    #     residual_method = equation_residuals.get(self.equation.name)
    #     if residual_method:
    #         residual_method()
    #     else:
    #         raise ValueError(f"Unknown equation name: {self.equation.name}")

    # def _calculate_projection_residuals(self, apriori=True):
    #     # Apriori error ||e|| <= C * h^2 * ||f"||
    #     if apriori:
    #         # compute apriori residual
    #         residuals = np.zeros(len(self.mesh.faces))
    #         for face_idx, face in enumerate(self.mesh.faces):
    #             residuals[face_idx] = 0 # placeholder
    #         self.solution.set_values("apriori_residual", residuals)
    #     else:
    #         # compute aposteriori residual
    #         residuals = None
    #         self.solution.set_values("aposteriori_residual", residuals)
    #         pass

    def _solve_projection(self):
        print('Solving L2 projection...') # M @ u = b
        M_mod = self.M[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.M[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        self.u[self.free] = np.linalg.solve(M_mod, b_mod)
        self.solution.set_values("u", self.u)
    
    def _solve_poisson(self):
        print('Solving Poisson equation...') # K @ u = b
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        self.u[self.free] = np.linalg.solve(K_mod, b_mod)
        self.solution.set_values("u", self.u)
        # residuals = np.zeros(len(self.mesh.faces))
        # for face_idx, face in enumerate(self.mesh.faces):
        #     load = np.linalg.norm([self.load_function(point) for point in self.mesh.points[face]])
        #     residuals[face_idx] += load * self.mesh.areas[face_idx] / 3
        # self.solution.set_values("face_residuals", residuals)

    def _solve_heat(self):
        print('Solving heat equation...') # M @ u' + K @ u = b
        #  (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        M_mod = self.M[np.ix_(self.free, self.free)]
        K_mod = self.K[np.ix_(self.free, self.free)]
        self.u = self.equation.parameters['u_initial']
        self.u[self.fixed] = self.boundary_conditions.fixed_values
        dt, iters = self.equation.parameters['dt'], self.equation.parameters['iters']
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        t_values = [0]
        u_values = [self.u]
        print(f't = {t_values[0]:.3f}, mean temp = {self.mesh.calculate_mean_value(u_values[0]):.3f}')
        for i in range(iters):
            self.u[self.free] = np.linalg.solve(M_mod + K_mod * dt, M_mod @ self.u[self.free] + b_mod * dt)
            t_values.append(dt * (i+1))
            u_values.append(self.u.copy())
            print(f't = {t_values[-1]:.3f}, mean temp = {self.mesh.calculate_mean_value(u_values[-1]):.3f}')
        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)

    def _solve_wave(self):
        print('Solving wave equation...') # M @ u" + K @ u = b
        u, dudt = self.equation.parameters['u_initial'], self.equation.parameters['dudt_initial']
        c = self.equation.parameters['c']
        dt, iters = self.equation.parameters['dt'], self.equation.parameters['iters']
        x = np.block([u, dudt])

        # Crank-Nicolson method - average of forward and backward Euler
        A_left = np.block([[self.M, -dt/2 * self.M],
                        [c**2 * dt/2 * self.K, self.M]])
        A_right = np.block([[self.M, dt/2 * self.M],
                            [-c**2 * dt/2 * self.K, self.M]])
        b_right = np.block([np.zeros_like(self.b), dt/2 * (self.b + np.roll(self.b, -1))])

        N = len(self.mesh.points)
        t_values = [0]
        u_values = [x[:N]]
        dudt_values = [x[N:]]
        total_energy = self.mesh.calculate_energy(u_values[-1], dudt_values[-1])
        print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        for i in range(iters):
            x = np.linalg.solve(A_left, A_right @ x + b_right)
            t_values.append(dt * (i+1))
            u_values.append(x[:N])
            dudt_values.append(x[N:])
            total_energy = self.mesh.calculate_energy(u_values[-1], dudt_values[-1])
            print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)
        self.solution.set_values("dudt_values", dudt_values)

    def _solve_linear_elastic(self):
        u = np.zeros(2 * len(self.mesh.points))
        u[self.fixed] = self.boundary_conditions.fixed_values
        
        E = self.equation.parameters['E']
        E = np.full(len(self.mesh.faces), E)
        nu = self.equation.parameters['nu']
        nu = np.full(len(self.mesh.faces), nu)
        
        material_func = np.vstack([E, nu]).T
        K = assemble_matrix(self.mesh.points, self.mesh.faces, calculate_element_stiffness_matrix, func=material_func, dim=2)
        K_mod = K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - K[np.ix_(self.free, self.fixed)] @ u[self.fixed] + self.r[self.free]
        u[self.free] = np.linalg.solve(K_mod, b_mod)

        deformed_mesh = Mesh(self.mesh.points + u.reshape((-1, 2)), self.mesh.faces, self.mesh.boundary)
        self.solution.set_values("u", u)
        self.solution.set_values("deformed_mesh", deformed_mesh)

        self.B = np.array([calculate_B(element, (material_func, e_idx)) for e_idx, element in enumerate(self.mesh.points[self.mesh.faces])]) # gradient matrix
        self.D = np.array([calculate_D(element, (material_func, e_idx)) for e_idx, element in enumerate(self.mesh.points[self.mesh.faces])]) # elasticity matrix
        
        eps_faces = np.zeros((len(self.mesh.faces), 3))
        sigma_faces = np.zeros((len(self.mesh.faces), 3))

        for face_idx, face in enumerate(self.mesh.faces):
            element = self.mesh.points[face]
            e_idxs = np.array([2*face, 2*face+1]).T.flatten()
            eps = self.B[face_idx] @ u[e_idxs]
            sigma = self.D[face_idx] @ eps
            eps_faces[face_idx] = eps
            sigma_faces[face_idx] = sigma

        self.solution.set_values("strain", np.linalg.norm(eps_faces, axis=-1))
        self.solution.set_values("stress", np.linalg.norm(sigma_faces, axis=-1))
        
        C_total = 1/2 * u.T @ K @ u
        C_faces = np.zeros(len(self.mesh.faces))
        for face_idx, face in enumerate(self.mesh.faces):
            e_idxs = np.array([2*face, 2*face+1]).T.flatten()
            u_face = u[e_idxs]
            K_face = calculate_element_stiffness_matrix(self.mesh.points[face], (material_func, face_idx), dim=2)
            C_faces[face_idx] = 1/2 * u_face.T @ K_face @ u_face
        self.solution.set_values("compliance", C_faces)
        self.solution.set_values("total_compliance", C_total)

    def adaptive_refinement(self, max_triangles=1000, max_iters=20):
        # TODO: there's a bug somewhere
        if 'face_residuals' not in self.solution.values:
            raise ValueError('No face residuals found in solution')
        refinement_mesh = RefinementMesh(self.mesh)
        while len(self.mesh.faces) < max_triangles or max_iters == 0:
            face_residuals = self.solution.values['face_residuals']
            max_residual = max(face_residuals)
            refine_idxs = []
            for face_idx, residual in enumerate(face_residuals):
                if residual > 0.9 * max_residual:
                    refine_idxs.append(face_idx)

            refinement_mesh.refine_triangles(refine_idxs)
            self.mesh = refinement_mesh.get_mesh()
            max_iters -= 1



