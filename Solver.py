import numpy as np
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
    def __init__(self, mesh, equation, boundary_conditions=None):
        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions(mesh)
        self.solution = Solution(mesh)
        self.dim = self.equation.dim

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.solution = Solution(mesh)

    def set_bc(self, bc):
        self.boundary_conditions = bc
        self.boundary_conditions.do(self.mesh.points.shape[0], dim=self.equation.dim)

    def preprocess(self):
        # todo: do more matrix building
        self.boundary_conditions.do(self.mesh.points.shape[0], dim=self.dim)

        # assemble mass and stiffness matrices and load vector
        self.M, self.M_faces = self._assemble_matrix(self._calculate_element_mass_matrix)
        if self.equation.dim == 1: # TODO: implement 2D
            self.K, self.K_faces = self._assemble_matrix(self._calculate_element_stiffness_matrix)
        else: # dim = 2
            rho = self.equation.parameters.get('rho', 1)
            E_min, p = 1e-6, 3
            E = E_min + np.full(len(self.mesh.faces), self.equation.parameters['E']) * rho**p
            nu = np.full(len(self.mesh.faces), self.equation.parameters['nu']) * rho**p
            self.material_func = np.vstack([E, nu]).T 
            self.K, self.K_faces = self._assemble_matrix(self._calculate_element_stiffness_matrix, self.material_func)
        self.b = self._assemble_vector(self._calculate_element_load_vector, self.boundary_conditions.force_load)
        self.b += self._assemble_vector(self._calculate_element_boundary_load_vector, self.boundary_conditions.neumann_load)
        
        # create solution vector
        self.u = np.zeros(len(self.mesh.points) * self.dim)
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

        return self.solution

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
        b_mod = self.b[self.free] - self.M[np.ix_(self.free, self.fixed)] @ self.u[self.fixed]
        self.u[self.free] = np.linalg.solve(M_mod, b_mod)
        self.solution.set_values("u", self.u)
    
    def _solve_poisson(self):
        print('Solving Poisson equation...') # K @ u = b
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed]
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
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed]
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
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed]
        self.u[self.free] = np.linalg.solve(K_mod, b_mod)

        eps_faces = np.zeros((len(self.mesh.faces), 3))
        sigma_faces = np.zeros((len(self.mesh.faces), 3))
        compliance_faces = np.zeros(len(self.mesh.faces))
        force_faces = np.zeros(len(self.mesh.faces))

        for face_idx, face in enumerate(self.mesh.faces):
            element = self.mesh.points[face]
            u_face = self.u[np.array([2*face, 2*face+1]).T.flatten()]
            eps_faces[face_idx] = self.B_elements[face_idx] @ u_face
            sigma_faces[face_idx] = self.D_elements[face_idx] @ eps_faces[face_idx]
            compliance_faces[face_idx] = u_face.T @ self.K_faces[face_idx] @ u_face
            # force_faces[face_idx] = np.linalg.norm(np.mean((self.K_faces[face_idx] @ u_face).reshape(-1, 2), axis=0))
            # if np.max(self.K_faces[face_idx] @ u_face) > 0.1:
            #     pass

        self.solution.set_values("u", self.u)
        self.solution.set_values("rho", self.equation.parameters.get('rho', np.full(len(self.mesh.faces), 1)))
        self.solution.set_values("strain", np.linalg.norm(eps_faces, axis=-1))
        self.solution.set_values("stress", np.linalg.norm(sigma_faces, axis=-1))
        self.solution.set_values("compliance", compliance_faces)
        self.solution.set_values("total_compliance", self.u.T @ self.K @ self.u) # = sum(compliance_faces)
        # self.solution.set_values("force", force_faces)

    def _assemble_matrix(self, calculate_element_matrix, params=None): # TODO: params inconsistent here, face indexed
        N = len(self.mesh.points)
        A = np.zeros((self.dim * N, self.dim * N))
        A_elements = []
        special = self.dim == 2 and calculate_element_matrix == self._calculate_element_stiffness_matrix
        if special: # TODO: remove this
            self.B_elements, self.D_elements = [], []
        for idx, face in enumerate(self.mesh.faces):
            idxs = np.array([self.dim*face + i for i in range(self.dim)]).T.flatten()
            if special:
                element_matrix, B, D = calculate_element_matrix(face, params[idx])
                A[np.ix_(idxs, idxs)] += element_matrix
                A_elements.append(element_matrix)
                self.B_elements.append(B)
                self.D_elements.append(D)
            else:
                element_matrix = calculate_element_matrix(face, params)
                A[np.ix_(idxs, idxs)] += element_matrix
                A_elements.append(element_matrix)

        return A, A_elements

    def _assemble_vector(self, calculate_element_vector, params): #
        b = np.zeros((len(self.mesh.points), self.dim))
        if calculate_element_vector == self._calculate_element_boundary_load_vector:
            for bedge in self.mesh.boundary:
                b[bedge] += calculate_element_vector(bedge, params[bedge])
        else:
            for face in self.mesh.faces:
                b[face] += calculate_element_vector(face, params[face])
        return b.flatten()

    def _calculate_element_load_vector(self, element, param): #
        return param * 1/3 * calculate_triangle_area(self.mesh.points[element])

    def _calculate_element_boundary_load_vector(self, element, param):
        return param * 1/2 * np.linalg.norm(self.mesh.points[element][0] - self.mesh.points[element][1])

    def _calculate_element_mass_matrix(self, element, param): #
        M = np.zeros((self.dim*len(element), self.dim*len(element)))
        M[::self.dim, ::self.dim] = 1
        M += np.eye(self.dim*len(element))
        area = calculate_triangle_area(self.mesh.points[element])
        if param is None:
            return 1/12 * area * M
        raise ValueError('Not implemented')
        # return 1/12 * area * param.flatten() * M

    def _calculate_element_stiffness_matrix(self, element, param): #
        if self.dim == 1: # TODO collapse
            P = np.hstack([np.ones((3, 1)), self.mesh.points[element]])
            area = calculate_triangle_area(self.mesh.points[element])
            phis = np.linalg.solve(P, np.eye(3))[1:].T
            param = 1
            return phis @ phis.T * area * param
        else: # dim == 2
            # outputs 6x6 element stiffness matrix = a(u, v) = int (sigma(u) : epsilon(v)) over element
            E, nu = param
            if E > 1:
                pass
            mu, lamb = Enu_to_Lame(E, nu) # TODO: make space varying?
            D = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + \
                lamb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
            area, a, b, c = calculate_hat_gradients(self.mesh.points[element])
            B = np.array([[b[0], 0, b[1], 0, b[2], 0],
                        [0, c[0], 0, c[1], 0, c[2]],
                        [c[0], b[0], c[1], b[1], c[2], b[2]]])
            return B.T @ D @ B * area, B, D

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



