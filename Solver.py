import numpy as np
from scipy.optimize import minimize

from utils.refinement import *
from BoundaryConditions import *
from Mesh import *
from Solution import *

class Equation:
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters
        self.dim = 2 if name == "linear_elastic" else 1

    def __copy__(self):
        return self.__class__(self.name, self.parameters.copy()) # TODO: check if this works for list values

class Solver:
    def __init__(self, mesh, equation, boundary_conditions=None):
        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions(mesh)
        self.solution = Solution(mesh)
        self.dim = self.equation.dim

        self.boundary_conditions.do(self.mesh.vertices.shape[0], dim=self.dim)

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.solution = Solution(mesh)

    def set_bc(self, bc):
        self.boundary_conditions = bc
        self.boundary_conditions.do(self.mesh.vertices.shape[0], dim=self.equation.dim)

    def preprocess(self):
        # assemble mass and stiffness matrices and load vector
        self.M, self.M_elements = self._assemble_matrix(self._calculate_element_mass_matrix)
        self.b = self._assemble_vector(self._calculate_element_load_vector, self.boundary_conditions.force_load)
        self.b += self._assemble_vector(self._calculate_element_boundary_load_vector, self.boundary_conditions.neumann_load)

        if self.equation.dim == 1: # TODO: implement 2D
            self.K, self.K_elements = self._assemble_matrix(self._calculate_element_stiffness_matrix)
        if self.equation.name == "linear_elastic":
            E = np.full(len(self.mesh.elements), self.equation.parameters['E'])
            nu = np.full(len(self.mesh.elements), self.equation.parameters['nu'])
            self.material_func = np.vstack([E, nu]).T 
            self.K, self.K_elements = self._assemble_matrix(self._calculate_element_stiffness_matrix, self.material_func)

    def solve(self):
        self.preprocess() # TODO: don't call this every time
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
    #         residuals = np.zeros(len(self.mesh.elements))
    #         for e_idx, element in enumerate(self.mesh.elements):
    #             residuals[e_idx] = 0 # placeholder
    #         self.solution.set_values("apriori_residual", residuals)
    #     else:
    #         # compute aposteriori residual
    #         residuals = None
    #         self.solution.set_values("aposteriori_residual", residuals)
    #         pass

    def _solve_linear_system(self, A, b, use_bc=True):
        # solves Au=b, taking fixed vars and loads into account
        x = np.zeros_like(b)
        if use_bc:
            free = self.boundary_conditions.free_idxs
            fixed = self.boundary_conditions.fixed_idxs
            x[fixed] = self.boundary_conditions.fixed_values
            A_mod = A[np.ix_(free, free)]
            b_mod = b[free] - A[np.ix_(free, fixed)] @ x[fixed]
            x[free] = np.linalg.solve(A_mod, b_mod)
            return x
        else:
            return np.linalg.solve(A, b)

    def _solve_nonlinear_system(self, A, b, x0, tol=1e-6, max_iters=100):
        # newton solver
        x = x0.copy() # TODO: x0 needed only for shape, do better
        for iter in range(max_iters):
            print(f'iter {iter}')
            dx = self._solve_linear_system(A(x), A(x) @ x - b(x))
            if np.linalg.norm(dx) < tol:
                break
            x -= dx
        return x

    def _solve_projection(self):
        print('Solving L2 projection...') # M @ u = b
        u = self._solve_linear_system(self.M, self.b)
        # u = self._solve_nonlinear_system(lambda _: self.M, lambda _: self.b, x0=np.zeros_like(self.b))
        self.solution.set_values("u", u)
    
    def _solve_poisson(self):
        print('Solving Poisson equation...') # K @ u = b
        u = self._solve_linear_system(self.K, self.b)
        self.solution.set_values("u", u)
        # residuals = np.zeros(len(self.mesh.elements))
        # for e_idx, element in enumerate(self.mesh.elements):
        #     load = np.linalg.norm([self.load_function(point) for point in self.mesh.vertices[element]])
        #     residuals[e_idx] += load * self.mesh.areas[e_idx] / 3
        # self.solution.set_values("element_residuals", residuals)

    def _solve_heat(self):
        print('Solving heat equation...') # M @ u' + K @ u = b
        #  (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        u = self.equation.parameters['u_initial']
        dt, iters = self.equation.parameters['dt'], self.equation.parameters['iters']

        t_values = [0]
        u_values = [u]
        for i in range(iters):
            u = self._solve_linear_system(self.M + self.K * dt, self.M @ u + self.b * dt)
            t_values.append(dt * (i+1))
            u_values.append(u.copy())
            print(f't = {t_values[-1]:.3f}, mean temp = {self.mesh.calculate_mean_value(u_values[-1]):.3f}')

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)

    def _solve_wave(self):
        print('Solving wave equation...') # M @ u" + K @ u = b
        u, dudt = self.equation.parameters['u_initial'], self.equation.parameters['dudt_initial']
        c = self.equation.parameters['c']
        dt, iters = self.equation.parameters['dt'], self.equation.parameters['iters']
        
        # Crank-Nicolson method - average of forward and backward Euler
        A_left = np.block([[self.M, -dt/2 * self.M],
                           [c**2 * dt/2 * self.K, self.M]])
        A_right = np.block([[self.M, dt/2 * self.M],
                            [-c**2 * dt/2 * self.K, self.M]])
        b_right = np.block([np.zeros_like(self.b), dt/2 * (self.b + np.roll(self.b, -1))])

        N = len(self.mesh.vertices)
        x = np.block([u, dudt])
        t_values = [0]
        u_values = [x[:N]]
        dudt_values = [x[N:]]
        total_energy = self.mesh.calculate_energy(u_values[-1], dudt_values[-1])
        print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        for i in range(iters):
            x = self._solve_linear_system(A_left, A_right @ x + b_right, use_bc=False) # TODO: bc not supported for wave
            t_values.append(dt * (i+1))
            u_values.append(x[:N])
            dudt_values.append(x[N:])
            total_energy = self.mesh.calculate_energy(u_values[-1], dudt_values[-1])
            print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)
        self.solution.set_values("dudt_values", dudt_values)

    def _solve_linear_elastic(self):
        u = self._solve_linear_system(self.K, self.b)

        eps_elements = np.zeros((len(self.mesh.elements), 3))
        sigma_elements = np.zeros((len(self.mesh.elements), 3))
        compliance_elements = np.zeros(len(self.mesh.elements))
        force_elements = np.zeros(len(self.mesh.elements))

        for e_idx, element in enumerate(self.mesh.elements):
            element = self.mesh.elements[e_idx]
            u_element = u[np.array([2*element, 2*element+1]).T.flatten()]
            eps_elements[e_idx] = self.B_elements[e_idx] @ u_element
            sigma_elements[e_idx] = self.D_elements[e_idx] @ eps_elements[e_idx]
            compliance_elements[e_idx] = u_element.T @ self.K_elements[e_idx] @ u_element
            # force_elements[e_idx] = np.linalg.norm(np.mean((self.K_elements[e_idx] @ u_element).reshape(-1, 2), axis=0))

        self.solution.set_values("u", u)
        self.solution.set_values("strain", np.linalg.norm(eps_elements, axis=-1))
        self.solution.set_values("stress", np.linalg.norm(sigma_elements, axis=-1))
        self.solution.set_values("compliance", compliance_elements)
        self.solution.set_values("total_compliance", 0.5 * (u.T @ self.K @ u)) # = sum(compliance_elements)
        # self.solution.set_values("force", force_elements)

    def _assemble_matrix(self, calculate_element_matrix, params=None): # TODO: params inconsistent here, element indexed
        N = len(self.mesh.vertices)
        A = np.zeros((self.dim * N, self.dim * N))
        A_elements = []
        special = self.dim == 2 and calculate_element_matrix == self._calculate_element_stiffness_matrix
        if special: # TODO: remove this
            self.B_elements, self.D_elements = [], []
        for e_idx, element in enumerate(self.mesh.elements):
            idxs = np.array([self.dim*element + i for i in range(self.dim)]).T.flatten()
            if special:
                element_matrix, B, D = calculate_element_matrix(e_idx, params[e_idx])
                A[np.ix_(idxs, idxs)] += element_matrix
                A_elements.append(element_matrix)
                self.B_elements.append(B)
                self.D_elements.append(D)
            else:
                element_matrix = calculate_element_matrix(e_idx, params)
                A[np.ix_(idxs, idxs)] += element_matrix
                A_elements.append(element_matrix)

        return A, A_elements

    def _assemble_vector(self, calculate_element_vector, params):
        b = np.zeros((len(self.mesh.vertices), self.dim))
        if calculate_element_vector == self._calculate_element_boundary_load_vector:
            for b_idx, bedge in enumerate(self.mesh.boundary):
                b[bedge] += calculate_element_vector(b_idx, params[bedge])
        else:
            for e_idx, element in enumerate(self.mesh.elements):
                b[element] += calculate_element_vector(e_idx, params[element])
        return b.flatten()

    def _calculate_element_load_vector(self, e_idx, param):
        return param * 1/3 * self.mesh.areas[e_idx]

    def _calculate_element_boundary_load_vector(self, b_idx, param): # TODO: generalize this
        boundary = self.mesh.boundary[b_idx]
        return param * 1/2 * np.linalg.norm(self.mesh.vertices[boundary][0] - self.mesh.vertices[boundary][1])

    def _calculate_element_mass_matrix(self, e_idx, param):
        element = self.mesh.elements[e_idx]
        M = np.zeros((self.dim*len(element), self.dim*len(element)))
        M[::self.dim, ::self.dim] = 1
        M += np.eye(self.dim*len(element))
        if param is None:
            return 1/12 * self.mesh.areas[e_idx] * M
        raise ValueError('Not implemented')
        # return 1/12 * self.mesh.areas[e_idx] * param.flatten() * M

    def _calculate_element_stiffness_matrix(self, e_idx, param): # TODO: hat gradients not fixed
        element = self.mesh.elements[e_idx]
        if self.dim == 1: # TODO collapse
            P = np.hstack([np.ones((3, 1)), self.mesh.vertices[element]])
            phis = np.linalg.solve(P, np.eye(3))[1:].T
            param = 1
            return phis @ phis.T * self.mesh.areas[e_idx] * param
        else: # dim == 2
            # outputs 6x6 element stiffness matrix = a(u, v) = int (sigma(u) : epsilon(v)) over element
            E, nu = param
            if E > 1:
                pass
            mu, lamb = Enu_to_Lame(E, nu) # TODO: make space varying?
            D = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + \
                lamb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
            # area, a, b, c = calculate_hat_gradients(self.mesh.vertices[element])
            hat_grad = self.mesh.shape_functions[e_idx].gradient
            b = hat_grad[:, 0]
            c = hat_grad[:, 1]
            B = np.array([[b[0], 0, b[1], 0, b[2], 0],
                        [0, c[0], 0, c[1], 0, c[2]],
                        [c[0], b[0], c[1], b[1], c[2], b[2]]])
            return B.T @ D @ B * self.mesh.areas[e_idx], B, D

    def adaptive_refinement(self, max_triangles=1000, max_iters=20):
        # TODO: there's a bug somewhere
        if 'element_residuals' not in self.solution.values:
            raise ValueError('No element residuals found in solution')
        refinement_mesh = RefinementMesh(self.mesh)
        while len(self.mesh.elements) < max_triangles or max_iters == 0:
            element_residuals = self.solution.values['element_residuals']
            max_residual = max(element_residuals)
            refine_idxs = []
            for e_idx, residual in enumerate(element_residuals):
                if residual > 0.9 * max_residual:
                    refine_idxs.append(e_idx)

            refinement_mesh.refine_triangles(refine_idxs)
            self.mesh = refinement_mesh.get_mesh()
            max_iters -= 1


