import numpy as np
from scipy.optimize import minimize

from utils.refinement import *
from BoundaryConditions import *
from Mesh import *
from Solution import *

class Equation:
    def __init__(self, name, parameters=None, dim=None):
        self.name = name
        self.parameters = parameters
        if dim is None:
            self.dim = 2 if name == "linear_elastic" else 1
            print(f"Warning: dim not provided, assuming dim={self.dim}")
        else:
            self.dim = dim

    def __copy__(self):
        return self.__class__(self.name, self.parameters.copy()) # TODO: check if this works for list values

class Solver:
    def __init__(self, femesh, equation, boundary_conditions=None):
        self.femesh = femesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions(femesh)
        self.dim = self.equation.dim
        self.solution = Solution(femesh, self.dim)

        self.boundary_conditions.do(self.femesh.vertices.shape[0], dim=self.dim)

    def solve(self):
        self.assemble_everything() # TODO: don't call this every time
        self.solution.reset()
        
        equation_solvers = {
            "projection": self.solve_projection,
            "poisson": self.solve_poisson,
            "heat": self.solve_heat,
            "wave": self.solve_wave,
            "linear_elastic": self.solve_linear_elastic,
        }

        if self.equation.name not in equation_solvers:
            raise ValueError(f"Unknown equation name: {self.equation.name}")

        equation_solvers[self.equation.name]()

        return self.solution

    def assemble_everything(self):
        if self.equation.name == "linear_elastic":
            E = np.full(len(self.femesh.elements), self.equation.parameters['E'])
            nu = np.full(len(self.femesh.elements), self.equation.parameters['nu'])
            mu, lamb = Enu_to_Lame(E, nu) 
            self.mu, self.lamb = mu, lamb
            self.femesh.prepare_matrices(dim=self.dim, mu=mu, lamb=lamb)
        else:
            self.femesh.prepare_matrices(dim=self.dim)

        self.b = (self.femesh.M @ self.boundary_conditions.force_load.flatten()).flatten()
        self.b += (self.femesh.M_b @ self.boundary_conditions.neumann_load.flatten()).flatten()

    def solve_linear_system(self, A, b, use_bc=True):
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

    def solve_nonlinear_system(self, A, b, x0, tol=1e-6, max_iters=100):
        # newton solver
        x = x0.copy()
        for iter in range(max_iters):
            print(f'iter {iter}')
            dx = self.solve_linear_system(A(x), A(x) @ x - b(x))
            if np.linalg.norm(dx) < tol:
                break
            x -= dx
        return x

    def solve_projection(self):
        print('Solving L2 projection...') # M @ u = b
        u = self.solve_linear_system(self.femesh.M, self.b)
        self.solution.set_values("u", u)
    
    def solve_poisson(self):
        print('Solving Poisson equation...') # K @ u = b
        u = self.solve_linear_system(self.femesh.K, self.b)
        self.solution.set_values("u", u)

    def solve_heat(self):
        print('Solving heat equation...') # M @ u' + K @ u = b
        #  (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        u = self.equation.parameters['u_initial']
        dt, iters = self.equation.parameters['dt'], self.equation.parameters['iters']

        t_values = [0]
        u_values = [u]
        for i in range(iters):
            u = self.solve_linear_system(self.femesh.M + self.femesh.K * dt, self.femesh.M @ u + self.b * dt)
            t_values.append(dt * (i+1))
            u_values.append(u.copy())
            print(f't = {t_values[-1]:.3f}, mean temp = {self.femesh.calculate_mean_value(u_values[-1]):.3f}')

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)

    def solve_wave(self):
        print('Solving wave equation...') # M @ u" + K @ u = b
        u, dudt = self.equation.parameters['u_initial'], self.equation.parameters['dudt_initial']
        c = self.equation.parameters['c']
        dt, iters = self.equation.parameters['dt'], self.equation.parameters['iters']
        
        # Crank-Nicolson method - average of forward and backward Euler
        A_left = np.block([[self.femesh.M, -dt/2 * self.femesh.M],
                           [c**2 * dt/2 * self.femesh.K, self.femesh.M]])
        A_right = np.block([[self.femesh.M, dt/2 * self.femesh.M],
                            [-c**2 * dt/2 * self.femesh.K, self.femesh.M]])
        b_right = np.block([np.zeros_like(self.b), dt/2 * (self.b + np.roll(self.b, -1))])

        N = len(self.femesh.vertices)
        x = np.block([u, dudt])
        t_values = [0]
        u_values = [x[:N]]
        dudt_values = [x[N:]]
        total_energy = self.femesh.calculate_energy(u_values[-1], dudt_values[-1])
        print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        for i in range(iters):
            x = self.solve_linear_system(A_left, A_right @ x + b_right, use_bc=False) # TODO: bc not supported for wave
            t_values.append(dt * (i+1))
            u_values.append(x[:N])
            dudt_values.append(x[N:])
            total_energy = self.femesh.calculate_energy(u_values[-1], dudt_values[-1])
            print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)
        self.solution.set_values("dudt_values", dudt_values)

    def solve_linear_elastic(self):
        u = self.solve_linear_system(self.femesh.K, self.b)

        eps_elements = []
        sigma_elements = []
        compliance_elements = []

        for e_idx, element in enumerate(self.femesh.elements):
            element = self.femesh.elements[e_idx]
            B = self.femesh.element_objs[e_idx].calculate_B()
            D = self.femesh.element_objs[e_idx].calculate_D(self.mu[e_idx], self.lamb[e_idx])
            u_element = u[np.array([self.dim*element + d for d in range(self.dim)]).T.flatten()]
            eps = B @ u_element
            sigma = D @ eps
            compliance = sigma @ eps * self.femesh.element_objs[e_idx].volume
            eps_elements.append(eps)
            sigma_elements.append(sigma)
            compliance_elements.append(compliance)

        self.solution.set_values("u", u)
        self.solution.set_values("strain", np.linalg.norm(eps_elements, axis=-1))
        self.solution.set_values("stress", np.linalg.norm(sigma_elements, axis=-1))
        self.solution.set_values("compliance", compliance_elements)

    def adaptive_refinement(self, max_triangles=1000, max_iters=20):
        # TODO: there's a bug somewhere
        if 'element_residuals' not in self.solution.values:
            raise ValueError('No element residuals found in solution')
        refinement_mesh = RefinementMesh(self.femesh)
        while len(self.femesh.elements) < max_triangles or max_iters == 0:
            element_residuals = self.solution.values['element_residuals']
            max_residual = max(element_residuals)
            refine_idxs = []
            for e_idx, residual in enumerate(element_residuals):
                if residual > 0.9 * max_residual:
                    refine_idxs.append(e_idx)

            refinement_mesh.refine_triangles(refine_idxs)
            self.femesh = refinement_mesh.get_mesh()
            max_iters -= 1

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

    # def calculate_projection_residuals(self, apriori=True):
    #     # Apriori error ||e|| <= C * h^2 * ||f"||
    #     if apriori:
    #         # compute apriori residual
    #         residuals = np.zeros(len(self.femesh.elements))
    #         for e_idx, element in enumerate(self.femesh.elements):
    #             residuals[e_idx] = 0 # placeholder
    #         self.solution.set_values("apriori_residual", residuals)
    #     else:
    #         # compute aposteriori residual
    #         residuals = None
    #         self.solution.set_values("aposteriori_residual", residuals)
    #         pass
