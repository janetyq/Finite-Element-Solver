import numpy as np
from utils.base_solver import *
from utils.matrices import *
from utils.plotting import *

class PoissonSolverResult(BaseSolverResult):
    def __init__(self, u_values, force=None):
        super().__init__(u_values)
        self.force = force

class PoissonSolver(BaseSolver):
    def __init__(self, points, faces, boundary, matrices=None):
        super().__init__(points, faces, boundary, matrices=matrices)

    def solve(self, dirichlet_bc=None, neumann_bc=None, robin_bc=None, load_function=None):
        print('Solving Poisson equation...')
        load_function = load_function if load_function is not None else lambda x: 1
        b = assemble_vector(self.points, self.faces, calculate_element_load_vector, load_function) # load vector
        if robin_bc is not None:
            W, g_D, g_N = robin_bc
            R = assemble_matrix(self.points, self.boundary, calculate_element_boundary_mass_matrix, W)
            r = assemble_vector(self.points, self.boundary, calculate_element_boundary_load_vector, 
                                lambda x: W(x) * g_D(x) + g_N(x))
            u = np.linalg.solve(self.K + R, b + r)
            self.result = PoissonSolverResult(u)
        elif neumann_bc is not None:
            g_N = neumann_bc
            C = assemble_vector(self.points, self.faces, calculate_element_load_vector, lambda x: 1) # TODO: why is this lambda 1?
            A_temp = np.zeros((self.N+1, self.N+1))
            A_temp[:self.N, :self.N] = self.K
            A_temp[:self.N, self.N] = C
            A_temp[self.N, :self.N] = C
            A_temp[self.N, self.N] = 0

            boundary_value = assemble_vector(self.points, self.boundary, calculate_element_boundary_load_vector, g_N)
            b_temp = np.append(b + boundary_value, 0)
            u = np.zeros(self.N)
            soln = np.linalg.solve(A_temp, b_temp)
            u[:self.N] = soln[:self.N]
            force = soln[self.N]
            self.result = PoissonSolverResult(u, force)
        elif dirichlet_bc is not None:
            g_D = dirichlet_bc
            u = np.zeros(self.N)
            # imposed boundary values
            boundary_value = np.apply_along_axis(g_D, axis=1, arr=self.points[self.boundary_idxs])
            b_temp = b[self.inner_idxs] - self.K[np.ix_(self.inner_idxs, self.boundary_idxs)] @ boundary_value
            K_temp = self.K[np.ix_(self.inner_idxs, self.inner_idxs)]
            u[self.inner_idxs] = np.linalg.solve(K_temp, b_temp)
            u[self.boundary_idxs] = boundary_value
            self.result = PoissonSolverResult(u)
        return self.result
    
    def plot_result(self, title='Poisson Solution', ax=None, show=True, gradient=False): # TODO: kind of weird
        if not gradient:
            return plot_colored_mesh(self.points, self.faces, self.result.u_values, title=title, contour=True, ax=ax, show=show)
        else:
            ax = plot_colored_mesh(self.points, self.faces, self.result.u_values, title=title, contour=True, ax=ax, show=False)
            return plot_gradient_arrows(self.points, self.faces, self.result.u_values, title=title, ax=ax, show=show)
    
    @classmethod
    def from_base_solver(cls, base_solver):
        return cls(base_solver.points, base_solver.faces, base_solver.boundary, matrices=(base_solver.M, base_solver.K))