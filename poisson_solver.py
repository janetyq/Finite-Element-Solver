import numpy as np
from utils.base_solver import *
from utils.matrices import *
from utils.mesh import *
from utils.helper import *

class PoissonSolverResult(BaseSolverResult):
    def __init__(self, u_values, force=None):
        super().__init__(u_values)
        self.force = force

class PoissonSolver(BaseSolver):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.load_function = None # TODO: better place to set this?

    def solve(self, dirichlet_bc=None, neumann_bc=None, robin_bc=None, load_function=None):
        self.load_function = load_function
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

    def calculate_face_residuals(self):
        residuals = np.zeros(len(self.faces))
        for face_idx, face in enumerate(self.faces):
            face_points = self.points[face]
            face_area = calculate_triangle_area(face_points)
            load = np.linalg.norm([self.load_function(point) for point in face_points])
            residuals[face_idx] += load * face_area / 3
        return residuals
    
    def plot_result(self, title='Poisson Solution', ax=None, show=True, contour=20, gradient=False): # TODO: kind of weird
        if not gradient:
            return self.mesh.plot_colored(self.result.u_values, title=title, ax=ax, show=show, contour=contour)
        else:
            ax = self.mesh.plot_colored(self.result.u_values, title=title, contour=contour, ax=ax, show=False)
            gradient = self.mesh.calculate_gradient(self.result.u_values)
            return self.mesh.plot_arrows(gradient, title=title, ax=ax, show=show)
    