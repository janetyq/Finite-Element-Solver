import numpy as np
from base_solver import *
from utils.matrices import *
from utils.mesh import *
from utils.helper import *

class PoissonSolverResult(BaseSolverResult):
    def __init__(self, u_values, force=None):
        super().__init__(u_values)
        self.force = force

class PoissonSolver(BaseSolver):
    '''
    Solves Poisson equation
        -laplace(u) = f
    '''
    def __init__(self, mesh):
        super().__init__(mesh)

    def initialize(self, boundary_conditions, load_function=lambda x: 1):
        super().initialize(boundary_conditions, load_function) # explicitly writing out, so I can see it

    def solve(self):
        print('Solving Poisson equation...') # K @ u = b
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        self.u[self.free] = np.linalg.solve(K_mod, b_mod)
        self.result = PoissonSolverResult(self.u)
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
    