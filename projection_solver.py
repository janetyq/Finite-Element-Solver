import numpy as np
from base_solver import *
from utils.matrices import *
from utils.mesh import *

class ProjectionSolverResult(BaseSolverResult):
    pass

class ProjectionSolver(BaseSolver):
    '''
    Solves for best L2 projection of a function onto the space of piecewise linear functions
    '''
    def __init__(self, mesh):
        super().__init__(mesh, dim=1)

    def initialize(self, boundary_conditions, load_function):
        super().initialize(boundary_conditions, load_function)

    def solve(self):
        print('Solving L2 projection...') # M @ u = b
        M_mod = self.M[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.M[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        self.u[self.free] = np.linalg.solve(M_mod, b_mod)
        self.solution = ProjectionSolverResult(self.u)
        return self.solution

    def plot_result(self, title='L2 Projection', ax=None, show=True):
        return self.mesh.plot_surface(self.solution.u_values, title=title, ax=ax, show=show)
