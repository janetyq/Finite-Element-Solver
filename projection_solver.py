import numpy as np
from utils.base_solver import *
from utils.matrices import *
from utils.mesh import *

# TODO
# add boundary conditions

class ProjectionSolverResult(BaseSolverResult):
    pass

class ProjectionSolver(BaseSolver):
    def __init__(self, mesh):
        super().__init__(mesh)

    def solve(self, load_function=None):
        print('Solving L2 projection...')
        load_function = load_function if load_function is not None else lambda x: 1
        b = assemble_vector(self.points, self.faces, calculate_element_load_vector, load_function) # load vector
        u = np.linalg.solve(self.M, b)
        self.solution = ProjectionSolverResult(u)
        return self.solution

    def plot_result(self):
        self.mesh.plot_surface(self.solution.u_values, title='L2 Projection')
