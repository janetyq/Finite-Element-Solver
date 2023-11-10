import numpy as np
from utils.base_solver import *
from utils.matrices import *
from utils.plotting import *

# TODO
# add boundary conditions

class ProjectionSolverResult(BaseSolverResult):
    pass

class ProjectionSolver(BaseSolver):
    def __init__(self, points, faces, boundary, matrices=None):
        super().__init__(points, faces, boundary, matrices=matrices)

    def solve(self, load_function=None):
        print('Solving L2 projection...')
        load_function = load_function if load_function is not None else lambda x: 1
        b = assemble_vector(self.points, self.faces, calculate_element_load_vector, load_function) # load vector
        u = np.linalg.solve(self.M, b)
        self.solution = ProjectionSolverResult(u)
        return self.solution

    def plot_result(self):
        plot_surface_mesh(self.points, self.faces, self.solution.u_values, title='L2 Projection')

    @classmethod
    def from_base_solver(cls, base_solver):
        return cls(base_solver.points, base_solver.faces, base_solver.boundary, matrices=(base_solver.M, base_solver.K))