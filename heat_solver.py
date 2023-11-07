import numpy as np
from utils.base_solver import *
from utils.matrices import *
from utils.measures import *
from utils.plotting import *

# TODO
# add boundary conditions to solve
# add load function to solve

class HeatSolverResult(BaseSolverResult):
    def __init__(self, t_values, u_values):
        super().__init__(u_values)
        self.t_values = t_values

class HeatSolver(BaseSolver):
    def __init__(self, points, faces, boundary, dt=0.1, num_iterations=10):
        super().__init__(points, faces, boundary)
        self.dt = dt
        self.num_iterations = num_iterations

    def solve(self, u_initial):
        b = assemble_vector(self.points, self.faces, calculate_element_load_vector, lambda x: 1) # TODO: add load function
        u = u_initial

        t_values = [0]
        u_values = [u_initial]

        for i in range(self.num_iterations):
            # backwards Euler
            u = np.linalg.solve(self.M + self.A * self.dt, self.M @ u + b * self.dt)
            t_values.append(self.dt * (i+1))
            u_values.append(u)


        self.result = HeatSolverResult(t_values, u_values)
        return self.result

    def plot_result(self):
        plot_colored_mesh_animation(self.points, self.faces, 
                                    self.result.t_values, self.result.u_values, 
                                    title='Heat Equation Simulation',
                                    cbar_label='Temperature')