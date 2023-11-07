import numpy as np
from utils.base_solver import *
from utils.matrices import *
from utils.measures import *
from utils.plotting import *

# TODO
# everything like heat solver
# figure out what this means
# rename solver result vars


class WaveSolverResult(BaseSolverResult):
    def __init__(self, t_values, u_values, dudt_values):
        super().__init__(u_values)
        self.t_values = t_values
        self.dudt_values = dudt_values

class WaveSolver(BaseSolver):
    def __init__(self, points, faces, boundary, dt=0.1, num_iterations=10, c=1):
        super().__init__(points, faces, boundary)
        self.dt = dt
        self.num_iterations = num_iterations
        # wave speed, TODO: function?
        self.c = 1

    def solve(self, u_initial, dudt_initial):
        b = assemble_vector(self.points, self.faces, calculate_element_load_vector, lambda x: 1)
        x = np.block([u_initial, dudt_initial])
        A_left = np.block([[self.M, -self.dt/2 * self.M],
                        [self.c**2 * self.dt/2 * self.A, self.M]])
        A_right = np.block([[self.M, self.dt/2 * self.M],
                            [-self.c**2 * self.dt/2 * self.A, self.M]])

        b_right = np.block([np.zeros_like(b), self.dt/2 * (b + np.roll(b, -1))])

        t_values = [0]
        u_values = [u_initial]
        dudt_values = [dudt_initial]

        for i in range(self.num_iterations):
            x = np.linalg.solve(A_left, A_right @ x + b_right)
            t_values.append(self.dt * (i+1))
            u_values.append(x[:self.N])
            dudt_values.append(x[self.N:])

        self.result = WaveSolverResult(t_values, u_values, dudt_values)
        return self.result

    def plot_result(self):
        plot_surface_mesh_animation(self.points, self.faces, 
                                    self.result.t_values, self.result.u_values, 
                                    title='Wave Equation Simulation')
    