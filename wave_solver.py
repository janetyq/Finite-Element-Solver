import numpy as np
from base_solver import *
from utils.matrices import *
from utils.mesh import *

# TODO
# add choice of boundary condition

class WaveSolverResult(BaseSolverResult):
    def __init__(self, t_values, u_values, dudt_values):
        super().__init__(u_values)
        self.t_values = t_values
        self.dudt_values = dudt_values

class WaveSolver(BaseSolver):
    '''
    Solves time evolution of wave equation
        u_tt = c^2 * u_xx   (c - wave speed)
    '''
    def __init__(self, mesh, dt=0.1, num_iterations=10, c=1):
        super().__init__(mesh)
        self.dt = dt
        self.num_iterations = num_iterations
        # wave speed, TODO: function?
        self.c = 1

    def initialize(self, u_initial, dudt_initial, boundary_conditions, load_function=lambda x: 1):
        super().initialize(boundary_conditions, load_function)
        self.u_initial = u_initial
        self.dudt_initial = dudt_initial

    def solve(self):
        print('Solving wave equation...')
        x = np.block([self.u_initial, self.dudt_initial])

        # Crank-Nicolson method - average of forward and backward Euler
        A_left = np.block([[self.M, -self.dt/2 * self.M],
                        [self.c**2 * self.dt/2 * self.K, self.M]])
        A_right = np.block([[self.M, self.dt/2 * self.M],
                            [-self.c**2 * self.dt/2 * self.K, self.M]])

        b_right = np.block([np.zeros_like(self.b), self.dt/2 * (self.b + np.roll(self.b, -1))])

        t_values = [0]
        u_values = [x[:self.N]]
        dudt_values = [x[self.N:]]
        total_energy = self.mesh.calculate_energy(u_values[-1], dudt_values[-1])
        print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        for i in range(self.num_iterations):
            x = np.linalg.solve(A_left, A_right @ x + b_right)

            t_values.append(self.dt * (i+1))
            u_values.append(x[:self.N])
            dudt_values.append(x[self.N:])
            total_energy = self.mesh.calculate_energy(u_values[-1], dudt_values[-1])
            print(f't = {t_values[-1]:.3f}, total energy = {total_energy:.3f}')

        self.result = WaveSolverResult(t_values, u_values, dudt_values)
        return self.result

    def plot_result(self):
        self.mesh.plot_surface_animation(
            self.result.t_values, 
            self.result.u_values, 
            title='Wave Equation Simulation'
        )
    