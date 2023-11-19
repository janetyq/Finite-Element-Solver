import numpy as np
from base_solver import *
from utils.matrices import *
from utils.mesh import *

class HeatSolverResult(BaseSolverResult):
    def __init__(self, t_values, u_values):
        super().__init__(u_values)
        self.t_values = t_values

class HeatSolver(BaseSolver):
    '''
    Solves time evolution of heat equation
        u_t = u_xx
    '''
    def __init__(self, mesh, dt=0.1, num_iterations=10):
        super().__init__(mesh, dim=1)
        self.dt = dt
        self.num_iterations = num_iterations

    def initialize(self, u_initial, boundary_conditions, load_function=lambda x: 1):
        super().initialize(boundary_conditions, load_function, u_initial=u_initial)

    def solve(self):
        print('Solving heat equation...') # (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        M_mod = self.M[np.ix_(self.free, self.free)]
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]

        t_values = [0]
        u_values = [self.u]
        print(f't = {t_values[0]:.3f}, mean temp = {self.mesh.calculate_mean_value(u_values[0]):.3f}')
        for i in range(self.num_iterations):
            self.u[self.free] = np.linalg.solve(M_mod + K_mod * self.dt, M_mod @ self.u[self.free] + b_mod * self.dt)
            t_values.append(self.dt * (i+1))
            u_values.append(self.u.copy())
            print(f't = {t_values[-1]:.3f}, mean temp = {self.mesh.calculate_mean_value(u_values[-1]):.3f}')

        self.result = HeatSolverResult(t_values, u_values)
        return self.result

    def plot_result(self, fixed_cbar=False):
        self.mesh.plot_colored_animation(
            self.result.t_values, 
            self.result.u_values, 
            title='Heat Equation Simulation', 
            cbar_label='Temperature', 
            fixed_cbar=fixed_cbar, 
            contour=5
        )
