import numpy as np
import matplotlib.pyplot as plt

from utils.matrices import *
from utils.mesh import *
from base_solver import *

class LinearElasticSolverResult: # TODO: inherit from BaseSolverResult?
    '''
    Stores results from linear elastic solver
    TODO: performs any post-processing
    '''

    def __init__(self, deformed_mesh, displacement):
        self.deformed_mesh = deformed_mesh.copy()
        self.displacement = displacement.copy()
        self.stress = None
        self.strain = None
        self.values = {}

    def add_stress_strain(self, stress, strain):
        self.stress = stress # sigma
        self.strain = strain # epsilon
    
    def add_var_value(self, var, value):
        self.values[var] = value.copy()


    def plot(self, variable="stress", title=None, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots()
        if variable == "stress":
            vonmises = np.sqrt(self.stress[:, 0]**2 + self.stress[:, 1]**2 - self.stress[:, 0] * self.stress[:, 1] + 3 * self.stress[:, 2]**2)
            self.deformed_mesh.plot_colored(vonmises, title='Deformed Mesh', ax=ax, show=False, cbar_label='von Mises Stress')
        elif variable == "rho":
            self.deformed_mesh.plot_colored(self.values['rho'], title='Deformed Mesh', ax=ax, show=False, cbar_label='Density')
        ax.set_title(title)
        if show:
            plt.show()

class LinearElasticSolver(BaseSolver):
    '''
    Solves linear elastics mechanics
        -grad(sigma) = f
        sigma = 2*mu*eps + lamb*tr(eps)*I
    '''
    def __init__(self, mesh):
        super().__init__(mesh, dim=2)
        self.result = None

    def initialize(self, boundary_conditions, load_function=lambda x: [0, 0], E=None, nu=None):
        self.initialize_material(E, nu)
        super().initialize(boundary_conditions, load_function)

    def initialize_material(self, E, nu):
        self.E = E if isinstance(E, (np.ndarray, list)) else np.full(len(self.faces), E)
        self.nu = nu if isinstance(nu, (np.ndarray, list)) else np.full(len(self.faces), nu)
        self.K = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix, func=np.vstack([self.E, self.nu]).T, dim=self.dim)

    def initialize_bc(self, boundary_conditions, load_function=lambda x: [0, 0]):
        super().initialize(boundary_conditions, load_function)

    def calculate_stress_strain_distribution(self):
        if self.result is None:
            raise Exception('Solver has not been run yet')

        func = np.vstack([self.E, self.nu]).T
        self.B = np.array([calculate_B(element, (func, e_idx)) for e_idx, element in enumerate(self.points[self.faces])]) # gradient matrix
        self.D = np.array([calculate_D(element, (func, e_idx)) for e_idx, element in enumerate(self.points[self.faces])]) # elasticity matrix
        
        eps_faces = np.zeros((len(self.faces), 3))
        sigma_faces = np.zeros((len(self.faces), 3))

        for face_idx, face in enumerate(self.faces):
            element = self.points[face]
            e_idxs = np.array([2*face, 2*face+1]).T.flatten()
            eps = self.B[face_idx] @ self.u[e_idxs]
            sigma = self.D[face_idx] @ eps
            eps_faces[face_idx] = eps
            sigma_faces[face_idx] = sigma

        self.result.strain = eps_faces
        self.result.stress = sigma_faces

        return sigma_faces, eps_faces

    def solve(self, stress_strain=True):
        print('Solving linear elastic equation...') # K @ u = b
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        self.u[self.free] = spsolve(K_mod, b_mod)
        displacement = self.u
        deformed_mesh = Mesh(self.points + displacement.reshape((-1, 2)), self.faces, self.boundary)
        self.result = LinearElasticSolverResult(deformed_mesh, displacement)
        if stress_strain is True:
            self.calculate_stress_strain_distribution()
        return self.result


    def calculate_compliance(self): #TODO: allow external results?
        if self.result is None:
            raise Exception('Solver has not been run yet')
        # calculates compliance based on displacement from solution
        C_total = 1/2 * self.result.displacement.T @ self.K @ self.result.displacement
        C_faces = np.zeros(len(self.faces))
        func = np.vstack([self.E, self.nu]).T
        for face_idx, face in enumerate(self.faces):
            e_idxs = np.array([2*face, 2*face+1]).T.flatten()
            u_face = self.result.displacement[e_idxs]
            K_face = calculate_element_stiffness_matrix(self.points[face], (func, face_idx), dim=2)
            C_faces[face_idx] = 1/2 * u_face.T @ K_face @ u_face
        return C_total, C_faces

    def plot_result(self, title=None, ax=None, show=True):
        if self.result is None:
            raise Exception('Solver has not been run yet')
        self.result.plot(title=title, ax=ax, show=show)
    
    def plot_deformation(self):
        if self.result is None:
            raise Exception('Solver has not been run yet')
        
        fig, ax = plt.subplots(1, 2)
        self.mesh.plot(title='Undeformed Mesh', ax=ax[0], show=False)
        self.plot_result(title='Deformed Mesh', ax=ax[1], show=False)
        plt.show()
