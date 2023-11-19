import numpy as np
import matplotlib.pyplot as plt
from utils.matrices import *
from utils.mesh import *
from base_solver import *

def Enu_to_Lame(E, nu):
    # mu - shear modulus, lambda - Lame constant
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lamb

def Lame_to_Enu(mu, lamb):
    # E - Young's modulus, nu - Poisson's ratio
    E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
    nu = lamb / (2 * (lamb + mu))
    return E, nu

class LinearElasticSolverResult: # TODO: inherit from BaseSolverResult?
    def __init__(self, deformed_mesh, displacement, eps_faces, sigma_faces):
        self.deformed_mesh = deformed_mesh
        self.displacement = displacement
        self.eps_faces = eps_faces
        self.sigma_faces = sigma_faces

class LinearElasticSolver(BaseSolver):
    '''
    Solves linear elastics mechanics
        -grad(sigma) = f
        sigma = 2*mu*eps + lamb*tr(eps)*I
    '''
    def __init__(self, mesh, E=1, nu=0.2, alpha=0):
        super().__init__(mesh, dim=2)
        self.E = E
        self.nu = nu
        self.alpha = alpha

        self.mu, self.lamb = Enu_to_Lame(E, nu)

        # TODO: done outside of BaseSolver.initialize because of dependence on mu, lamb
        self.K = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix, func=[self.mu, self.lamb], dim=self.dim)
        
        # TODO: do I want this
        # self.B = np.zeros((2*self.N, 2*self.N)) # gradient matrix
        # self.D = np.zeros((2*self.N, 2*self.N)) # elasticity matrix

    def initialize(self, boundary_conditions, load_function=lambda x: [0, 0]):
        super().initialize(boundary_conditions, load_function)

    def solve(self):
        print('Solving linear elastic equation...') # K @ u = b
        K_mod = self.K[np.ix_(self.free, self.free)]
        b_mod = self.b[self.free] - self.K[np.ix_(self.free, self.fixed)] @ self.u[self.fixed] + self.r[self.free]
        self.u[self.free] = np.linalg.solve(K_mod, b_mod)

        eps_faces = np.zeros((len(self.faces), 3))
        sigma_faces = np.zeros((len(self.faces), 3))

        for face_idx, face in enumerate(self.faces):
            element = self.points[face]
            e_idxs = np.array([2*face, 2*face+1]).T.flatten()
            B = calculate_B(element, [self.mu, self.lamb])
            D = calculate_D(element, [self.mu, self.lamb])
            eps = B @ self.u[e_idxs]
            sigma = D @ B @ self.u[e_idxs]
            eps_faces[face_idx] = eps
            sigma_faces[face_idx] = sigma

        # print('Stress (avg):', np.mean(vonmises))
        # print('Stress (max):', np.max(vonmises))
        # print('Strain (avg):', np.mean(eps_faces))

        displaced_points = self.points + self.u.reshape((-1, 2))
        deformed_mesh = Mesh(displaced_points, self.faces, self.boundary)
        self.result = LinearElasticSolverResult(deformed_mesh, displaced_points, eps_faces, sigma_faces)
        return self.result
    
    def plot_result(self):
        fig, ax = plt.subplots(1, 2)
        self.mesh.plot(title='Undeformed Mesh', ax=ax[0], show=False)
        sigma_faces = self.result.sigma_faces
        vonmises = np.sqrt(sigma_faces[:, 0]**2 + sigma_faces[:, 1]**2 - sigma_faces[:, 0] * sigma_faces[:, 1] + 3 * sigma_faces[:, 2]**2)
        self.result.deformed_mesh.plot_colored(vonmises, title='Deformed Mesh', ax=ax[1], show=False, cbar_label='von Mises Stress')
        plt.show()
        