import numpy as np
import matplotlib.pyplot as plt
from utils.matrices import *
from utils.mesh import *
from utils.half_edge import *
from base_solver import *

class LinearElasticSolverResult: # TODO: inherit from BaseSolverResult?
    def __init__(self, deformed_mesh, displacement, eps_faces, sigma_faces):
        self.deformed_mesh = deformed_mesh
        self.displacement = displacement
        self.eps_faces = eps_faces
        self.sigma_faces = sigma_faces

    def copy(self):
        return LinearElasticSolverResult(self.deformed_mesh.copy(), self.displacement.copy(), self.eps_faces.copy(), self.sigma_faces.copy())

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
        self.initialize_material_matrices()

    def initialize_material_matrices(self):
        self.K = assemble_matrix(self.points, self.faces, calculate_element_stiffness_matrix, func=[self.E, self.nu], dim=self.dim)
        self.B = np.array([calculate_B(element, [self.mu, self.lamb]) for element in self.points[self.faces]]) # gradient matrix
        self.D = np.array([calculate_D(element, [self.mu, self.lamb]) for element in self.points[self.faces]]) # elasticity matrix

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
            eps = self.B[face_idx] @ self.u[e_idxs]
            sigma = self.D[face_idx] @ eps
            eps_faces[face_idx] = eps
            sigma_faces[face_idx] = sigma

        # print('Stress (avg):', np.mean(vonmises))
        # print('Stress (max):', np.max(vonmises))
        # print('Strain (avg):', np.mean(eps_faces))

        displacement = self.u.copy()
        deformed_mesh = Mesh(self.points + displacement.reshape((-1, 2)), self.faces, self.boundary)
        self.result = LinearElasticSolverResult(deformed_mesh, displacement, eps_faces, sigma_faces)
        return self.result

    def solve_with_density(self, rho):
        E = 1e-9 + self.E * rho**self.alpha
        self.K = assemble_matrix2(self.points, self.faces, calculate_element_stiffness_matrix, func=[E, self.nu], dim=self.dim)
        self.solve()

    def topology_optimization(self, rho, E_0, penalization=3, num_iterations=10, alpha=0.001, beta=0.5):
        # rho is list of densities for each face element

        self.initialize(self.boundary_conditions, self.load_function)

        total_volume = np.sum([calculate_triangle_area(self.points[face]) for face in self.faces])
        target_volume = 0.5 * total_volume

        he_mesh = HalfEdgeMesh(self.mesh)
        
        for _ in range(num_iterations):
            self.solve_with_density(rho)
            C = 1/2 * self.result.displacement.T @ self.K @ self.result.displacement
            C_faces = np.zeros(len(rho))
            for face_idx, face in enumerate(self.faces):
                element = self.points[face]
                e_idxs = np.array([2*face, 2*face+1]).T.flatten()
                u_face = self.result.displacement[e_idxs]
                K_face = calculate_element_stiffness_matrix(element, [E[face_idx], self.nu], dim=2)
                C_faces[face_idx] = 1/2 * u_face.T @ K_face @ u_face
            dC_drho = penalization * E_0 * rho**(penalization-1) * C_faces

            # self.mesh.plot_colored(dC_drho, title='dC_drho', show=True)

            rho += alpha * dC_drho

            current_volume = np.sum([rho[face_idx] * calculate_triangle_area(self.points[face]) for face_idx, face in enumerate(self.faces)])
            rho += beta*(target_volume - current_volume) / total_volume
            rho = np.clip(rho, 0, 1)

            current_volume = np.sum([rho[face_idx] * calculate_triangle_area(self.points[face]) for face_idx, face in enumerate(self.faces)])
            print('C:', C)
            print('Volume ratio:', current_volume/total_volume)
            print('max displacement', np.max(self.result.displacement))

            # smoothing
            smoothed_rho = np.zeros(len(rho))
            for face_idx, face in enumerate(self.mesh.faces):
                neighbor_value = np.mean([rho[f_idx] for f_idx in he_mesh.get_f_neighbor_f_idxs(face_idx)])
                smoothed_rho[face_idx] = 0.5 * neighbor_value + 0.5 * rho[face_idx]
            rho = smoothed_rho

            if _ % 20 == 1:
                self.mesh.plot_colored(rho, title='Density', show=True)

        current_volume = np.sum([rho[face_idx] * calculate_triangle_area(self.points[face]) for face_idx, face in enumerate(self.faces)])
        rho += (target_volume - current_volume) / total_volume

        self.solve_with_density(rho)
        C = 1/2 * self.result.displacement.T @ self.K @ self.result.displacement
        print('Final')
        print('C:', C) 
        current_volume = np.sum([rho[face_idx] * calculate_triangle_area(self.points[face]) for face_idx, face in enumerate(self.faces)])
        print('Volume ratio:', current_volume/total_volume)
        print('max displacement', np.max(self.result.displacement))

        return rho


    
    def plot_result(self, result=None):
        if result is None:
            result = self.result
        fig, ax = plt.subplots(1, 2)
        self.mesh.plot(title='Undeformed Mesh', ax=ax[0], show=False)
        sigma_faces = result.sigma_faces
        vonmises = np.sqrt(sigma_faces[:, 0]**2 + sigma_faces[:, 1]**2 - sigma_faces[:, 0] * sigma_faces[:, 1] + 3 * sigma_faces[:, 2]**2)
        result.deformed_mesh.plot_colored(vonmises, title='Deformed Mesh', ax=ax[1], show=False, cbar_label='von Mises Stress')
        plt.show()
        