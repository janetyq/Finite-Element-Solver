import numpy as np
import matplotlib.pyplot as plt
from utils.matrices import *
from utils.plotting import *

# TODO:
# make a base solver for this

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

class LinearElasticSolverResult:
    def __init__(self, displacement, displaced_points, eps_faces, sigma_faces):
        self.displacement = displacement
        self.displaced_points = displaced_points
        self.eps_faces = eps_faces
        self.sigma_faces = sigma_faces

class LinearElasticSolver:
    def __init__(self, points, faces, boundary, E=1, nu=0.2, alpha=0):
        self.points = points
        self.faces = faces
        self.boundary = boundary
        self.E = E
        self.nu = nu
        self.alpha = alpha

        self.mu, self.lamb = Enu_to_Lame(E, nu)
        self.N = len(points)
        self.boundary_idxs = list(set(boundary.ravel()))
        self.inner_idxs = list(set(range(self.N)) - set(self.boundary_idxs)) # TODO: is this needed

        self.initialize_matrices()

        self.result = None

    def initialize_matrices(self):
        self.K = np.zeros((2*self.N, 2*self.N)) # stiffness matrix
        self.M = np.zeros((2*self.N, 2*self.N)) # mass matrix
        # TODO: do I want this
        # self.B = np.zeros((2*self.N, 2*self.N)) # gradient matrix
        # self.D = np.zeros((2*self.N, 2*self.N)) # elasticity matrix

        for f in self.faces:
            element = self.points[f]
            element_mass_matrix = calculate_element_elastic_mass_matrix(element)
            element_stiffness_matrix = calculate_element_elastic_stiffness_matrix(element, self.mu, self.lamb)
            f_idxs = np.array([2*f, 2*f+1]).T.flatten() # get indices of x and y coords of face nodes
            self.M[np.ix_(f_idxs, f_idxs)] += element_mass_matrix
            self.K[np.ix_(f_idxs, f_idxs)] += element_stiffness_matrix

    def solve(self, body_force=None, dirichlet_bc=None, neumann_bc=None):
        body_force = body_force if body_force is not None else lambda x: np.array([[0, 0]])
        F = np.zeros(2*self.N) # load vector
        for f in self.faces:    
            element = self.points[f]
            forces = np.apply_along_axis(body_force, axis=1, arr=element).flatten()
            f_idxs = np.array([2*f, 2*f+1]).T.flatten()
            element_mass_matrix = calculate_element_elastic_mass_matrix(element) # TODO: does this need to be recalculated?
            F[np.ix_(f_idxs)] += element_mass_matrix @ forces
        
        if neumann_bc is not None:
            neumann_boundary, neumann_force = neumann_bc
            for b in neumann_boundary:
                element = self.points[b]
                E = np.linalg.norm(element[0] - element[1])
                forces = np.apply_along_axis(neumann_force, axis=1, arr=element).flatten()
                b_idxs = np.array([2*b, 2*b+1]).T.flatten()
                F[np.ix_(b_idxs)] += 1/2 * forces * E

        d = np.zeros(2*self.N) # displacement vector

        # boundary conditions
        if dirichlet_bc is not None:
            fixed_idxs, fixed_displacements = dirichlet_bc

            fixed = np.array([2*fixed_idxs, 2*fixed_idxs+1]).T.flatten()
            d[fixed] = np.array(fixed_displacements).flatten()
            free = list(set(range(2*self.N)) - set(fixed))

            F_temp = F[free] - self.K[np.ix_(free, fixed)] @ d[fixed]
            K_temp = self.K[np.ix_(free, free)]
            d[free] = np.linalg.solve(K_temp, F_temp)
        else:
            d = np.linalg.solve(self.K, F) # TODO: is this allowed

        eps_faces = np.zeros((len(self.faces), 3))
        sigma_faces = np.zeros((len(self.faces), 3))

        for face_idx, face in enumerate(self.faces):
            element = self.points[face]
            e_idxs = np.array([2*face, 2*face+1]).T.flatten()
            B, D = calculate_something(element, self.mu, self.lamb)
            eps = B @ d[e_idxs]
            sigma = D @ B @ d[e_idxs]
            eps_faces[face_idx] = eps
            sigma_faces[face_idx] = sigma
            pass

        # print('Stress (avg):', np.mean(vonmises))
        # print('Stress (max):', np.max(vonmises))
        # print('Strain (avg):', np.mean(eps_faces))

        displaced_points = self.points + d.reshape((-1, 2))
        self.result = LinearElasticSolverResult(d, displaced_points, eps_faces, sigma_faces)
        return self.result
    
    def plot_result(self):
        fig, ax = plt.subplots(1, 2)
        plot_mesh(self.points, self.faces, ax=ax[0], title='Undeformed Mesh', show=False)
        sigma_faces = self.result.sigma_faces
        vonmises = np.sqrt(sigma_faces[:, 0]**2 + sigma_faces[:, 1]**2 - sigma_faces[:, 0] * sigma_faces[:, 1] + 3 * sigma_faces[:, 2]**2)
        plot_colored_mesh(self.result.displaced_points, self.faces, vonmises, title='Deformed Mesh', ax=ax[1], show=False, cbar_label='von Mises Stress')
        plt.show()
        