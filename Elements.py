import numpy as np
from utils.helper import *


class LinearElement:
    '''
    Base class for linear elements

    N nodes in (N-1)-dimensional space
    Shape function phi(x) = a + b*x_1 + c*x_2 + ... + z * x_{N-1}
    '''    
    N = None
    def __init__(self, vertices):
        self.vertices = vertices

        dN_dphi = np.vstack([-np.ones(self.N-1), np.eye(self.N-1)])
        J = (self.vertices[1:] - self.vertices[0]).T
        self.shape_gradient = dN_dphi @ np.linalg.pinv(J)

        self.dF_dx = self.calculate_dF_dx()

    def calculate_mass_matrix(self, dim, **kwargs):
        M = np.zeros((dim*self.N, dim*self.N))
        M[::dim, ::dim] = 1
        M += np.eye(dim*self.N)
        return 1/(self.N*(self.N+1)) * self.volume * M

    def calculate_stiffness_matrix(self, dim, **kwargs):
        if dim == 1:
            return self.shape_gradient @ self.shape_gradient.T * self.volume
        # otherwise, the equation is linear elastic
        mu, lamb, idx = kwargs['mu'], kwargs['lamb'], kwargs['idx']
        B, D = self.calculate_B(), self.calculate_D(kwargs['mu'][idx], kwargs['lamb'][idx])
        return B.T @ D @ B * self.volume

    # TODO: haven't checked if these make sense
    def deformation_gradient(self, u_element):
        # F = I + grad_u = I + grad_phi^T @ u
        return np.eye(N-1) + self.shape_gradient.T @ u_element

    def calculate_dF_dx(self):
        # dF_dx = I x grad_phi^T, TODO: figure out kronecker product
        dF_dx = np.zeros((self.N-1, self.N-1, self.N, self.N-1))
        for i in range(self.N-1):
            for j in range(self.N-1):
                for m in range(self.N):
                    for n in range(self.N-1):
                        if j == n:
                            dF_dx[i, j, m, n] = self.shape_gradient[m, i]
        return dF_dx

    def calculate_gradient(self, u_element):
        # grad_u = grad_phi @ u
        return self.shape_gradient.T @ u_element


class LinearLineElement(LinearElement):
    '''
    1D linear element

    Shape function phi(x) = a + b*x
    '''
    N = 2
    def __init__(self, vertices):
        self.volume = np.linalg.norm(vertices[1] - vertices[0])
        super().__init__(vertices)
    

class LinearTriangleElement(LinearElement): # TODO: perhaps put quadrature in here too?
    '''
    2D linear triangle element

    Shape function phi(x) = a + b*x + c*y
    '''
    N = 3
    def __init__(self, vertices):
        self.volume = calculate_polygon_area(vertices)
        super().__init__(vertices)
        # d2F_dx2 = 0
    
    def calculate_B(self):
        b, c = self.shape_gradient.T
        return np.array([[b[0],   0 , b[1],   0 , b[2],   0 ],
                         [  0 , c[0],   0 , c[1],   0 , c[2]],
                         [c[0], b[0], c[1], b[1], c[2], b[2]]])
    
    def calculate_D(self, mu, lamb):
        return np.array([
            [2*mu + lamb, lamb, 0],
            [lamb, 2*mu + lamb, 0],
            [0, 0, mu]
        ])


class LinearTetrahedralElement(LinearElement):
    '''
    3D linear tetrahedral element
    '''
    N = 4
    def __init__(self, vertices):
        self.volume = calculate_tetrahedron_volume(vertices)
        super().__init__(vertices)

    def calculate_B(self):
        a, b, c = self.shape_gradient.T
        return np.array([
            [a[0], 0, 0, a[1], 0, 0, a[2], 0, 0, a[3], 0, 0],
            [0, b[0], 0, 0, b[1], 0, 0, b[2], 0, 0, b[3], 0],
            [0, 0, c[0], 0, 0, c[1], 0, 0, c[2], 0, 0, c[3]],
            [b[0], a[0], 0, b[1], a[1], 0, b[2], a[2], 0, b[3], a[3], 0],
            [0, c[0], b[0], 0, c[1], b[1], 0, c[2], b[2], 0, c[3], b[3]],
            [c[0], 0, a[0], c[1], 0, a[1], c[2], 0, a[2], c[3], 0, a[3]]
        ])
    
    def calculate_D(self, mu, lamb):
        return np.array([
            [2*mu + lamb, lamb, lamb, 0, 0, 0],
            [lamb, 2*mu + lamb, lamb, 0, 0, 0],
            [lamb, lamb, 2*mu + lamb, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])

