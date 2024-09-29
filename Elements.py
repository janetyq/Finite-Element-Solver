import numpy as np
from utils.helper import *

class LinearTriangleElement: # TODO: perhaps put quadrature in here too?
    '''
    2D linear triangle element

    Shape function phi(x) = a + b*x + c*y
    '''
    def __init__(self, vertices):
        self.vertices = vertices
        self.volume = calculate_polygon_area(vertices)

        a, b, c = [], [], []
        for i in range(3):
            x_j, x_k = vertices[(i+1)%3], vertices[(i+2)%3]
            a.append((x_j[0]*x_k[1] - x_k[0]*x_j[1]) / (2 * self.volume))
            b.append((x_j[1] - x_k[1]) / (2 * self.volume))
            c.append((x_k[0] - x_j[0]) / (2 * self.volume))

        self.gradient = np.array([b, c]).T # shape (3, 2)

        self.dF_dx = self.calculate_dF_dx()
        # d2F_dx2 = 0

    def deformation_gradient(self, u_element):
        # F = I + grad_u = I + grad_phi^T @ u
        return np.eye(2) + self.gradient.T @ u_element

    def calculate_dF_dx(self):
        # dF_dx = I x grad_phi^T, TODO: figure out kronecker product
        dF_dx = np.zeros((2, 2, 3, 2))
        for i in range(2):
            for j in range(2):
                for m in range(3):
                    for n in range(2):
                        if j == n:
                            dF_dx[i, j, m, n] = self.gradient[m, i]
        return dF_dx


class LinearTetrahedralElement:
    def __init__(self, vertices):
        self.vertices = vertices
        self.volume = calculate_tetrahedron_volume(vertices)

        a, b, c, d = [], [], [], []
        for i in range(4):
            x_j, x_k, x_l = vertices[(i+1)%4], vertices[(i+2)%4], vertices[(i+3)%4]
            a.append(np.linalg.det([x_j, x_k, x_l]) / (6 * self.volume))
            b.append(np.linalg.det([x_k, x_l, x_j]) / (6 * self.volume))
            c.append(np.linalg.det([x_l, x_j, x_k]) / (6 * self.volume))
            d.append(np.linalg.det([x_k, x_j, x_l]) / (6 * self.volume))

        self.gradient = np.array([b, c, d]).T

        self.dF_dx = self.calculate_dF_dx()

    def deformation_gradient(self, u_element):
        # F = I + grad_u = I + grad_phi^T @ u
        return np.eye(3) + self.gradient.T @ u_element

    def calculate_dF_dx(self):
        dF_dx = np.zeros((3, 3, 4, 3))
        for i in range(3):
            for j in range(3):
                for m in range(4):
                    for n in range(3):
                        if j == n:
                            dF_dx[i, j, m, n] = self.gradient[m, i]
        return dF_dx
