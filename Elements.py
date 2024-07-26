import numpy as np

class LinearTriangleElement: # TODO: perhaps put quadrature in here too?
    '''
    2D linear triangle element

    Shape function phi(x) = a + b*x + c*y
    '''
    def __init__(self, vertices, area):
        self.vertices = vertices
        self.area = area

        a, b, c = [], [], []
        for i in range(3):
            x_j, x_k = vertices[(i+1)%3], vertices[(i+2)%3]
            a.append((x_j[0]*x_k[1] - x_k[0]*x_j[1]) / (2 * area))
            b.append((x_j[1] - x_k[1]) / (2 * area))
            c.append((x_k[0] - x_j[0]) / (2 * area))

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