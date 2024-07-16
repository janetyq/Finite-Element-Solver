import numpy as np
from utils.helper import check_gradient

class LinearElasticEnergyDensity: # TODO: inheritance
    '''
    Strain energy density on 2D triangular elements

    F: deformation gradient dx/dX
    S: strain tensor
    W: strain energy density
    '''
    def __init__(self):
        self.E = 200
        self.nu = 0.4
        self.mu, self.lamb = self.Enu_to_Lame(self.E, self.nu)

    def set_shape_function(self, shape_function):
        self.shape_function = shape_function

    def calc_W_from_x(self, x):
        F = self.calc_F_from_x(x)
        return self.calc_W_from_F(F)

    def calc_F_from_x(self, x_element): 
        F = np.eye(2)
        for i in range(2):
            for j in range(2):
                for k in range(3): # summation over k
                    F[i, j] += x_element[k, i] * self.shape_function.gradient[k, j]
        return F
    
    def calc_W_from_S(self, S):
        return 0.5 * (self.lamb * np.trace(S)**2 + 2 * self.mu * np.trace(S.T @ S))

    def calc_S_from_F(self, F):
        return 0.5 * (F.T @ F - np.eye(2))

    def calc_W_from_F(self, F):
        return self.calc_W_from_S(self.calc_S_from_F(F))

    # TODO: better understand gradient order
    # TODO: fix shape function issue

    def calc_dW_dS(self, S):
        '''
        W: 1
        E: (2, 2)
        dW_dS: (2, 2)
        '''
        return self.lamb * np.trace(S) * np.eye(2) + 2 * self.mu * S

    def calc_dS_dF(self, F):
        '''
        E: (2, 2)
        F: (2, 2)
        dS_dF: (2, 2, 2, 2)
        '''
        dS_dF = np.zeros((2, 2, 2, 2))
        for i in range(2):
            for j in range(2):
                for m in range(2):
                    for n in range(2):
                        if j == n:
                            dS_dF[i, j, m, n] += 0.5 * F[i, m]
                        if j == m:
                            dS_dF[i, j, m, n] += 0.5 * F[i, n]
        return dS_dF

    def calc_dF_dx(self, x_element):
        dF_dx = np.zeros((3, 2, 2, 2))
        for k in range(3):
            for l in range(2):
                for i in range(2):
                    for j in range(2):
                        dF_dx[k, l, i, j] = self.shape_function.gradient[k, j] if l == i else 0.0
        return dF_dx

    def calc_dW_dx(self, x):
        F = self.calc_F_from_x(x)
        S = self.calc_S_from_F(F)
        dW_dS = self.calc_dW_dS(S)
        dS_dF = self.calc_dS_dF(F)
        dF_dx = self.calc_dF_dx(x)
        return np.einsum('mn,ijmn,klij->kl', dW_dS, dS_dF, dF_dx)

    def check_gradients(self):
        check_gradient(self.calc_F_from_x, self.calc_dF_dx, (3, 2))
        check_gradient(self.calc_S_from_F, self.calc_dS_dF, (2, 2))
        check_gradient(self.calc_W_from_S, self.calc_dW_dS, (2, 2))
        check_gradient(self.calc_W_from_x, self.calc_dW_dx, (3, 2))
        print("Gradient checks completed")

    def Enu_to_Lame(self, E, nu):
        mu = E / (2 * (1 + nu))
        lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, lamb

    def Lame_to_Enu(self, mu, lamb):
        E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
        nu = lamb / (2 * (lamb + mu))
        return E, nu

