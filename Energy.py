import numpy as np
from utils.helper import check_gradient

class LinearElasticEnergyDensity: # TODO: inheritance
    '''
    Strain energy density on 2D triangular elements

    F: deformation gradient dx/dX
    S: strain tensor
    W: strain energy density
    '''
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu, self.lamb = self.Enu_to_Lame(self.E, self.nu)

    # TODO: better understand gradient order
    # Calculate grad_u -> F, S, W, dS_dF, dW_dS, dW_dF
    def set_grad_u(self, grad_u):
        self.F = np.eye(2) + grad_u
        self.S = self.calc_S_from_F(self.F)
        self.W = self.calc_W_from_S(self.S)
        self.dS_dF = self.calc_dS_dF(self.F)
        self.dW_dS = self.calc_dW_dS(self.S)
        self.dW_dF = np.einsum('mn,ijmn->ij', self.dW_dS, self.dS_dF)
    
    def calc_S_from_F(self, F):
        return 0.5 * (F.T @ F - np.eye(2))

    def calc_W_from_S(self, S):
        return 0.5 * (self.lamb * np.trace(S)**2 + 2 * self.mu * np.trace(S.T @ S))

    def calc_dS_dF(self, F):
        '''
        S: (2, 2)
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

    def calc_dW_dS(self, S):
        '''
        W: 1
        S: (2, 2)
        dW_dS: (2, 2)
        '''
        return self.lamb * np.trace(S) * np.eye(2) + 2 * self.mu * S

    def check_gradients(self):
        check_gradient(self.calc_S_from_F, self.calc_dS_dF, (2, 2))
        check_gradient(self.calc_W_from_S, self.calc_dW_dS, (2, 2))
        check_gradient(self.calc_W_from_F, self.calc_dW_dF, (2, 2))
        print("Gradient checks completed")

    def Enu_to_Lame(self, E, nu):
        mu = E / (2 * (1 + nu))
        lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, lamb

    def Lame_to_Enu(self, mu, lamb):
        E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
        nu = lamb / (2 * (lamb + mu))
        return E, nu
