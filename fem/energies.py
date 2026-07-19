import logging

import numpy as np
from fem.materials import Enu_to_Lame
from fem.numerics import check_gradient

logger = logging.getLogger(__name__)

class LinearElasticEnergyDensity: # TODO: inheritance
    '''
    Strain energy density on 2D triangular elements

    F: deformation gradient dx/dX (2, 2)
    S: strain tensor (2, 2)
    W: strain energy density (1)
    '''
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu, self.lamb = Enu_to_Lame(self.E, self.nu)

    # Calculate grad_u -> F, S, W, dS_dF, dW_dS, dW_dF
    def set_grad_u(self, grad_u):
        self.F = np.eye(2) + grad_u
        self.S = self.calculate_S_from_F(self.F)
        self.W = self.calculate_W_from_S(self.S)
        self.dS_dF = self.calculate_dS_dF(self.F)
        self.dW_dS = self.calculate_dW_dS(self.S)
        self.dW_dF = np.einsum('ij,ijmn->mn', self.dW_dS, self.dS_dF)
        self.d2S_dF2 = self.calculate_d2S_dF2(self.F)
        self.d2W_dS2 = self.calculate_d2W_dS2(self.S)
    
    def calculate_S_from_F(self, F):
        # note: this is not the linear approx of infinitesimal strain theory,
        #       so iterative solver does not converge in 1 iteration
        return 0.5 * (F.T @ F - np.eye(2))

    def calculate_W_from_S(self, S):
        return 0.5 * (self.lamb * np.trace(S)**2 + 2 * self.mu * np.trace(S.T @ S))

    def calculate_dS_dF(self, F):
        dS_dF = np.zeros((2, 2, 2, 2))
        for i in range(2):
            for j in range(2):
                for m in range(2):
                    for n in range(2):
                        if j == n:
                            dS_dF[i, j, m, n] += 0.5 * F[m, i]
                        if i == n:
                            dS_dF[i, j, m, n] += 0.5 * F[m, j]
        return dS_dF

    def calculate_d2S_dF2(self, F):
        d2S_dF2 = np.zeros((2, 2, 2, 2, 2, 2))
        for i in range(2):
            for j in range(2):
                for m in range(2):
                    for n in range(2):
                        for k in range(2):
                            for q in range(2):
                                if j == n and k == m and i == q:
                                    d2S_dF2[i, j, m, n, k, q] += 0.5
                                if i == n and k == m and j == q:
                                    d2S_dF2[i, j, m, n, k, q] += 0.5
        return d2S_dF2

    def calculate_dW_dS(self, S):
        return self.lamb * np.trace(S) * np.eye(2) + 2 * self.mu * S

    def calculate_W_from_F(self, F):
        return self.calculate_W_from_S(self.calculate_S_from_F(F))

    def calculate_dW_dF(self, F):
        S = self.calculate_S_from_F(F)
        return np.einsum('ij,ijmn->mn', self.calculate_dW_dS(S), self.calculate_dS_dF(F))

    def calculate_d2W_dS2(self, S):
        d2W_dS2 = np.zeros((2, 2, 2, 2))
        for i in range(2):
            for j in range(2):
                for m in range(2):
                    for n in range(2):
                        if i == j and m == n:
                            d2W_dS2[i, j, m, n] += self.lamb
                        if i == m and j == n:
                            d2W_dS2[i, j, m, n] += 2 * self.mu
        return d2W_dS2

    def check_gradients(self):
        check_gradient(self.calculate_S_from_F, self.calculate_dS_dF, (2, 2))
        check_gradient(self.calculate_W_from_S, self.calculate_dW_dS, (2, 2))
        check_gradient(self.calculate_W_from_F, self.calculate_dW_dF, (2, 2))
        logger.info("Gradient checks completed")

class NeohookeanEnergyDensity:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.mu, self.lamb = Enu_to_Lame(self.E, self.nu)

    def set_grad_u(self, grad_u):
        raise NotImplementedError(
            "NeohookeanEnergyDensity is not implemented yet; "
            "use LinearElasticEnergyDensity for now."
        )