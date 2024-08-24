from scipy.optimize import minimize

from Energies import *
from Solution import *

class EnergySolver:
    def __init__(self, mesh, equation, boundary_conditions, verbose=True):
        assert equation.name == "linear_elastic", "EnergySolver only supports linear elastic equation"
        # note: does not exactly match linear elastic solve for larger deformations bc doesn't use small strain approx

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions
        self.solution = Solution(mesh)

        self.boundary_conditions.do(self.mesh.vertices.shape[0], dim=2)
        self.free = self.boundary_conditions.free_idxs
        self.fixed = self.boundary_conditions.fixed_idxs
        self.fixed_values = self.boundary_conditions.fixed_values

        self.energy_density = self._select_energy(equation)

        # other options, TODO: inheritance to hide these options
        self.verbose = verbose

        # TODO: flat u + bc handling weird
        
        # elt_idx = 10
        # check_gradient(lambda u: self.element_energy(elt_idx, u), lambda u: self.element_gradient(elt_idx, u), (3, 2))
        # check_hessian(lambda u: self.element_gradient(elt_idx, u), lambda u: self.element_hessian(elt_idx, u), (3, 2))

        # check_gradient(self.energy, self.energy_gradient, len(self.mesh.vertices)*2)
        # check_hessian(self.energy_gradient, self.energy_hessian, len(self.mesh.vertices)*2)

    def _select_energy(self, equation):
        if equation.name == "linear_elastic":
            return LinearElasticEnergyDensity(equation.parameters['E'], equation.parameters['nu'])
        else:
            raise ValueError(f"Unknown equation name: {equation.name}")

    def element_energy(self, e_idx, u_element):
        grad_u_element = self.mesh.calculate_element_gradient(e_idx, u_element)
        self.energy_density.set_grad_u(grad_u_element)
        return self.energy_density.W

    def element_gradient(self, e_idx, u_element):
        grad_u_element = self.mesh.calculate_element_gradient(e_idx, u_element)
        self.energy_density.set_grad_u(grad_u_element)
        dW_dF = self.energy_density.dW_dF
        dF_dx = self.mesh.shape_functions[e_idx].dF_dx
        dW_dx = np.einsum('ij,ijmn->mn', dW_dF, dF_dx)
        return dW_dx

    def element_hessian(self, e_idx, u_element):
        # d2W_dx2 = dW_dS @ (d2S_dF2 @ dF_dx @ dF_dx) + d2W_dS2 @ (dS_dx @ dS_dx)
        grad_u_element = self.mesh.calculate_element_gradient(e_idx, u_element)
        self.energy_density.set_grad_u(grad_u_element)
        d2W_dS2 = self.energy_density.d2W_dS2
        d2S_dF2 = self.energy_density.d2S_dF2
        dW_dS = self.energy_density.dW_dS
        dS_dF = self.energy_density.dS_dF
        dF_dx = self.mesh.shape_functions[e_idx].dF_dx
        dS_dx = np.einsum('klij,ijmn->klmn', dS_dF, dF_dx)
        term1 = np.einsum('abcdij,ijmn->abcdmn', d2S_dF2, dF_dx)
        term1 = np.einsum('abijcd,ijmn->abcdmn', term1, dF_dx)
        term1 = np.einsum('ij...,ij...->...', dW_dS, term1)
        term2 = np.einsum('klij,ijmn->klmn', d2W_dS2, dS_dx)
        term2 = np.einsum('ijkl,ijmn->klmn', term2, dS_dx)
        return term1 + term2

    def energy(self, u):
        u[self.fixed] = self.fixed_values
        total = 0
        for e_idx, element in enumerate(self.mesh.elements):
            total += self.element_energy(e_idx, u.reshape(-1, 2)[element]) * self.mesh.areas[e_idx]
        return total

    def energy_gradient(self, u):
        u[self.fixed] = self.fixed_values
        total_energy_gradient = np.zeros((len(self.mesh.vertices), 2))
        for e_idx, element in enumerate(self.mesh.elements):
            total_energy_gradient[element] += self.element_gradient(e_idx, u.reshape(-1, 2)[element]) * self.mesh.areas[e_idx]
        total_energy_gradient = total_energy_gradient.flatten()
        total_energy_gradient[self.fixed] = 0
        return total_energy_gradient

    def energy_hessian(self, u): #TODO: not implemented
        u[self.fixed] = self.fixed_values
        n = len(self.mesh.vertices)
        total_energy_hessian = np.zeros((n, 2, n, 2))
        for e_idx, element in enumerate(self.mesh.elements):
            ix = np.ix_(element, range(2), element, range(2))
            total_energy_hessian[ix] += self.element_hessian(e_idx, u.reshape(-1, 2)[element]) * self.mesh.areas[e_idx]
        total_energy_hessian = total_energy_hessian.reshape(n*2, n*2)
        total_energy_hessian[self.fixed, :] = 0
        total_energy_hessian[:, self.fixed] = 0
        return total_energy_hessian

    def solve(self, max_iters=100):
        u = np.zeros(len(self.mesh.vertices) * 2)
        u[self.fixed] = self.fixed_values
        print("Initial energy:", self.energy(u))
        u = self._newton_solve(u)
        self.solution.set_values("u", u)
        return self.solution

    def _newton_solve(self, u, max_iters=100):
        for iter in range(max_iters):
            if self.verbose:
                print(f"{iter} {self.energy(u)}")
            gradient = self.energy_gradient(u)
            hessian = self.energy_hessian(u)
            try:
                newton_step = np.linalg.solve(hessian, -gradient)
            except:
                print("Singular hessian, adding regularization")
                newton_step = np.linalg.solve(hessian + 1e-8 * np.eye(hessian.shape[0]), -gradient)
            if np.linalg.norm(newton_step) < 1e-6:
                break
            u += newton_step
            
        return u
