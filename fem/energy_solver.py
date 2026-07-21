import logging
from typing import TYPE_CHECKING

import numpy as np

from fem.boundary import BoundaryConditions
from fem.energies import LinearElasticEnergyDensity
from fem.solution import Solution
from fem.solver import Equation, LinearElastic
from fem.typing import DofVector, FloatArray, Matrix

if TYPE_CHECKING:
    from fem.mesh.femesh import FEMesh

logger = logging.getLogger(__name__)

class EnergySolver:
    def __init__(
        self,
        femesh: 'FEMesh',
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions,
        verbose: bool = True,
    ) -> None:
        assert isinstance(equation, LinearElastic), "EnergySolver only supports linear elastic equation"
        # note: does not exactly match linear elastic solve for larger deformations bc doesn't use small strain approx

        self.femesh = femesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions
        self.dim = self.equation.dim
        self.solution = Solution(femesh, self.dim)
        # Not an assert: this is a real capability boundary (the energy densities
        # are 2D-only), so it must hold even under `python -O`.
        #
        # Checked against the mesh, not self.dim: LinearElastic.dim is a ClassVar
        # of 2, so the old `self.dim != 2` could never fire and a tet mesh reached
        # LinearElasticEnergyDensity.set_grad_u before failing on its (3, 2) gradient.
        if femesh.spatial_dim != 2:
            raise NotImplementedError(
                f'EnergySolver only supports 2D for now '
                f'(got a mesh with spatial_dim={femesh.spatial_dim})'
            )
        # This solver minimizes the internal elastic energy and never builds a
        # load vector, so a source term would be accepted and then quietly
        # ignored -- the answer would just be the unforced one.
        if equation.source is not None:
            raise NotImplementedError(
                'EnergySolver does not support a source term yet: it minimizes the '
                'internal energy only, with no external work term, so the source '
                'would be silently dropped. Use Solver for forced problems.'
            )

        self.resolved_bc = self.boundary_conditions.resolve(femesh, self.dim)
        self.free = self.resolved_bc.free_idxs
        self.fixed = self.resolved_bc.fixed_idxs
        self.fixed_values = self.resolved_bc.fixed_values

        self.energy_density = self._select_energy(equation)

        # other options, TODO: inheritance to hide these options
        self.verbose = verbose

        # TODO: flat u + bc handling weird

        # elt_idx = 10
        # check_gradient(lambda u: self.element_energy(elt_idx, u), lambda u: self.element_gradient(elt_idx, u), (3, 2))
        # check_hessian(lambda u: self.element_gradient(elt_idx, u), lambda u: self.element_hessian(elt_idx, u), (3, 2))

        # check_gradient(self.energy, self.energy_gradient, len(self.femesh.vertices)*2)
        # check_hessian(self.energy_gradient, self.energy_hessian, len(self.femesh.vertices)*2)

    def _select_energy(self, equation: Equation) -> LinearElasticEnergyDensity:
        if isinstance(equation, LinearElastic):
            return LinearElasticEnergyDensity(equation.E, equation.nu)
        else:
            raise ValueError(f"Unsupported equation type: {type(equation).__name__}")

    def element_energy(self, e_idx: int, u_element: FloatArray) -> float:
        grad_u_element = self.femesh.element_objs[e_idx].calculate_gradient(u_element)
        self.energy_density.set_grad_u(grad_u_element)
        return self.energy_density.W

    def element_gradient(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        grad_u_element = self.femesh.element_objs[e_idx].calculate_gradient(u_element)
        self.energy_density.set_grad_u(grad_u_element)
        dW_dF = self.energy_density.dW_dF
        dF_dx = self.femesh.element_objs[e_idx].dF_dx
        dW_dx = np.einsum('ij,ijmn->mn', dW_dF, dF_dx)
        return dW_dx

    def element_hessian(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        # d2W_dx2 = dW_dS @ (d2S_dF2 @ dF_dx @ dF_dx) + d2W_dS2 @ (dS_dx @ dS_dx)
        grad_u_element = self.femesh.element_objs[e_idx].grad_phi.T @ u_element
        self.energy_density.set_grad_u(grad_u_element)
        d2W_dS2 = self.energy_density.d2W_dS2
        d2S_dF2 = self.energy_density.d2S_dF2
        dW_dS = self.energy_density.dW_dS
        dS_dF = self.energy_density.dS_dF
        dF_dx = self.femesh.element_objs[e_idx].dF_dx
        dS_dx = np.einsum('klij,ijmn->klmn', dS_dF, dF_dx)
        term1 = np.einsum('abcdij,ijmn->abcdmn', d2S_dF2, dF_dx)
        term1 = np.einsum('abijcd,ijmn->abcdmn', term1, dF_dx)
        term1 = np.einsum('ij...,ij...->...', dW_dS, term1)
        term2 = np.einsum('klij,ijmn->klmn', d2W_dS2, dS_dx)
        term2 = np.einsum('ijkl,ijmn->klmn', term2, dS_dx)
        return term1 + term2

    def energy(self, u: DofVector) -> float:
        u[self.fixed] = self.fixed_values
        total = 0
        for e_idx, element in enumerate(self.femesh.elements):
            total += self.element_energy(e_idx, u.reshape(-1, self.dim)[element]) * self.femesh.element_objs[e_idx].volume
        return total

    def energy_gradient(self, u: DofVector) -> DofVector:
        u[self.fixed] = self.fixed_values
        total_energy_gradient = np.zeros((len(self.femesh.vertices), self.dim))
        for e_idx, element in enumerate(self.femesh.elements):
            total_energy_gradient[element] += self.element_gradient(e_idx, u.reshape(-1, self.dim)[element]) * self.femesh.element_objs[e_idx].volume
        total_energy_gradient = total_energy_gradient.flatten()
        total_energy_gradient[self.fixed] = 0
        return total_energy_gradient

    def energy_hessian(self, u: DofVector) -> Matrix: #TODO: not implemented
        u[self.fixed] = self.fixed_values
        n = len(self.femesh.vertices)
        total_energy_hessian = np.zeros((n, self.dim, n, self.dim))
        for e_idx, element in enumerate(self.femesh.elements):
            ix = np.ix_(element, range(self.dim), element, range(self.dim))
            total_energy_hessian[ix] += self.element_hessian(e_idx, u.reshape(-1, self.dim)[element]) * self.femesh.element_objs[e_idx].volume
        total_energy_hessian = total_energy_hessian.reshape(n*self.dim, n*self.dim)
        total_energy_hessian[self.fixed, :] = 0
        total_energy_hessian[:, self.fixed] = 0
        return total_energy_hessian

    def solve(self, max_iters: int = 100) -> Solution:
        u = np.zeros(len(self.femesh.vertices) * self.dim)
        u[self.fixed] = self.fixed_values
        logger.info("Initial energy: %s", self.energy(u))
        u = self.newton_solve(u)
        self.solution.set_values("u", u)
        return self.solution

    def newton_solve(self, u: DofVector, max_iters: int = 100) -> DofVector:
        for iter in range(max_iters):
            if self.verbose:
                logger.info("%d %s", iter, self.energy(u))
            gradient = self.energy_gradient(u)
            hessian = self.energy_hessian(u)
            try:
                newton_step = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                logger.warning("Singular hessian, adding regularization")
                newton_step = np.linalg.solve(hessian + 1e-8 * np.eye(hessian.shape[0]), -gradient)
            if np.linalg.norm(newton_step) < 1e-6:
                break
            u += newton_step
            
        return u
