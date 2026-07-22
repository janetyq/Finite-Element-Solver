import logging

import numpy as np

from fem.boundary import BoundaryConditions
from fem.energies import StVenantKirchhoffEnergyDensity
from fem.forms import EnergyForm
from fem.mesh.mesh import Mesh
from fem.solution import Solution
from fem.solver import Equation, LinearElastic
from fem.space import FunctionSpace
from fem.typing import DofVector, FloatArray, Matrix

logger = logging.getLogger(__name__)

class EnergySolver:
    def __init__(
        self,
        mesh: Mesh,
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions,
        verbose: bool = True,
    ) -> None:
        assert isinstance(equation, LinearElastic), "EnergySolver only supports linear elastic equation"
        # note: does not exactly match linear elastic solve for larger deformations bc doesn't use small strain approx

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.space = FunctionSpace(mesh, n_components=self.n_components)
        self.solution = Solution(mesh, self.n_components)
        # Not an assert: this is a real capability boundary (the energy densities
        # are 2D-only), so it must hold even under `python -O`.
        #
        # Checked against the mesh, not the component count: those are different
        # quantities and only the mesh knows this one. The guard used to compare the
        # component count against 2, which for LinearElastic is unconditionally 2, so
        # it never fired -- a tet mesh reached StVenantKirchhoffEnergyDensity.set_grad_u
        # and failed there on its (3, 2) gradient instead.
        if mesh.spatial_dim != 2:
            raise NotImplementedError(
                f'EnergySolver only supports 2D for now '
                f'(got a mesh with spatial_dim={mesh.spatial_dim})'
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

        self.resolved_bc = self.boundary_conditions.resolve(mesh, self.n_components)
        self.free = self.resolved_bc.free_idxs
        self.fixed = self.resolved_bc.fixed_idxs
        self.fixed_values = self.resolved_bc.fixed_values

        self.energy_density = self._select_energy(equation)
        self.form = EnergyForm(self.energy_density)

        # other options, TODO: inheritance to hide these options
        self.verbose = verbose

        # TODO: flat u + bc handling weird

        # elt_idx = 10
        # check_gradient(lambda u: self.element_energy(elt_idx, u), lambda u: self.element_gradient(elt_idx, u), (3, 2))
        # check_hessian(lambda u: self.element_gradient(elt_idx, u), lambda u: self.element_hessian(elt_idx, u), (3, 2))

        # check_gradient(self.energy, self.energy_gradient, len(self.mesh.vertices)*2)
        # check_hessian(self.energy_gradient, self.energy_hessian, len(self.mesh.vertices)*2)

    def _select_energy(self, equation: Equation) -> StVenantKirchhoffEnergyDensity:
        if isinstance(equation, LinearElastic):
            return StVenantKirchhoffEnergyDensity(equation.E, equation.nu)
        else:
            raise ValueError(f"Unsupported equation type: {type(equation).__name__}")

    # Per-element quantities delegate to the EnergyForm, which owns the tensor
    # assembly. Kept as methods (rather than inlined at the call sites) so the
    # parked gradient/hessian checks in __init__ still have something to point at.
    def element_energy(self, e_idx: int, u_element: FloatArray) -> float:
        return self.form.element_energy(self.space.element_objs[e_idx], u_element)

    def element_gradient(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        return self.form.element_residual(self.space.element_objs[e_idx], u_element)

    def element_hessian(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        return self.form.element_tangent(self.space.element_objs[e_idx], u_element)

    def energy(self, u: DofVector) -> float:
        u[self.fixed] = self.fixed_values
        return self.space.total_energy(self.form, u)

    def energy_gradient(self, u: DofVector) -> DofVector:
        u[self.fixed] = self.fixed_values
        gradient = self.space.assemble_residual(self.form, u)
        gradient[self.fixed] = 0
        return gradient

    def energy_hessian(self, u: DofVector) -> Matrix:
        u[self.fixed] = self.fixed_values
        hessian = self.space.assemble_tangent(self.form, u)
        # Eliminate the constrained DOFs: zero their coupling, then put 1 on the
        # diagonal. Without the diagonal the matrix is exactly singular (rank
        # n_free, not n_dofs), so every Newton step raised LinAlgError and fell
        # through to the 1e-8 regularization -- which perturbs the free block too,
        # capping accuracy around 1e-8 for no reason.
        hessian[self.fixed, :] = 0
        hessian[:, self.fixed] = 0
        hessian[self.fixed, self.fixed] = 1
        return hessian

    def solve(self, max_iters: int = 100) -> Solution:
        u = np.zeros(len(self.mesh.vertices) * self.n_components)
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
