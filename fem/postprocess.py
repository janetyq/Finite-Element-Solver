"""Derived fields recovered from a solved system, and how to read them off it.

A solve stores its primary field `u` and the per-element derived field its physics
recovers (a scalar solve keeps the flux `grad u`; an elastic solve keeps the stress).
Those per-element fields are element-constant, so they are discontinuous across element
boundaries. Two consumers want them:

- **Nodal recovery** turns a per-element field into one continuous value per node, the
  volume-weighted average `FunctionSpace.recover_nodal` builds. That is the smooth field
  nodal output and P2 plotting draw, and the Zienkiewicz-Zhu recovery the error estimator
  measures against.
- **Error estimation** jumps the field across interior edges and checks its boundary
  residual against the applied traction.

`DerivedField` is the one equation-specific seam both share: it names which stored field
is the recoverable flux for a given physics, and how that flux behaves on a boundary edge.
Poisson's is the gradient; elasticity's is the stress. `Equation.derived_field` returns it,
the post-processing analogue of `Equation.operator`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from fem.elements import ElementGeometry
    from fem.forms import LinearElasticForm
    from fem.solution import FieldSolution
    from fem.typing import BoolArray, FloatArray


@runtime_checkable
class DerivedField(Protocol):
    '''The recoverable flux a physics reads off a solved system.

    `evaluate` returns it as `(n_elements, n_components, spatial_dim)`, reading the
    per-element field the solve already stored: the trailing spatial axis contracts
    against an edge normal to give the traction the residual estimator jumps, and the
    whole array is what the recovery estimator and nodal output smooth. Poisson's gradient
    is `(n_el, 1, d)`; a component-major stress is `(n_el, d, d)`, so one shape serves both.
    '''

    def evaluate(self, solution: FieldSolution) -> FloatArray:
        '''(n_elements, n_components, spatial_dim) element-constant flux, read off `solution`.'''
        ...

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, n_qp, n_components, spatial_dim) flux at `geometry`'s quadrature points.

        The spatially-resolved flux, recomputed from `solution.u` at each quadrature point
        rather than read as the one stored per-element value `evaluate` returns. A P1 flux
        is element-constant, so it repeats across the points; a P2 flux varies linearly
        within the element, and the recovery estimator needs that variation to measure a
        higher-order solution's error. `geometry` is the space's geometry at whatever rule
        the estimator chose.
        '''
        ...

    def divergence(self, solution: FieldSolution) -> FloatArray:
        '''(n_elements, n_components) divergence of the flux, constant per straight element.

        The strong-form interior residual is `source + div(flux)`: for Poisson,
        `f + div(grad u) = f + laplacian(u)`; for elasticity, `b + div(sigma)`. It is
        identically zero for a P1 element (a constant flux has no divergence), which is
        why the P1 residual estimator drops it; a P2 flux varies, so div is a real term.
        '''
        ...

    def boundary_residual(
        self, flux_e0: FloatArray, outward_normal: FloatArray,
        neumann: FloatArray, free: BoolArray,
    ) -> float:
        '''The squared boundary residual on one boundary edge, 0 where there is none.

        `flux_e0` is this field's value on the boundary element (its `(k, d)` slice),
        so `flux_e0 @ outward_normal` is the discrete traction; `neumann` is the applied
        load averaged over the edge; `free` masks the components with a live test function
        there (a direction fixed at both endpoints carries a reaction, not a residual).
        Physics only: the estimator applies the `h_K * edge_length` weight.
        '''
        ...


class GradientField:
    '''The scalar diffusion flux: the stored gradient `grad u`, and no boundary term.

    Poisson's estimator jumps the normal gradient across interior edges; its boundary
    edges carry no residual (the homogeneous natural condition it is posed with), so the
    boundary residual is identically zero.
    '''

    def evaluate(self, solution: FieldSolution) -> FloatArray:
        from fem.solution import ScalarFieldSolution
        if not isinstance(solution, ScalarFieldSolution):
            raise TypeError(
                'the diffusion flux needs a scalar solution carrying grad u; '
                f'got {type(solution).__name__}'
            )
        return solution.flux[:, None, :]            # (n_el, 1, d)

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        from fem.solution import ScalarFieldSolution
        if not isinstance(solution, ScalarFieldSolution):
            raise TypeError(
                'the diffusion flux needs a scalar solution carrying grad u; '
                f'got {type(solution).__name__}'
            )
        u_elements = solution.u[solution.space.element_nodes]   # (n_el, N)
        grad = geometry.gradients(u_elements)                  # (n_el, n_qp, d)
        return grad[:, :, None, :]                             # (n_el, n_qp, 1, d)

    def divergence(self, solution: FieldSolution) -> FloatArray:
        from fem.solution import ScalarFieldSolution
        if not isinstance(solution, ScalarFieldSolution):
            raise TypeError(
                'the diffusion flux needs a scalar solution carrying grad u; '
                f'got {type(solution).__name__}'
            )
        space = solution.space
        hessian = space.element_field_hessian(solution.u[space.element_nodes])   # (n_el, d, d)
        laplacian = np.einsum('eii->e', hessian)               # div(grad u)
        return laplacian[:, None]                              # (n_el, 1)

    def boundary_residual(
        self, flux_e0: FloatArray, outward_normal: FloatArray,
        neumann: FloatArray, free: BoolArray,
    ) -> float:
        return 0.0


class StressField:
    '''The elastic flux: the stored in-plane Cauchy stress `sigma`, with a Neumann residual.

    The recovered or jumped field is the `(n_el, d, d)` in-plane stress. On a boundary edge
    the residual is `||g - sigma.n||^2` over the components with a live test function there,
    the masked term that lets a traction-free stress concentration register while a pinned
    direction's reaction traction is not counted as error.

    `form` is the small-strain elastic form, carried so `sample` can recompute the stress at
    quadrature points for the P2 recovery estimator. `evaluate` and `boundary_residual` read
    the state the solve already stored, so they need no form, and it stays optional.
    '''

    def __init__(self, form: 'LinearElasticForm | None' = None) -> None:
        self.form = form

    def evaluate(self, solution: FieldSolution) -> FloatArray:
        from fem.solution import ElasticSolution
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the elastic flux needs recovered stress; got a bare FieldSolution'
            )
        return solution.stress[:, :2, :2]           # (n_el, d, d)

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        from fem.solution import ElasticSolution
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the elastic flux needs a displacement solution; got a bare FieldSolution'
            )
        if self.form is None:
            raise ValueError(
                'StressField needs its elastic form to sample stress at quadrature points; '
                'build it through Equation.derived_field'
            )
        space = solution.space
        u_elements = solution.u.reshape(-1, space.n_components)[space.element_nodes]
        return self.form.stress_field(geometry, u_elements)   # (n_el, n_qp, d, d)

    def divergence(self, solution: FieldSolution) -> FloatArray:
        from fem.materials import Enu_to_Lame
        from fem.solution import ElasticSolution
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the elastic flux needs a displacement solution; got a bare FieldSolution'
            )
        if self.form is None:
            raise ValueError(
                'StressField needs its elastic form to take the stress divergence; '
                'build it through Equation.derived_field'
            )
        space = solution.space
        u_elements = solution.u.reshape(-1, space.n_components)[space.element_nodes]
        hessian = space.element_field_hessian(u_elements)      # (n_el, d, d, n_comp)
        # Navier form of div(sigma): (lambda + mu) grad(div u) + mu laplacian(u).
        # grad(div u)_i = sum_k d2 u_k / dx_i dx_k = sum_k H[i, k, k];
        # laplacian(u)_i = sum_j H[j, j, i]. Both read off the per-component Hessian.
        mu, lamb = Enu_to_Lame(self.form.material.E, self.form.material.nu)
        grad_div = np.einsum('eikk->ei', hessian)
        laplacian = np.einsum('ejji->ei', hessian)
        return (np.asarray(lamb) + np.asarray(mu))[..., None] * grad_div + np.asarray(mu)[..., None] * laplacian

    def boundary_residual(
        self, flux_e0: FloatArray, outward_normal: FloatArray,
        neumann: FloatArray, free: BoolArray,
    ) -> float:
        if not free.any():
            return 0.0
        traction = flux_e0 @ outward_normal
        return float(np.sum(((neumann - traction) * free)**2))
