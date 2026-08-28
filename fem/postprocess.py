"""Derived fields recovered from a solved system, and how to read them off it.

A solve stores its primary field `u` and the per-element derived field its physics
recovers (a scalar solve keeps the flux `grad u`; an elastic solve keeps the stress), one
value per element: exact for P1, the element mean for P2. Two consumers want more:

- **Nodal recovery** gives one continuous value per node, the smooth field nodal output
  and P2 plotting draw. It re-evaluates the field from `u` at the nodes or quadrature
  points (`FunctionSpace.nodal_gradient`, `ElasticSolution.nodal_stress`) rather than
  averaging the per-element values, so a P2 field's variation within the element, and
  its boundary value, survive; `FunctionSpace.recover_nodal` is the per-element fallback.
- **Error estimation** jumps the field across interior edges and checks its boundary
  residual against the applied traction, sampling it at quadrature points on P2.

`DerivedField` is the one equation-specific seam both share: it names which stored field
is the recoverable flux for a given physics, and how that flux behaves on a boundary edge.
Poisson's is the gradient; elasticity's is the stress. The operator names it
(`Form.derived_field`), so an estimator reads it off `problem.operator`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from fem.elements import ElementGeometry
    from fem.forms import RecoversElasticFields
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
        '''(n_elements, n_components, spatial_dim) per-element flux, read off `solution`.'''
        ...

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, n_qp, n_components, spatial_dim) flux at `geometry`'s quadrature points.

        Recomputed from `solution.u` at each point rather than read from the stored
        per-element value, so a P2 flux keeps its variation within the element.
        '''
        ...

    def divergence(self, solution: FieldSolution) -> FloatArray:
        '''(n_elements, n_components) divergence of the flux, constant per straight element.

        The strong-form interior residual is `source + div(flux)`: `f + laplacian(u)`
        for Poisson, `b + div(sigma)` for elasticity. Zero for P1 (a constant flux);
        a real term for P2.
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

    `form` is the elastic form that recovers the stress, needed by `sample` to recompute
    it from `u` at quadrature points; `evaluate` and `boundary_residual` read the stored
    state and work without it. `divergence` is the small-strain Navier operator, so it
    needs a `LinearElasticForm` and refuses a finite-strain form.
    '''

    def __init__(self, form: 'RecoversElasticFields | None' = None) -> None:
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
                'build it through the form\'s derived_field'
            )
        space = solution.space
        u_elements = solution.u.reshape(-1, space.n_components)[space.element_nodes]
        # The in-plane block: the estimators jump and recover the in-plane stress and
        # have no use for the out-of-plane lift.
        d = geometry.reference_dim
        return self.form.sample(geometry, u_elements).stress[:, :, :d, :d]

    def divergence(self, solution: FieldSolution) -> FloatArray:
        from fem.forms import LinearElasticForm
        from fem.materials import Enu_to_Lame
        from fem.solution import ElasticSolution
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the elastic flux needs a displacement solution; got a bare FieldSolution'
            )
        if not isinstance(self.form, LinearElasticForm):
            raise NotImplementedError(
                'the stress divergence is the small-strain Navier operator, which needs a '
                f'LinearElasticForm; got {type(self.form).__name__}. The recovery '
                'estimator needs no divergence and works on any elastic form.'
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
