"""Derived fields recovered from a solved system, and how to read them off it.

A solve stores its primary field `u` and the per-element derived field its physics
recovers (a scalar solve keeps the gradient `grad u`; an elastic solve keeps the stress), one
value per element: exact for P1, the element mean for P2. Two consumers want more:

- **Nodal recovery** gives one continuous value per node, the smooth field nodal output
  and P2 plotting draw. It re-evaluates the field from `u` at the nodes or quadrature
  points (`DiffusionSolution.nodal_gradient`, `ElasticSolution.nodal_stress`) rather than
  averaging the per-element values, so a P2 field's variation within the element, and
  its boundary value, survive; `fem.post.recovery.recover_nodal` is the per-element fallback.
- **Error estimation** jumps the field across interior edges and checks its boundary
  residual against the applied traction, sampling it at quadrature points on P2.

`Flux` is the one equation-specific seam both share: it names the recoverable flux for
a given physics, how it is read off the solution, and how it behaves on a boundary edge.
Poisson's is the diffusive flux `kappa grad u`; elasticity's is the stress. The operator names it
(`Form.flux`), so an estimator reads it off `problem.operator`.

`StressFlux.divergence` imports `fem.physics.forms` and `fem.physics.materials` lazily:
`forms` imports this module at top level, so the reverse edge stays function-local.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from fem.post.solution import ElasticSolution, DiffusionSolution

if TYPE_CHECKING:
    from fem.elements import ElementGeometry
    from fem.physics.forms import RecoversElasticState
    from fem.post.solution import FieldSolution
    from fem.typing import BoolArray, FieldValue, FloatArray


@runtime_checkable
class Flux(Protocol):
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

        Recomputed from `solution.dofs` at each point rather than read from the stored
        per-element value, so a P2 flux keeps its variation within the element.
        '''
        ...

    def divergence(self, solution: FieldSolution) -> FloatArray:
        '''(n_elements, n_components) divergence of the flux, constant per straight element.

        The strong-form interior residual is `source + div(flux)`: `f + div(kappa grad u)`
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


class GradientFlux:
    '''The diffusive flux `kappa grad u` of `DiffusionForm`, with a Neumann residual.

    `coefficient` is the form's kappa, a constant or a callable of position, so a
    varying conductivity enters the jump, the interior residual, and the boundary
    residual the way it enters the operator. The stored `gradient` is `grad u` alone;
    kappa is applied here, at the centroid (`evaluate`), at the geometry's points
    (`sample`), or through its nodal interpolant (`divergence`).

    On a boundary edge the residual is `(g - kappa du/dn)^2` where the test function is
    live, so a Neumann value the discrete flux misses registers; a Dirichlet edge carries
    a reaction, not a residual. `divergence` expands `div(kappa grad u)` as
    `kappa laplacian(u) + grad(kappa) . grad(u)`, reading `grad(kappa)` off the nodal
    interpolant of kappa on the space, exact for an affine coefficient.
    '''

    def __init__(self, coefficient: FieldValue = 1.0) -> None:
        self.coefficient = coefficient

    @staticmethod
    def _diffusion(solution: FieldSolution) -> DiffusionSolution:
        if not isinstance(solution, DiffusionSolution):
            raise TypeError(
                'the diffusion flux needs a scalar solution carrying grad u; '
                f'got {type(solution).__name__}'
            )
        return solution

    def _kappa_at(self, points: FloatArray) -> FloatArray:
        '''kappa at each of `points`, `(n_points,)`.'''
        from fem.regions import evaluate_field
        return evaluate_field(self.coefficient, points, 1)[:, 0]

    def evaluate(self, solution: FieldSolution) -> FloatArray:
        solution = self._diffusion(solution)
        kappa = self._kappa_at(solution.mesh.centroids)                # (n_el,)
        return (kappa[:, None] * solution.gradient)[:, None, :]       # (n_el, 1, d)

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        solution = self._diffusion(solution)
        u_elements = solution.dofs[solution.space.element_nodes]   # (n_el, N)
        grad = geometry.gradients(u_elements)                  # (n_el, n_qp, d)
        n_el, n_qp = geometry.weight_detJ.shape
        points = geometry.points.reshape(n_el * n_qp, geometry.spatial_dim)
        kappa = self._kappa_at(points).reshape(n_el, n_qp)     # (n_el, n_qp)
        return (kappa[..., None] * grad)[:, :, None, :]        # (n_el, n_qp, 1, d)

    def divergence(self, solution: FieldSolution) -> FloatArray:
        solution = self._diffusion(solution)
        space = solution.space
        hessian = space.element_hessian(solution.dofs[space.element_nodes])   # (n_el, d, d)
        laplacian = np.einsum('eii->e', hessian)               # (n_el,)
        kappa = self._kappa_at(solution.mesh.centroids)         # (n_el,)
        grad_kappa = space.gradient(space.interpolate(self.coefficient))   # (n_el, d)
        divergence = kappa * laplacian + np.einsum('ed,ed->e', grad_kappa, solution.gradient)
        return divergence[:, None]                             # (n_el, 1)

    def boundary_residual(
        self, flux_e0: FloatArray, outward_normal: FloatArray,
        neumann: FloatArray, free: BoolArray,
    ) -> float:
        if not free.any():
            return 0.0
        normal_flux = flux_e0 @ outward_normal                 # (1,)
        return float(np.sum(((neumann - normal_flux) * free)**2))


class ScaledFlux:
    '''`factor * flux`: the flux of a `ScaledForm`, every reading scaled with the operator.

    `c * DiffusionForm(kappa)` has the flux `c kappa grad u`, and the wave operator is
    written that way (`c^2` times the Laplacian), so the scaled form's flux is its term's
    scaled by the same factor. The boundary residual reads the already-scaled `flux_e0`,
    so it delegates unchanged.
    '''

    def __init__(self, factor: float, flux: Flux) -> None:
        self.factor = factor
        self.flux = flux

    def evaluate(self, solution: FieldSolution) -> FloatArray:
        return self.factor * self.flux.evaluate(solution)

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        return self.factor * self.flux.sample(solution, geometry)

    def divergence(self, solution: FieldSolution) -> FloatArray:
        return self.factor * self.flux.divergence(solution)

    def boundary_residual(
        self, flux_e0: FloatArray, outward_normal: FloatArray,
        neumann: FloatArray, free: BoolArray,
    ) -> float:
        return self.flux.boundary_residual(flux_e0, outward_normal, neumann, free)


class StressFlux:
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

    def __init__(self, form: 'RecoversElasticState | None' = None) -> None:
        self.form = form

    def evaluate(self, solution: FieldSolution) -> FloatArray:
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the elastic flux needs recovered stress; got a bare FieldSolution'
            )
        return solution.stress[:, :2, :2]           # (n_el, d, d)

    def sample(self, solution: FieldSolution, geometry: ElementGeometry) -> FloatArray:
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the elastic flux needs a displacement solution; got a bare FieldSolution'
            )
        if self.form is None:
            raise ValueError(
                'StressFlux needs its elastic form to sample stress at quadrature points; '
                'build it through the form\'s flux'
            )
        space = solution.space
        u_elements = solution.dofs.reshape(-1, space.n_components)[space.element_nodes]
        # The in-plane block: the estimators jump and recover the in-plane stress and
        # have no use for the out-of-plane lift.
        d = geometry.reference_dim
        return self.form.sample(geometry, u_elements).stress[:, :, :d, :d]

    def divergence(self, solution: FieldSolution) -> FloatArray:
        from fem.physics.forms import LinearElasticForm
        from fem.physics.materials import Enu_to_Lame
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
        u_elements = solution.dofs.reshape(-1, space.n_components)[space.element_nodes]
        hessian = space.element_hessian(u_elements)      # (n_el, d, d, n_comp)
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
