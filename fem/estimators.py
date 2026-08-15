"""A-posteriori error estimators: per-element indicators over a solved system.

An estimator answers "where is the discrete solution least trustworthy?" as one
non-negative number per element, which an `AdaptiveRefinement` driver turns into a
refinement decision. It is an *operation on a solved system*, not data of the PDE,
so it lives here rather than on `Equation` -- the equation only *names* its physics
(`Equation.flux`), the way it names its `operator` and `energy_density`.

Two families share one seam and one physics hook:

- **Residual** (`ResidualEstimator`): measures how badly the computed field fails
  the PDE -- an interior term (the source it does not balance), an interior-edge
  jump (the flux discontinuity between neighbours), and a boundary term (the
  applied traction the discrete flux does not match). This is a direct check of
  equilibrium and needs the mesh's edge normals, so it is 2D-only for now.

- **Recovery** (`RecoveryEstimator`): the Zienkiewicz-Zhu idea. The discrete flux
  is discontinuous (element-constant for P1); a *recovered* continuous flux `sigma*`
  -- here the volume-weighted nodal average `FunctionSpace.element_to_vertex` builds
  -- is much closer to the exact flux, so `eta_K = ||sigma* - sigma_h||_K` measures
  the error. It reads no edge normals, so it is dimension-general (validated in 2D).

The one equation-specific input, shared by both, is the **`Flux`**: *which* field
to jump/recover (Poisson's gradient, elasticity's stress) and -- residual-only --
what the boundary residual is. Everything else is neutral machinery, exactly the
`Form`/`assemble` split the rest of the package uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from fem.regions import evaluate_field
from fem.solution import ElasticSolution

if TYPE_CHECKING:
    from fem.adaptivity import RefinableSolver
    from fem.boundary import ResolvedBC
    from fem.equations import Equation
    from fem.solution import FieldSolution
    from fem.space import FunctionSpace
    from fem.typing import BoolArray, ElementField, FieldValue, FloatArray


# -- the solved-system view the flux hooks read -------------------------------


@dataclass(frozen=True)
class Solved:
    '''The resolved view of a solved system a `Flux` reads: space, solution, BCs.

    Built once per `estimate` and handed to the flux hooks so neither the engine
    nor the flux re-derives the DOF partition. `is_fixed[v, c]` marks vertex `v`'s
    component `c` as Dirichlet-constrained -- the mask a boundary residual uses to
    ignore a pinned direction's reaction traction.
    '''
    space: FunctionSpace
    solution: FieldSolution
    resolved: ResolvedBC
    is_fixed: BoolArray   # (n_vertices, n_components)


def _solved(solver: RefinableSolver) -> Solved:
    '''Resolve `solver`'s latest solve into the view the flux hooks read.

    Raises if the solver has not solved yet -- the single guard both estimators
    lean on, so `estimate` never has to narrow the solution itself.
    '''
    solution = solver.solution
    if solution is None:
        raise ValueError('the error estimator requires a solved system')
    space = solver.space
    resolved = solver.boundary_conditions.resolve(space.nodes, space.n_components)
    is_fixed = np.zeros((len(space.mesh.vertices), space.n_components), dtype=bool)
    is_fixed.ravel()[resolved.fixed_idxs] = True
    return Solved(space, solution, resolved, is_fixed)


# -- the one equation-specific concept: which flux to jump / recover ----------


@runtime_checkable
class Flux(Protocol):
    '''The physical flux an estimator jumps or recovers, from a solved system.

    The only equation-specific piece, and it is shared by both estimator families.
    `evaluate` returns the flux as `(n_elements, n_components, spatial_dim)`: the
    trailing spatial axis contracts against an edge normal to give the traction
    whose jump the residual estimator measures, and the whole array is the field
    the recovery estimator smooths. Poisson's gradient is `(n_el, 1, d)`; a
    component-major stress is `(n_el, d, d)`, so one code path serves both.
    '''

    def evaluate(self, ctx: Solved) -> FloatArray:
        '''(n_elements, n_components, spatial_dim) element-constant flux at the state.'''
        ...

    def boundary_residual(
        self, ctx: Solved, flux: FloatArray, v0: int, v1: int, e0: int,
        outward_normal: FloatArray,
    ) -> float:
        '''The squared boundary residual on one boundary edge -- 0 where there is none.

        Physics only: the engine applies the `h_K * edge_length` weight. The edge
        runs between vertices `v0`, `v1` on element `e0`, with `outward_normal`
        already oriented out of the domain; `flux` is this estimator's own
        `evaluate` result, so `flux[e0] @ outward_normal` is the discrete traction.
        '''
        ...

    def error_density(self, diff: FloatArray) -> FloatArray:
        '''Pointwise squared norm of a recovered-minus-discrete flux difference.

        `diff` is `(n_elements, n_qp, n_components, spatial_dim)`; the result is
        `(n_elements, n_qp)`, integrated by the recovery estimator against the
        quadrature weights.
        '''
        ...


@dataclass(frozen=True)
class GradientFlux:
    '''The scalar diffusion flux: the field gradient `grad u`, and no boundary term.

    Poisson's estimator jumps the normal gradient across interior edges; its
    boundary edges carry no residual (the natural condition it was posed with is
    homogeneous), so `boundary_residual` is identically zero.
    '''

    def evaluate(self, ctx: Solved) -> FloatArray:
        grad = ctx.space.gradient(ctx.solution.u)   # (n_el, d)
        return grad[:, None, :]                     # (n_el, 1, d)

    def boundary_residual(
        self, ctx: Solved, flux: FloatArray, v0: int, v1: int, e0: int,
        outward_normal: FloatArray,
    ) -> float:
        return 0.0

    def error_density(self, diff: FloatArray) -> FloatArray:
        return np.sum(diff**2, axis=(-1, -2))


@dataclass(frozen=True)
class StressFlux:
    '''The elastic flux: the in-plane Cauchy stress `sigma`, with a Neumann residual.

    The recovered/ jumped field is the `(n_el, d, d)` in-plane stress. On a boundary
    edge the residual is `||g - sigma.n||^2` over the components with a live test
    function there -- a component fixed at *both* endpoints has no free hat and is
    dropped, so a pinned direction's reaction traction is not counted as error while
    a roller's free direction still is. This is the masked-norm boundary term the
    stress-concentration estimator relies on.
    '''

    def evaluate(self, ctx: Solved) -> FloatArray:
        solution = ctx.solution
        if not isinstance(solution, ElasticSolution):
            raise TypeError(
                'the error estimator needs recovered stress; got a bare FieldSolution'
            )
        return solution.stress[:, :2, :2]           # (n_el, d, d)

    def boundary_residual(
        self, ctx: Solved, flux: FloatArray, v0: int, v1: int, e0: int,
        outward_normal: FloatArray,
    ) -> float:
        # Free iff *either* endpoint's hat is free -- fixed at both is what removes
        # the component from the assembled system, and only then is its traction a
        # reaction rather than a residual.
        free = ~(ctx.is_fixed[v0] & ctx.is_fixed[v1])
        if not free.any():
            return 0.0
        g = 0.5 * (ctx.resolved.neumann_load[v0] + ctx.resolved.neumann_load[v1])
        t = flux[e0] @ outward_normal
        return float(np.sum(((g - t) * free)**2))

    def error_density(self, diff: FloatArray) -> FloatArray:
        return np.sum(diff**2, axis=(-1, -2))


# -- the outer seam every estimator satisfies ---------------------------------


@runtime_checkable
class ErrorEstimator(Protocol):
    '''A per-element error indicator over a solved system.

    The single method `AdaptiveRefinement` drives. Residual and recovery
    estimators implement it, and so would a future goal-oriented one, though they
    share no internals -- which is why this is the seam the driver depends on.
    '''

    def estimate(self, solver: RefinableSolver) -> ElementField:
        '''(n_elements,) non-negative error indicator for `solver`'s latest solve.'''
        ...


def _rotate90(edge_vec: FloatArray) -> FloatArray:
    '''The 2D edge normal: the edge vector turned a quarter turn. Not yet oriented.'''
    return np.array([-edge_vec[1], edge_vec[0]])


@dataclass(frozen=True)
class ResidualEstimator:
    '''Residual-based estimator: interior residual + flux jump + boundary residual.

    `eta_K^2 = h_K^2 ||f||^2_K + (h_K/2) sum_edges ||[[flux.n]]||^2_e
                                + h_K sum_(bnd edges) ||boundary residual||^2_e`

    The engine owns every geometric quantity -- `h_K`, the edge normals, the
    accumulation -- and delegates the three physics pieces to the `flux`: the flux
    field it jumps, and (per boundary edge) the boundary residual. The interior
    term is the source `f` the P1 field cannot balance (`div(flux) = 0` inside a
    constant-strain element), read at the element centroid.

    2D only: the jump and boundary terms need edge normals. A 3D mesh would need
    face normals, which the recovery estimator sidesteps entirely.
    '''
    flux: Flux
    source: FieldValue = None

    def estimate(self, solver: RefinableSolver) -> ElementField:
        mesh, space = solver.mesh, solver.space
        if mesh.spatial_dim != 2:
            raise NotImplementedError('the residual error estimator needs face normals (2D only)')

        ctx = _solved(solver)            # raises if the solver has not solved
        flux = self.flux.evaluate(ctx)   # (n_el, k, d)

        h_K = mesh.element_diameters
        n_elements = len(mesh.elements)

        centroids = mesh.vertices[mesh.elements].mean(axis=1)
        f = evaluate_field(self.source, centroids, space.n_components)   # (n_el, k)
        interior = h_K**2 * np.sum(f**2, axis=1) * space.element_volumes

        jump_term = np.zeros(n_elements)
        boundary_term = np.zeros(n_elements)

        for edge, adjacent in mesh.edge_to_elements.items():
            v0, v1 = edge
            edge_vec = mesh.vertices[v1] - mesh.vertices[v0]
            edge_len = float(np.linalg.norm(edge_vec))
            normal = _rotate90(edge_vec) / edge_len

            if len(adjacent) == 2:
                # Interior edge: the flux is continuous in the true solution but
                # jumps between the piecewise-constant discrete neighbours.
                e0, e1 = adjacent
                t0 = flux[e0] @ normal
                t1 = flux[e1] @ normal
                jump2 = float(np.sum((t0 - t1)**2))
                edge_contribution = edge_len * jump2
                jump_term[e0] += (h_K[e0] / 2) * edge_contribution / 2
                jump_term[e1] += (h_K[e1] / 2) * edge_contribution / 2
                continue

            # Boundary edge: orient the normal out of the domain (g is directional,
            # unlike the interior jump where either sign cancels), then delegate the
            # residual to the flux -- zero for a scalar problem.
            (e0,) = adjacent
            centroid = mesh.vertices[mesh.elements[e0]].mean(axis=0)
            midpoint = 0.5 * (mesh.vertices[v0] + mesh.vertices[v1])
            if np.dot(midpoint - centroid, normal) < 0:
                normal = -normal
            residual2 = self.flux.boundary_residual(ctx, flux, int(v0), int(v1), e0, normal)
            boundary_term[e0] += h_K[e0] * edge_len * residual2

        eta_squared = interior + jump_term + boundary_term
        return np.sqrt(np.maximum(eta_squared, 0.0))


@dataclass(frozen=True)
class RecoveryEstimator:
    '''Zienkiewicz-Zhu recovery estimator: `eta_K = ||sigma* - sigma_h||_K`.

    The discrete flux `sigma_h` is element-constant (P1) and discontinuous; the
    recovered `sigma*` is its volume-weighted nodal average, a continuous field that
    -- being superconvergent -- stands in for the unknown exact flux. Their gap,
    integrated over each element, estimates the error.

    Needs no edge normals, so unlike the residual estimator it is dimension-general;
    validated in 2D. Recovery by simple averaging is biased at boundaries and
    re-entrant corners -- it still orders elements well enough to drive refinement,
    but the effectivity there is looser (patch recovery would tighten it).
    '''
    flux: Flux

    def estimate(self, solver: RefinableSolver) -> ElementField:
        space = solver.space
        ctx = _solved(solver)                             # raises if the solver has not solved
        sigma_h = self.flux.evaluate(ctx)                 # (n_el, k, d), constant per element
        sigma_star = space.element_to_vertex(sigma_h)     # (n_nodes, k, d), continuous

        # Integrate ||sigma* - sigma_h||^2 over each element. sigma* is P1 (linear),
        # sigma_h constant, so the integrand is quadratic -- a degree-2 rule is exact.
        geometry = space.geometry_at(2)
        per_element = sigma_star[space.element_nodes]     # (n_el, N, k, d)
        sigma_star_qp = np.einsum('qn,en...->eq...', geometry.shape, per_element)
        diff = sigma_star_qp - sigma_h[:, None]           # (n_el, n_qp, k, d)
        density = self.flux.error_density(diff)           # (n_el, n_qp)
        eta_squared = np.einsum('eq,eq->e', density, geometry.weight_detJ)
        return np.sqrt(np.maximum(eta_squared, 0.0))


# -- factories: build an estimator from an equation's named flux --------------


def residual_estimator(equation: Equation) -> ResidualEstimator:
    '''The residual estimator for `equation`, from its flux and source.'''
    return ResidualEstimator(equation.flux(), equation.source)


def recovery_estimator(equation: Equation) -> RecoveryEstimator:
    '''The Zienkiewicz-Zhu recovery estimator for `equation`, from its flux.'''
    return RecoveryEstimator(equation.flux())
