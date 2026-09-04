"""A-posteriori error estimators: per-element indicators over a solved system.

An estimator answers "where is the discrete solution least trustworthy?" as one
non-negative number per element, which `AdaptiveRefinement` turns into a refinement
decision. Three are provided:

- **Residual** (`ResidualEstimator`): how badly the computed field fails the PDE,
  through an interior term (`source + div(flux)`), an interior-edge flux jump, and a
  boundary term (applied traction minus discrete traction). Needs edge normals, so
  2D and straight-sided only.
- **Recovery** (`RecoveryEstimator`): Zienkiewicz-Zhu. The discrete flux is
  discontinuous across elements; its L2 projection onto the nodal space is much
  closer to the exact flux, so `eta_K = ||sigma* - sigma_h||_K` measures the error.
  Dimension-general and works on curved elements.
- **Goal-oriented** (`GoalOrientedEstimator`): the product of primal and dual
  recovery indicators, refining toward a quantity of interest.

An estimator has no physics of its own. The one physics-specific input, the
`Flux` (which field to jump or recover, and what its boundary residual is),
is read off the problem's operator (`Form.flux`), and the source off the
problem, at `estimate` time. A custom estimate is any callable of `(problem, solution)`.

`GoalOrientedEstimator` imports `fem.analysis.sensitivity` lazily to solve the dual;
`sensitivity` sits above the estimators, so the edge stays function-local.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from fem.numerics import scatter_add
from fem.post.recovery import project_to_nodal
from fem.regions import evaluate_field

if TYPE_CHECKING:
    from fem.conditions import ResolvedConditions
    from fem.physics.derived import Flux
    from fem.problem import Problem
    from fem.analysis.sensitivity import QuantityOfInterest
    from fem.post.solution import FieldSolution
    from fem.space import FunctionSpace
    from fem.typing import BoolArray, ElementValues, FieldValue, FloatArray, IntArray


# -- the solved-system view the flux hooks read -------------------------------


@dataclass(frozen=True)
class Solved:
    '''The resolved view of a solved system an estimator reads: space, solution, conditions.

    Built once per `estimate` so the engine does not re-derive the DOF partition per
    edge. `is_fixed[v, c]` marks vertex `v`'s component `c` as Dirichlet-constrained,
    the mask the residual estimator turns into the `free` argument of a boundary
    residual so a pinned direction's reaction traction is not counted.
    '''
    space: FunctionSpace
    solution: FieldSolution
    resolved: ResolvedConditions
    is_fixed: BoolArray   # (n_vertices, n_components)


def _solved(problem: Problem, solution: FieldSolution) -> Solved:
    '''The view the flux hooks read, from a problem and its solution.'''
    if problem.is_time_dependent:
        raise ValueError(
            'an error estimate is for one steady problem; pass the snapshot problem.at(t) '
            'the solution was solved at'
        )
    space = problem.space
    resolved = problem.resolved
    # Sized by nodes, not mesh vertices: a P2 space fixes edge-midpoint DOFs too, whose
    # indices run past the vertex count. The residual estimator reads only vertex rows,
    # so this stays a strict generalization of the P1 mask.
    is_fixed = np.zeros((space.n_nodes, space.n_components), dtype=bool)
    is_fixed.ravel()[resolved.fixed_idxs] = True
    return Solved(space, solution, resolved, is_fixed)


def _flux(problem: Problem) -> Flux:
    '''The recoverable flux the problem's operator names.'''
    flux = problem.operator.flux()
    if flux is None:
        raise NotImplementedError(
            f'{type(problem.operator).__name__} names no flux, so an error '
            'estimate is not defined for it.'
        )
    return flux


def _source(problem: Problem) -> FieldValue:
    '''The problem's volume source as a pointwise field; None for no source.'''
    # The interior residual reads the source pointwise at centroids; the Source is that
    # field wrapped as a load term.
    source = problem.source
    return None if source is None else source.field


# -- the outer seam every estimator satisfies ---------------------------------


@runtime_checkable
class ErrorEstimator(Protocol):
    '''A per-element error indicator over a solved problem: the one method
    `AdaptiveRefinement` drives.'''

    def estimate(self, problem: Problem, solution: FieldSolution) -> ElementValues:
        '''(n_elements,) non-negative error indicator for `solution` of `problem`.'''
        ...


def _rotate90(edge_vec: FloatArray) -> FloatArray:
    '''The 2D edge normal: the edge vector turned a quarter turn. Not yet oriented.'''
    return np.array([-edge_vec[1], edge_vec[0]])


@dataclass(frozen=True)
class ResidualEstimator:
    '''Residual-based estimator: interior residual + flux jump + boundary residual.

    `eta_K^2 = h_K^2 ||f + div(flux)||^2_K + (h_K/2) sum_edges ||[[flux.n]]||^2_e
                                           + h_K sum_(bnd edges) ||boundary residual||^2_e`

    The geometry (`h_K`, edge normals, accumulation) is handled here; the `flux`
    supplies the physics: the field to jump, its divergence, and the boundary residual.
    On P1 the flux is element-constant, so `div(flux)` is zero and the jump reads one
    value per element; on P2 the interior term carries the divergence and the jump is
    read at the shared edge from each side. The interior residual is read at the
    centroid and the boundary term uses the element's per-element traction, both
    exact on P1 and a light approximation on P2.

    2D only: the jump and boundary terms need edge normals. The flux and the source are
    the problem's own: its operator's derived field and its volume source.
    '''

    def estimate(self, problem: Problem, solution: FieldSolution) -> ElementValues:
        space = problem.space
        mesh = space.mesh
        flux_field = _flux(problem)
        source = _source(problem)
        if mesh.spatial_dim != 2:
            raise NotImplementedError('the residual error estimator needs face normals (2D only)')
        if space.element_type.GEOMETRY_DEGREE > 1:
            # The interior term's divergence comes from `element_hessian`, which is
            # exact only on straight elements; a curved Jacobian adds a first-derivative
            # term it omits.
            raise NotImplementedError(
                f'the residual error estimator is straight-sided only; '
                f'{space.element_type.__name__} is curved. Use RecoveryEstimator.'
            )

        ctx = _solved(problem, solution)
        flux = flux_field.evaluate(ctx.solution)  # (n_el, k, d)

        h_K = mesh.element_diameters
        n_elements = len(mesh.elements)

        # The strong-form interior residual `f + div(flux)`, read at the element centroid.
        # `div(flux)` is zero for P1 (a constant flux).
        centroids = mesh.centroids
        f = evaluate_field(source, centroids, space.n_components)   # (n_el, k)
        residual = f + flux_field.divergence(ctx.solution)          # (n_el, k)
        interior = h_K**2 * np.sum(residual**2, axis=1) * space.element_volumes

        jump_term = np.zeros(n_elements)
        boundary_term = np.zeros(n_elements)

        vertices = mesh.vertices
        edges = mesh.edges                    # (E, 2) sorted vertex pairs
        edge_elements = mesh.edge_elements    # (E, 2), -1 in slot 1 on a boundary edge
        is_interior = edge_elements[:, 1] >= 0

        # The flux each element carries on each of its edges, gathered per global edge and
        # side, so the jump is read at the shared edge from both neighbours.
        edge_side_flux = self._per_side_edge_flux(
            flux_field, space, ctx.solution, edges, edge_elements)

        # Interior edges, all at once: the flux is continuous in the true solution but
        # jumps between the discrete neighbours meeting at the edge.
        pairs = edges[is_interior]
        e0, e1 = edge_elements[is_interior, 0], edge_elements[is_interior, 1]
        edge_vecs = vertices[pairs[:, 1]] - vertices[pairs[:, 0]]              # (Ei, 2)
        edge_lens = np.linalg.norm(edge_vecs, axis=1)                         # (Ei,)
        normals = np.stack([-edge_vecs[:, 1], edge_vecs[:, 0]], axis=1) / edge_lens[:, None]
        flux_jump = edge_side_flux[is_interior, 0] - edge_side_flux[is_interior, 1]  # (Ei, k, d)
        jumps = np.einsum('ekd,ed->ek', flux_jump, normals)                  # (Ei, k)
        contribution = edge_lens * np.sum(jumps**2, axis=1)                   # (Ei,)
        jump_term += scatter_add(e0, (h_K[e0] / 2) * contribution / 2, n_elements)
        jump_term += scatter_add(e1, (h_K[e1] / 2) * contribution / 2, n_elements)

        # Boundary edges are comparatively few and the flux's boundary residual
        # is a per-edge physics hook, so these stay a Python loop.
        neumann_load = ctx.resolved.neumann_load
        for edge_idx in np.nonzero(~is_interior)[0]:
            v0, v1 = int(edges[edge_idx, 0]), int(edges[edge_idx, 1])
            e_bnd = int(edge_elements[edge_idx, 0])
            edge_vec = vertices[v1] - vertices[v0]
            edge_len = float(np.linalg.norm(edge_vec))
            normal = _rotate90(edge_vec) / edge_len
            # Orient the normal out of the domain: g is directional, unlike the
            # interior jump where either sign cancels.
            centroid = vertices[mesh.elements[e_bnd]].mean(axis=0)
            midpoint = 0.5 * (vertices[v0] + vertices[v1])
            if np.dot(midpoint - centroid, normal) < 0:
                normal = -normal
            # A component is free where either endpoint carries a live test function;
            # fixed at both, its traction is a reaction, not a residual.
            free = ~(ctx.is_fixed[v0] & ctx.is_fixed[v1])
            g = 0.5 * (neumann_load[v0] + neumann_load[v1])
            residual2 = flux_field.boundary_residual(flux[e_bnd], normal, g, free)
            boundary_term[e_bnd] += h_K[e_bnd] * edge_len * residual2

        eta_squared = interior + jump_term + boundary_term
        return np.sqrt(np.maximum(eta_squared, 0.0))

    @staticmethod
    def _per_side_edge_flux(
        flux: Flux, space: FunctionSpace, solution: FieldSolution,
        edges: IntArray, edge_elements: IntArray,
    ) -> FloatArray:
        '''(n_edges, 2, k, d) the flux each side carries at each edge's midpoint.

        For every element the flux is sampled at its three edge midpoints (through the
        space's cached `geometry_at_edge_midpoints`), then scattered into `[edge, side]` so
        an interior edge holds the value from both neighbours (side 0 is
        `edge_elements[:, 0]`). A boundary edge fills only side 0. Sampled on P1 as well:
        a variable coefficient makes the flux vary within a P1 element even though the
        gradient does not.
        '''
        n_el = len(space.element_nodes)
        edge_flux = flux.sample(solution, space.geometry_at_edge_midpoints)   # (n_el, 3, k, d)

        # The global edge each (element, slot) names: the local edge opposite corner s
        # joins the other two corners, matched into the sorted `edges` table by key.
        corners = np.asarray(space.mesh.elements)                  # (n_el, 3)
        slot_pairs = np.stack(
            [corners[:, [1, 2]], corners[:, [0, 2]], corners[:, [0, 1]]], axis=1)  # (n_el, 3, 2)
        n_vertices = len(space.mesh.vertices)
        pair_key = (slot_pairs.min(axis=2).astype(np.int64) * n_vertices
                    + slot_pairs.max(axis=2))                      # (n_el, 3)
        edge_key = edges[:, 0].astype(np.int64) * n_vertices + edges[:, 1]
        order = np.argsort(edge_key)
        slot_edge = order[np.searchsorted(edge_key[order], pair_key.ravel())]   # (n_el*3,)

        elem_of_slot = np.repeat(np.arange(n_el), 3)
        side = (edge_elements[slot_edge, 0] != elem_of_slot).astype(int)
        k, d = edge_flux.shape[2], edge_flux.shape[3]
        edge_side_flux = np.zeros((len(edges), 2, k, d))
        edge_side_flux[slot_edge, side] = edge_flux.reshape(n_el * 3, k, d)
        return edge_side_flux


@dataclass(frozen=True)
class RecoveryEstimator:
    '''Zienkiewicz-Zhu recovery estimator: `eta_K = ||sigma* - sigma_h||_K`.

    The discrete flux `sigma_h` is discontinuous across elements; the recovered
    `sigma*` is its L2 projection onto the continuous nodal space, a superconvergent
    field that stands in for the unknown exact flux. Their gap, integrated over each
    element, estimates the error.

    Both fields are read at the same quadrature points, at a degree-`2p` rule on a
    degree-`p` element (the flux is degree `p - 1`, so the squared gap is `2(p - 1)`).
    Needs no edge normals, so it is dimension-general (validated in 2D). L2-projection
    recovery is biased at boundaries and re-entrant corners; it still orders elements
    well enough to drive refinement, though the effectivity there is looser. The flux
    is the problem's operator's derived field.
    '''

    def estimate(self, problem: Problem, solution: FieldSolution) -> ElementValues:
        space = problem.space
        flux = _flux(problem)
        ctx = _solved(problem, solution)
        degree = 2 * space.element_type.SHAPE_DEGREE
        geometry = space.geometry_at(degree)
        sigma_h = flux.sample(ctx.solution, geometry)           # (n_el, n_qp, k, d)
        sigma_star = project_to_nodal(space, sigma_h, geometry)  # (n_nodes, k, d), continuous

        # Integrate ||sigma* - sigma_h||^2 over each element, both fields at the same points.
        per_element = sigma_star[space.element_nodes]     # (n_el, N, k, d)
        sigma_star_qp = np.einsum('qn,en...->eq...', geometry.shape, per_element)
        diff = sigma_star_qp - sigma_h                    # (n_el, n_qp, k, d)
        # Pointwise squared Frobenius norm over the flux's component axes. The same for
        # every flux (a scalar gradient or a stress tensor), so the engine owns it.
        density = np.sum(diff**2, axis=(-1, -2))          # (n_el, n_qp)
        eta_squared = np.einsum('eq,eq->e', density, geometry.weight_detJ)
        return np.sqrt(np.maximum(eta_squared, 0.0))


# -- goal-oriented (dual-weighted) refinement ---------------------------------


@dataclass(frozen=True)
class GoalOrientedEstimator:
    '''Goal-oriented refinement: reduce the error in a quantity of interest.

    A global estimator refines wherever the solution is rough; this refines where
    refinement most improves a specific output `J(u)` (a point value, a reaction, an
    aggregated stress). The indicator is the product of two recovery indicators,

        eta_K = eta_K^primal(u_h) * eta_K^dual(z_h),

    from the error representation `J(u) - J(u_h) = a(e, z - z_h)` localized by
    Cauchy-Schwarz element by element: `|J(u) - J(u_h)| <= sum_K eta_K^primal eta_K^dual`,
    with `eta^primal` measuring where the primal solution is inaccurate and `eta^dual`
    where the goal is sensitive to that inaccuracy. This is a common goal-oriented
    heuristic, not the dual-weighted-residual estimate, which weights the primal element
    residual by the dual's interpolation error and can be much sharper. The dual (adjoint)
    solution `z` solves `Kᵀ z = ∂J/∂u` through `SensitivityAnalysis`. Built on the
    recovery estimator, so it is dimension-general; the dual solve refactors the
    operator once per round.
    '''
    quantity_of_interest: 'QuantityOfInterest'

    def estimate(self, problem: Problem, solution: FieldSolution) -> ElementValues:
        from fem.analysis.sensitivity import SensitivityAnalysis

        base = RecoveryEstimator()
        eta_primal = base.estimate(problem, solution)

        z = SensitivityAnalysis(problem).adjoint(self.quantity_of_interest, solution.dofs)
        # The same typed solution a forward solve gives, so the recovery estimator can
        # read the dual flux like the primal's.
        eta_dual = base.estimate(problem, problem.solution(z))

        return eta_primal * eta_dual
