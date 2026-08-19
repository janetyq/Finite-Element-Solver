"""A-posteriori error estimators: per-element indicators over a solved system.

An estimator answers "where is the discrete solution least trustworthy?" as one
non-negative number per element, which an `AdaptiveRefinement` driver turns into a
refinement decision. It is an operation on a solved system, not data of the PDE, so
it lives here rather than on `Equation`: the equation only names its physics
(`Equation.derived_field`), as it names its `operator` and `energy_density`.

Two families share one seam and one physics hook:

- **Residual** (`ResidualEstimator`): measures how badly the computed field fails
  the PDE, through an interior term (the strong residual `source + div(flux)` the
  field leaves unbalanced), an interior-edge jump (the flux discontinuity between
  neighbours), and a boundary term (the applied traction the discrete flux does not
  match). A direct check of equilibrium, it needs the mesh's edge normals, so it is
  2D-only for now. It drives P1 and P2 alike.

- **Recovery** (`RecoveryEstimator`): the Zienkiewicz-Zhu idea. The discrete flux
  is discontinuous across elements (element-constant for P1, linear within each for
  P2). A recovered continuous flux `sigma*`, the L2 projection onto the nodal space
  `FunctionSpace.project_to_nodal` builds from the flux sampled at quadrature points,
  is much closer to the exact flux, so `eta_K = ||sigma* - sigma_h||_K` measures the
  error. It reads no edge normals, so it is dimension-general (validated in 2D), and
  sampling per point rather than per element makes it follow a P2 solution.

The one equation-specific input, shared by both, is the **`DerivedField`** (`Equation.derived_field`,
from `fem.postprocess`): which field to jump or recover (Poisson's gradient, elasticity's
stress) and, for the residual estimator only, what the boundary residual is. Everything
else is neutral machinery, the same `Form`/`assemble` split the rest of the package uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from fem.quadrature import QuadratureRule
from fem.regions import evaluate_field

# The three edge midpoints in reference coordinates, ordered opposite corner 0, 1, 2 to
# match `element_nodes[:, 3:6]`. Sampling the flux here gives its value on each edge, the
# per-side traction whose jump the residual estimator measures. For P1 (a constant flux)
# any point gives the element's one value, so this reduces to the element-constant jump.
_REFERENCE_EDGE_MIDPOINTS = np.array([[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])

if TYPE_CHECKING:
    from fem.adaptivity import RefinableSolver
    from fem.boundary import ResolvedBC
    from fem.postprocess import DerivedField
    from fem.equations import Equation
    from fem.solution import FieldSolution
    from fem.space import FunctionSpace
    from fem.typing import BoolArray, ElementField, FieldValue, FloatArray, IntArray


# -- the solved-system view the flux hooks read -------------------------------


@dataclass(frozen=True)
class Solved:
    '''The resolved view of a solved system an estimator reads: space, solution, BCs.

    Built once per `estimate` so the engine does not re-derive the DOF partition per
    edge. `is_fixed[v, c]` marks vertex `v`'s component `c` as Dirichlet-constrained,
    the mask the residual estimator turns into the `free` argument of a boundary
    residual so a pinned direction's reaction traction is not counted.
    '''
    space: FunctionSpace
    solution: FieldSolution
    resolved: ResolvedBC
    is_fixed: BoolArray   # (n_vertices, n_components)


def _solved(solver: RefinableSolver) -> Solved:
    '''Resolve `solver`'s latest solve into the view the flux hooks read.

    Raises if the solver has not solved yet: the single guard both estimators lean
    on, so `estimate` never has to narrow the solution itself.
    '''
    solution = solver.solution
    if solution is None:
        raise ValueError('the error estimator requires a solved system')
    space = solver.space
    resolved = solver.boundary_conditions.resolve(space.nodes, space.n_components)
    # Sized by nodes, not mesh vertices: a P2 space fixes edge-midpoint DOFs too, whose
    # indices run past the vertex count. The residual estimator reads only vertex rows,
    # so this stays a strict generalization of the P1 mask.
    is_fixed = np.zeros((space.n_nodes, space.n_components), dtype=bool)
    is_fixed.ravel()[resolved.fixed_idxs] = True
    return Solved(space, solution, resolved, is_fixed)


# -- the outer seam every estimator satisfies ---------------------------------


@runtime_checkable
class ErrorEstimator(Protocol):
    '''A per-element error indicator over a solved system.

    The single method `AdaptiveRefinement` drives. Residual and recovery estimators
    implement it, and so would a future goal-oriented one, though they share no
    internals. That is why this is the seam the driver depends on.
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

    `eta_K^2 = h_K^2 ||f + div(flux)||^2_K + (h_K/2) sum_edges ||[[flux.n]]||^2_e
                                           + h_K sum_(bnd edges) ||boundary residual||^2_e`

    The engine owns every geometric quantity (`h_K`, the edge normals, the
    accumulation) and delegates the physics to the `flux`: the field it jumps, its
    divergence (the interior residual `f + div(flux)`), and the per-boundary-edge
    residual. On P1 the flux is element-constant, so `div(flux)` is zero and the jump
    reads one value per element; on P2 the flux varies, so the interior term carries the
    divergence and the jump is read at the shared edge from each side.

    The interior residual is read at the element centroid, and the boundary term uses the
    element's stored (representative) traction, both exact on P1 and a light approximation
    on P2 where those quantities vary within the element.

    2D only: the jump and boundary terms need edge normals. A 3D mesh would need face
    normals, which the recovery estimator sidesteps entirely.
    '''
    flux: DerivedField
    source: FieldValue = None

    def estimate(self, solver: RefinableSolver) -> ElementField:
        mesh, space = solver.mesh, solver.space
        if mesh.spatial_dim != 2:
            raise NotImplementedError('the residual error estimator needs face normals (2D only)')

        ctx = _solved(solver)                    # raises if the solver has not solved
        flux = self.flux.evaluate(ctx.solution)  # (n_el, k, d)

        h_K = mesh.element_diameters
        n_elements = len(mesh.elements)

        # The strong-form interior residual `f + div(flux)`, read at the element centroid.
        # `div(flux)` is zero for P1 (a constant flux), so this is `f` alone there; a P2
        # flux varies, and its divergence is the term the linear estimator was missing.
        centroids = mesh.vertices[mesh.elements].mean(axis=1)
        f = evaluate_field(self.source, centroids, space.n_components)   # (n_el, k)
        residual = f + self.flux.divergence(ctx.solution)               # (n_el, k)
        interior = h_K**2 * np.sum(residual**2, axis=1) * space.element_volumes

        jump_term = np.zeros(n_elements)
        boundary_term = np.zeros(n_elements)

        vertices = mesh.vertices
        edges = mesh.edges                    # (E, 2) sorted vertex pairs
        edge_elements = mesh.edge_elements    # (E, 2), -1 in slot 1 on a boundary edge
        is_interior = edge_elements[:, 1] >= 0

        # The flux each element carries on each of its edges, gathered per global edge and
        # side. On P2 the flux varies along the edge, so a jump read at the shared edge is
        # sharper than one between the two elements' representative values; on P1 it is the
        # element-constant jump unchanged.
        edge_side_flux = self._per_side_edge_flux(space, ctx.solution, edges, edge_elements)

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
        np.add.at(jump_term, e0, (h_K[e0] / 2) * contribution / 2)
        np.add.at(jump_term, e1, (h_K[e1] / 2) * contribution / 2)

        # Boundary edges are comparatively few and the flux's boundary residual
        # is a per-edge physics hook, so these stay a Python loop.
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
            # fixed at both, its traction is a reaction, not a residual. The engine owns
            # this BC context and hands the flux the primitives it needs.
            free = ~(ctx.is_fixed[v0] & ctx.is_fixed[v1])
            g = 0.5 * (ctx.resolved.neumann_load[v0] + ctx.resolved.neumann_load[v1])
            residual2 = self.flux.boundary_residual(flux[e_bnd], normal, g, free)
            boundary_term[e_bnd] += h_K[e_bnd] * edge_len * residual2

        eta_squared = interior + jump_term + boundary_term
        return np.sqrt(np.maximum(eta_squared, 0.0))

    def _per_side_edge_flux(
        self, space: FunctionSpace, solution: FieldSolution,
        edges: IntArray, edge_elements: IntArray,
    ) -> FloatArray:
        '''(n_edges, 2, k, d) the flux each side carries at each edge's midpoint.

        For every element the flux is sampled at its three edge midpoints, then scattered
        into `[edge, side]` so an interior edge holds the value from both neighbours (side
        0 is `edge_elements[:, 0]`). A boundary edge fills only side 0. The jump term reads
        the difference of the two sides. For P1 the samples are the element-constant flux,
        so this is the old element-to-element jump.
        '''
        element_nodes = space.element_nodes
        n_el = len(element_nodes)
        # Sample the flux at the reference edge midpoints (slot s opposite corner s).
        rule = QuadratureRule(_REFERENCE_EDGE_MIDPOINTS, np.ones(3), degree=2)
        geometry = space.element_type.geometry(space.node_coords[element_nodes], rule)
        edge_flux = self.flux.sample(solution, geometry)           # (n_el, 3, k, d)

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

    The discrete flux `sigma_h` is discontinuous across elements (element-constant for
    P1, linear within each element for P2); the recovered `sigma*` is its L2 projection
    onto the continuous nodal space, a smooth field that, being superconvergent, stands
    in for the unknown exact flux. Their gap, integrated over each element, estimates the
    error.

    Both fields are read at the same quadrature points, at a rule that resolves the
    higher-order one: the flux is degree `p - 1` on a degree-`p` element, so the squared
    gap is degree `2(p - 1)`, and a degree-`2p` rule integrates it (degree 2 on P1,
    degree 4 on P2). Sampling `sigma_h` per point rather than once per element is what
    makes the estimate follow a P2 solution's within-element variation.

    Needs no edge normals, so unlike the residual estimator it is dimension-general
    (validated in 2D). L2-projection recovery is biased at boundaries and re-entrant
    corners; it still orders elements well enough to drive refinement, though the
    effectivity there is looser (patch recovery would tighten it).
    '''
    flux: DerivedField

    def estimate(self, solver: RefinableSolver) -> ElementField:
        space = solver.space
        ctx = _solved(solver)                             # raises if the solver has not solved
        degree = 2 * space.element_type.SHAPE_DEGREE
        geometry = space.geometry_at(degree)
        sigma_h = self.flux.sample(ctx.solution, geometry)      # (n_el, n_qp, k, d)
        sigma_star = space.project_to_nodal(sigma_h, geometry)  # (n_nodes, k, d), continuous

        # Integrate ||sigma* - sigma_h||^2 over each element, both fields at the same points.
        per_element = sigma_star[space.element_nodes]     # (n_el, N, k, d)
        sigma_star_qp = np.einsum('qn,en...->eq...', geometry.shape, per_element)
        diff = sigma_star_qp - sigma_h                    # (n_el, n_qp, k, d)
        # Pointwise squared Frobenius norm over the flux's component axes. The same for
        # every flux (a scalar gradient or a stress tensor), so the engine owns it.
        density = np.sum(diff**2, axis=(-1, -2))          # (n_el, n_qp)
        eta_squared = np.einsum('eq,eq->e', density, geometry.weight_detJ)
        return np.sqrt(np.maximum(eta_squared, 0.0))


# -- factories: build an estimator from an equation's derived field -----------


def _derived_field(equation: Equation) -> DerivedField:
    '''The equation's recoverable field, or a clear error if it names none.'''
    field = equation.derived_field()
    if field is None:
        raise NotImplementedError(
            f'{type(equation).__name__} names no derived field, so adaptive refinement '
            'is not defined for it.'
        )
    return field


def residual_estimator(equation: Equation) -> ResidualEstimator:
    '''The residual estimator for `equation`, from its derived field and source.'''
    return ResidualEstimator(_derived_field(equation), equation.source)


def recovery_estimator(equation: Equation) -> RecoveryEstimator:
    '''The Zienkiewicz-Zhu recovery estimator for `equation`, from its derived field.'''
    return RecoveryEstimator(_derived_field(equation))
