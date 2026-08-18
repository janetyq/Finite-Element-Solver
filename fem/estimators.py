"""A-posteriori error estimators: per-element indicators over a solved system.

An estimator answers "where is the discrete solution least trustworthy?" as one
non-negative number per element, which an `AdaptiveRefinement` driver turns into a
refinement decision. It is an operation on a solved system, not data of the PDE, so
it lives here rather than on `Equation`: the equation only names its physics
(`Equation.derived_field`), as it names its `operator` and `energy_density`.

Two families share one seam and one physics hook:

- **Residual** (`ResidualEstimator`): measures how badly the computed field fails
  the PDE, through an interior term (the source it does not balance), an
  interior-edge jump (the flux discontinuity between neighbours), and a boundary
  term (the applied traction the discrete flux does not match). A direct check of
  equilibrium, it needs the mesh's edge normals, so it is 2D-only for now.

- **Recovery** (`RecoveryEstimator`): the Zienkiewicz-Zhu idea. The discrete flux
  is discontinuous (element-constant for P1). A recovered continuous flux `sigma*`,
  the volume-weighted nodal average `FunctionSpace.recover_nodal` builds, is
  much closer to the exact flux, so `eta_K = ||sigma* - sigma_h||_K` measures the
  error. It reads no edge normals, so it is dimension-general (validated in 2D).

The one equation-specific input, shared by both, is the **`DerivedField`** (`Equation.derived_field`,
from `fem.postprocess`): which field to jump or recover (Poisson's gradient, elasticity's
stress) and, for the residual estimator only, what the boundary residual is. Everything
else is neutral machinery, the same `Form`/`assemble` split the rest of the package uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from fem.regions import evaluate_field

if TYPE_CHECKING:
    from fem.adaptivity import RefinableSolver
    from fem.boundary import ResolvedBC
    from fem.postprocess import DerivedField
    from fem.equations import Equation
    from fem.solution import FieldSolution
    from fem.space import FunctionSpace
    from fem.typing import BoolArray, ElementField, FieldValue, FloatArray


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
    is_fixed = np.zeros((len(space.mesh.vertices), space.n_components), dtype=bool)
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

    `eta_K^2 = h_K^2 ||f||^2_K + (h_K/2) sum_edges ||[[flux.n]]||^2_e
                                + h_K sum_(bnd edges) ||boundary residual||^2_e`

    The engine owns every geometric quantity (`h_K`, the edge normals, the
    accumulation) and delegates the three physics pieces to the `flux`: the flux
    field it jumps, and (per boundary edge) the boundary residual. The interior term
    is the source `f` the P1 field cannot balance (`div(flux) = 0` inside a
    constant-strain element), read at the element centroid.

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

        centroids = mesh.vertices[mesh.elements].mean(axis=1)
        f = evaluate_field(self.source, centroids, space.n_components)   # (n_el, k)
        interior = h_K**2 * np.sum(f**2, axis=1) * space.element_volumes

        jump_term = np.zeros(n_elements)
        boundary_term = np.zeros(n_elements)

        vertices = mesh.vertices
        edges = mesh.edges                    # (E, 2) sorted vertex pairs
        edge_elements = mesh.edge_elements    # (E, 2), -1 in slot 1 on a boundary edge
        is_interior = edge_elements[:, 1] >= 0

        # Interior edges, all at once: the flux is continuous in the true
        # solution but jumps between the piecewise-constant discrete neighbours.
        pairs = edges[is_interior]
        e0, e1 = edge_elements[is_interior, 0], edge_elements[is_interior, 1]
        edge_vecs = vertices[pairs[:, 1]] - vertices[pairs[:, 0]]              # (Ei, 2)
        edge_lens = np.linalg.norm(edge_vecs, axis=1)                         # (Ei,)
        normals = np.stack([-edge_vecs[:, 1], edge_vecs[:, 0]], axis=1) / edge_lens[:, None]
        jumps = np.einsum('ekd,ed->ek', flux[e0] - flux[e1], normals)         # (Ei, k)
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


@dataclass(frozen=True)
class RecoveryEstimator:
    '''Zienkiewicz-Zhu recovery estimator: `eta_K = ||sigma* - sigma_h||_K`.

    The discrete flux `sigma_h` is element-constant (P1) and discontinuous; the
    recovered `sigma*` is its volume-weighted nodal average, a continuous field that,
    being superconvergent, stands in for the unknown exact flux. Their gap, integrated
    over each element, estimates the error.

    Needs no edge normals, so unlike the residual estimator it is dimension-general
    (validated in 2D). Recovery by simple averaging is biased at boundaries and
    re-entrant corners; it still orders elements well enough to drive refinement, but
    the effectivity there is looser (patch recovery would tighten it).
    '''
    flux: DerivedField

    def estimate(self, solver: RefinableSolver) -> ElementField:
        space = solver.space
        ctx = _solved(solver)                             # raises if the solver has not solved
        sigma_h = self.flux.evaluate(ctx.solution)        # (n_el, k, d), constant per element
        sigma_star = space.recover_nodal(sigma_h)         # (n_nodes, k, d), continuous

        # Integrate ||sigma* - sigma_h||^2 over each element. sigma* is P1 (linear),
        # sigma_h constant, so the integrand is quadratic, and a degree-2 rule is exact.
        geometry = space.geometry_at(2)
        per_element = sigma_star[space.element_nodes]     # (n_el, N, k, d)
        sigma_star_qp = np.einsum('qn,en...->eq...', geometry.shape, per_element)
        diff = sigma_star_qp - sigma_h[:, None]           # (n_el, n_qp, k, d)
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
