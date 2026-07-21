"""Boundary conditions, specified against geometry and resolved against a mesh.

The split here is the whole design. A `BoundaryConditions` is a *specification*:
mesh-independent and discretization-independent, describing what the user means ("the left edge
is pinned"). A `ResolvedBC` is what a solver needs: concrete DOF indices and load
vectors for one particular mesh and one particular number of DOFs per node.

Keeping the specification is what lets a condition survive remeshing -- resolve
it again against the new mesh -- and keeping the resolution immutable and
per-component-count is what stops one shared BC object from silently reconfiguring itself
when handed to a solver for a different equation.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from fem.regions import evaluate_field, is_mesh_bound
from fem.typing import (
    DofIndices,
    FieldValue,
    FloatArray,
    Region,
    VertexIndices,
    VertexField,
)

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh

logger = logging.getLogger(__name__)


class BCType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    # Robin (a*u + b*du/dn = g) contributes a term to the *left-hand side*, unlike
    # the other two. Accepted by `add` so the intent can be expressed, refused by
    # `resolve` until it is wired up -- the hook is the already-assembled but
    # currently unused boundary matrices K_b / M_b, added to the system in
    # Solver.assemble_everything.
    ROBIN = "robin"


@dataclass(frozen=True)
class ResolvedBC:
    '''Boundary conditions reduced to what a solver indexes into.

    Frozen and built per (mesh, n_components) so it cannot drift out of step with either.
    '''
    n_vertices: int
    n_components: int
    fixed_idxs: DofIndices      # DOF indices held by Dirichlet conditions
    free_idxs: DofIndices       # the complement
    fixed_values: FloatArray    # values at fixed_idxs, same order
    neumann_load: VertexField   # (n_vertices, n_components) traction field
    dirichlet_vertices: VertexIndices
    neumann_vertices: VertexIndices


class BoundaryConditions:
    '''A mesh-independent specification of the conditions on a domain boundary.'''

    def __init__(self) -> None:
        self.conditions: list[tuple[BCType, Region, FieldValue]] = []

    def add(self, bc_type: BCType | str, region: Region, value: FieldValue) -> None:
        '''Apply `value` of type `bc_type` on `region`.

        `region` is a callable over point coordinates (see fem.regions); `value`
        is either a constant or a callable of position. Both are resolved lazily,
        so a condition means the same thing on any mesh.
        '''
        bc_type = BCType(bc_type)  # accepts BCType or its value; unknown raises ValueError
        if not callable(region):
            raise TypeError(
                'region must be a callable over point coordinates -- pass a helper '
                'from fem.regions (e.g. on_plane(0, 0.0)), or at_indices([...]) for '
                f'specific nodes. Got {type(region).__name__}.'
            )
        self.conditions.append((bc_type, region, value))

    def is_mesh_bound(self) -> bool:
        '''Whether any condition is tied to one mesh's vertex numbering, and so
        cannot be carried across a remesh.'''
        return any(is_mesh_bound(region) for _, region, _ in self.conditions)

    def check_remeshable(self) -> None:
        if self.is_mesh_bound():
            raise NotImplementedError(
                'this specification uses at_indices, which names vertices of one '
                'specific mesh and cannot survive the renumbering a remesh does. '
                'Describe the region geometrically (see fem.regions) to make it '
                'remeshable.'
            )

    def select(self, mesh: 'Mesh', region: Region) -> VertexIndices:
        '''Boundary vertices of `mesh` inside `region`.

        Regions are evaluated over every vertex and then intersected with the
        boundary, which makes "a boundary condition on an interior vertex"
        unrepresentable rather than something to diagnose afterwards.
        '''
        selected = np.flatnonzero(region(mesh.vertices))
        boundary = np.asarray(mesh.boundary_idxs, dtype=int)

        if is_mesh_bound(region):
            # Naming a node explicitly is a claim about that node, so silently
            # dropping an interior one would hide a mistake. Describing a region
            # is a filter, where the intersection *is* the intent.
            interior = np.setdiff1d(selected, boundary)
            if len(interior):
                raise ValueError(
                    f'boundary conditions on non-boundary vertices: {sorted(interior)}'
                )
            return selected
        return np.intersect1d(selected, boundary)

    def entries(self, mesh: 'Mesh') -> list[tuple[BCType, VertexIndices, FloatArray]]:
        '''[(bc_type, vertex_idxs, values), ...] resolved against `mesh`.

        Region resolution only -- no DOF numbering, so this needs no `n_components` and is
        what inspection and plotting use.
        '''
        out = []
        for bc_type, region, value in self.conditions:
            idxs = self.select(mesh, region)
            values = np.array([np.atleast_1d(np.asarray(
                value(mesh.vertices[i]) if callable(value) else value, dtype=float
            )) for i in idxs]) if len(idxs) else np.zeros((0, 1))
            out.append((bc_type, idxs, values))
        return out

    def resolve(self, mesh: 'Mesh', n_components: int) -> ResolvedBC:
        '''Reduce this specification to a `ResolvedBC` for `mesh` at `n_components` DOFs per node.'''
        n = len(mesh.vertices)
        dirichlet: dict[int, FloatArray] = {}
        neumann = np.zeros((n, n_components))
        dirichlet_vertices, neumann_vertices = [], []

        for bc_type, region, value in self.conditions:
            if bc_type is BCType.ROBIN:
                raise NotImplementedError(
                    'Robin conditions are not implemented yet: they add a term to the '
                    'system matrix (via the boundary matrices K_b / M_b in '
                    'Solver.assemble_everything), not just to the load.'
                )

            idxs = self.select(mesh, region)
            values = evaluate_field(value, mesh.vertices[idxs], n_components)

            if bc_type is BCType.DIRICHLET:
                for v_idx, v in zip(idxs, values):
                    # Overlapping regions are normal (a corner belongs to two
                    # edges); overlapping regions that disagree are a real
                    # conflict, and last-write-wins would bury it.
                    if v_idx in dirichlet and not np.allclose(dirichlet[v_idx], v):
                        raise ValueError(
                            f'conflicting Dirichlet values at vertex {v_idx}: '
                            f'{dirichlet[v_idx]} and {v}'
                        )
                    dirichlet[v_idx] = v
                dirichlet_vertices.extend(int(i) for i in idxs)
            else:
                neumann[idxs] += values
                neumann_vertices.extend(int(i) for i in idxs)

        dirichlet_vertices = np.unique(dirichlet_vertices).astype(int)
        neumann_vertices = np.unique(neumann_vertices).astype(int)

        # A fixed node ignores its traction, so the pairing is ambiguous either way.
        overlap = np.intersect1d(dirichlet_vertices, neumann_vertices)
        if len(overlap):
            raise ValueError(
                f'vertices carry both Dirichlet and Neumann conditions: {sorted(overlap)}'
            )

        fixed_idxs = np.array(
            [n_components*v + d for v in sorted(dirichlet) for d in range(n_components)], dtype=int
        )
        fixed_values = np.array(
            [dirichlet[v][d] for v in sorted(dirichlet) for d in range(n_components)], dtype=float
        )
        free_idxs = np.setdiff1d(np.arange(n * n_components), fixed_idxs)

        return ResolvedBC(
            n_vertices=n,
            n_components=n_components,
            fixed_idxs=fixed_idxs,
            free_idxs=free_idxs,
            fixed_values=fixed_values,
            neumann_load=neumann,
            dirichlet_vertices=dirichlet_vertices,
            neumann_vertices=neumann_vertices,
        )
