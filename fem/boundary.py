"""Boundary conditions, specified against geometry and resolved against a mesh.

The split here is the whole design. A `BoundaryConditions` is a specification:
mesh-independent and discretization-independent, describing what the user means ("the left edge
is pinned"). A `ResolvedBC` is what a solver needs: concrete DOF indices and load
vectors for one particular mesh and one particular number of DOFs per node.

Keeping the specification lets a condition survive remeshing (resolve it again
against the new mesh), and keeping the resolution immutable and per-component-count
stops one shared BC object from silently reconfiguring itself when handed to a
solver for a different equation.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from fem.regions import _coerce_components, evaluate_field, is_mesh_bound
from fem.typing import (
    BoolArray,
    DofIndices,
    Elements,
    FieldValue,
    FloatArray,
    IntArray,
    Region,
    VertexIndices,
    VertexField,
    Vertices,
)

logger = logging.getLogger(__name__)


class NodeGeometry(Protocol):
    '''The node geometry `resolve` and `select` read: coordinates, the boundary
    facets as node-index tuples, and the boundary node indices.

    `Mesh` satisfies it directly (its nodes are the vertices), and so does the
    `NodeSet` a P2 `FunctionSpace` builds (whose nodes include the edge midpoints),
    which lets one resolver pin vertex and edge DOFs alike. The members are
    read-only properties so both a plain-attribute `Mesh` and a frozen `NodeSet`
    satisfy it.
    '''
    @property
    def vertices(self) -> Vertices: ...
    @property
    def boundary(self) -> Elements: ...
    @property
    def boundary_idxs(self) -> IntArray: ...


def _evaluate_dirichlet_value(value: FieldValue, points: Vertices, n_components: int) -> FloatArray:
    '''Like `evaluate_field`, but a component may be `None`, left as `NaN`, meaning
    "this DOF stays free" rather than "pinned to this value". Dirichlet-specific: a
    free component is only meaningful for an essential condition, never a load.
    '''
    values = _coerce_components(value, points, n_components)
    if values.shape != (len(points), n_components):
        raise ValueError(
            f'field must give {n_components} component(s) per point, got shape {values.shape} '
            f'for {len(points)} point(s)'
        )
    return values


class BCType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    # Robin (∂u/∂n + κu = g) contributes to both sides, unlike the other two: a
    # boundary term κ∫u·v to the operator and ∫g·v to the load. Its value is the
    # pair (kappa, g); `resolve` turns it into a RobinContribution that LinearProblem
    # assembles through a MaskedMassForm over the region's boundary facets.
    ROBIN = "robin"


@dataclass(frozen=True)
class NeumannContribution:
    '''One resolved Neumann condition: a traction ∫_∂Ω_R g·v over a region's facets.

    `facet_mask` marks the boundary facets in the region; a `MaskedMassForm` over them
    integrates `traction` (the nodal field g) across those facets alone, the same
    region-restricted integral a Robin condition uses. Masking keeps a traction from
    spreading onto a neighbouring edge through a shared corner node, which the
    unmasked boundary mass it replaces did, inflating the applied resultant.
    '''
    facet_mask: BoolArray       # one entry per boundary facet
    traction: VertexField       # (n_vertices, n_components), nonzero on the region nodes


@dataclass(frozen=True)
class RobinContribution:
    '''One resolved Robin condition (∂u/∂n + κu = g on a region).

    It contributes to both sides of the system: the boundary term κ∫_∂Ω_R u·v to
    the operator and ∫_∂Ω_R g·v to the load. `facet_mask` marks which of the mesh's
    boundary facets lie in the region, so a `MaskedMassForm` over them assembles the
    region-restricted boundary integral; `kappa` scales the operator term and `g`
    (nonzero on the Robin nodes) drives the load. The assembly itself waits for a
    FunctionSpace, so this carries only the data, keyed to the mesh, the same
    spec-then-resolve split the rest of the BC layer uses.
    '''
    facet_mask: BoolArray       # one entry per boundary facet
    kappa: float
    g: VertexField              # (n_vertices, n_components), nonzero on the Robin nodes


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
    robin: tuple[RobinContribution, ...] = ()   # boundary terms for the operator + load
    # One masked traction integral per Neumann condition; the assembled load is built from
    # these so each region stays on its own facets. `neumann_load` above is the same data
    # as a global nodal field, kept only for the error estimator (which reads g nodally).
    neumann: tuple[NeumannContribution, ...] = ()


class BoundaryConditions:
    '''A mesh-independent specification of the conditions on a domain boundary.'''

    def __init__(self) -> None:
        self.conditions: list[tuple[BCType, Region, FieldValue]] = []
        # Robin is stored apart: it carries a coefficient kappa as well as data g,
        # and contributes to both sides of the system, so it does not fit the
        # (type, region, value) shape the value-on-region conditions share.
        self.robin_conditions: list[tuple[Region, float, FieldValue]] = []

    def _check_region(self, region: Region) -> None:
        if not callable(region):
            raise TypeError(
                'region must be a callable over point coordinates; pass a helper '
                'from fem.regions (e.g. on_plane(0, 0.0)), or at_indices([...]) for '
                f'specific nodes. Got {type(region).__name__}.'
            )

    def add(self, bc_type: BCType | str, region: Region, value: FieldValue) -> None:
        '''Apply `value` of type `bc_type` on `region`.

        `region` is a callable over point coordinates (see fem.regions); `value`
        is either a constant or a callable of position. Both are resolved lazily,
        so a condition means the same thing on any mesh. For Robin conditions use
        `add_robin`, which also takes the coefficient.

        For DIRICHLET on a vector field, a component may be `None` to leave it
        free rather than pinned: `[0, None]` pins x and leaves y natural, a
        roller rather than a clamp. A vertex may pick up its remaining component
        from a second, overlapping `add` call (e.g. one point elsewhere pinning y
        to remove the last rigid-body mode); the two merge rather than conflict as
        long as they agree on any component both specify. Meaningless for
        NEUMANN/Robin (a load has no "free" component), so `None` there raises.
        '''
        bc_type = BCType(bc_type)  # accepts BCType or its value; unknown raises ValueError
        if bc_type is BCType.ROBIN:
            raise ValueError('use add_robin(region, kappa, g) for Robin conditions')
        self._check_region(region)
        self.conditions.append((bc_type, region, value))

    def add_robin(self, region: Region, kappa: float, g: FieldValue) -> None:
        '''Apply a Robin condition ∂u/∂n + kappa*u = g on `region`.

        `kappa` is a constant coefficient; `g` is a constant or a callable of
        position. Contributes the boundary term kappa*int u*v to the operator and
        int g*v to the load, both over the region's boundary facets.
        '''
        self._check_region(region)
        self.robin_conditions.append((region, float(kappa), g))

    def is_mesh_bound(self) -> bool:
        '''Whether any condition is tied to one mesh's vertex numbering, and so
        cannot be carried across a remesh.'''
        regions = [r for _, r, _ in self.conditions] + [r for r, _, _ in self.robin_conditions]
        return any(is_mesh_bound(region) for region in regions)

    def check_remeshable(self) -> None:
        if self.is_mesh_bound():
            raise NotImplementedError(
                'this specification uses at_indices, which names vertices of one '
                'specific mesh and cannot survive the renumbering a remesh does. '
                'Describe the region geometrically (see fem.regions) to make it '
                'remeshable.'
            )

    def select(self, nodes: NodeGeometry, region: Region) -> VertexIndices:
        '''Boundary nodes of `nodes` inside `region`.

        Regions are evaluated over every node and then intersected with the
        boundary, which makes "a boundary condition on an interior node"
        unrepresentable rather than something to diagnose afterwards. For a P2 node
        set this picks up the edge-midpoint nodes on the boundary automatically,
        since they satisfy the same geometric region their endpoints do.
        '''
        selected = np.flatnonzero(region(nodes.vertices))
        boundary = np.asarray(nodes.boundary_idxs, dtype=int)

        if is_mesh_bound(region):
            # Naming a node explicitly is a claim about that node, so silently
            # dropping an interior one would hide a mistake. Describing a region
            # is a filter, where the intersection is the intent.
            interior = np.setdiff1d(selected, boundary)
            if len(interior):
                raise ValueError(
                    f'boundary conditions on non-boundary nodes: {sorted(interior)}'
                )
            return selected
        return np.intersect1d(selected, boundary)

    def entries(self, nodes: NodeGeometry) -> list[tuple[BCType, VertexIndices, FloatArray]]:
        '''[(bc_type, node_idxs, values), ...] resolved against `nodes`.

        Region resolution only, no DOF numbering, so this needs no `n_components` and is
        what inspection and plotting use.
        '''
        def resolved_values(idxs, value):
            # Display only, so this stays permissive rather than dispatching on
            # bc_type the way resolve() does: a Dirichlet component may
            # legitimately be None/free, and a stray None elsewhere is a user
            # mistake worth seeing here (as a literal NaN) rather than one
            # this inspection path hides by raising before resolve() can.
            return _coerce_components(value, nodes.vertices[idxs], 1) if len(idxs) \
                else np.zeros((0, 1))

        out = []
        for bc_type, region, value in self.conditions:
            idxs = self.select(nodes, region)
            out.append((bc_type, idxs, resolved_values(idxs, value)))
        for region, _kappa, g in self.robin_conditions:
            idxs = self.select(nodes, region)
            out.append((BCType.ROBIN, idxs, resolved_values(idxs, g)))
        return out

    def resolve(self, nodes: NodeGeometry, n_components: int) -> ResolvedBC:
        '''Reduce this specification to a `ResolvedBC` for `nodes` at `n_components` DOFs per node.'''
        n = len(nodes.vertices)
        dirichlet: dict[int, FloatArray] = {}
        neumann = np.zeros((n, n_components))
        neumann_contributions: list[NeumannContribution] = []
        robin: list[RobinContribution] = []
        dirichlet_vertices, neumann_vertices = [], []

        for bc_type, region, value in self.conditions:
            idxs = self.select(nodes, region)

            if bc_type is BCType.DIRICHLET:
                values = _evaluate_dirichlet_value(value, nodes.vertices[idxs], n_components)
                for v_idx, v in zip(idxs, values):
                    # Overlapping regions are normal (a corner belongs to two
                    # edges, or, for a roller, an edge and the one point
                    # that pins its other component); a component that both
                    # conditions specify but disagree on is a real conflict,
                    # and last-write-wins would bury it. A component either
                    # side leaves free (NaN) never conflicts; the other
                    # side's value (fixed or itself free) wins.
                    if v_idx in dirichlet:
                        existing = dirichlet[v_idx]
                        both_given = ~np.isnan(existing) & ~np.isnan(v)
                        if both_given.any() and not np.allclose(existing[both_given], v[both_given]):
                            raise ValueError(
                                f'conflicting Dirichlet values at vertex {v_idx}: '
                                f'{existing} and {v}'
                            )
                        v = np.where(np.isnan(v), existing, v)
                    dirichlet[v_idx] = v
                dirichlet_vertices.extend(int(i) for i in idxs)
            else:
                values = evaluate_field(value, nodes.vertices[idxs], n_components)
                neumann[idxs] += values
                # Per-condition facet mask, so the traction integrates over this region's
                # facets alone (as Robin does) and a corner node cannot carry it onto a
                # neighbour. The global `neumann` field above is for the error estimator.
                traction = np.zeros((n, n_components))
                traction[idxs] = values
                facet_mask = np.asarray(np.isin(nodes.boundary, idxs).all(axis=1), dtype=bool)
                neumann_contributions.append(NeumannContribution(facet_mask, traction))
                neumann_vertices.extend(int(i) for i in idxs)

        for region, kappa, g_field in self.robin_conditions:
            idxs = self.select(nodes, region)
            g = np.zeros((n, n_components))
            g[idxs] = evaluate_field(g_field, nodes.vertices[idxs], n_components)
            # A boundary facet is in the region iff all its nodes are: the
            # all-nodes rule that keeps the boundary integral crisp.
            facet_mask = np.asarray(np.isin(nodes.boundary, idxs).all(axis=1), dtype=bool)
            robin.append(RobinContribution(facet_mask, kappa, g))

        dirichlet_vertices = np.unique(dirichlet_vertices).astype(int)
        neumann_vertices = np.unique(neumann_vertices).astype(int)

        # A fixed DOF ignores any traction on it, so the ambiguity to reject is per
        # (vertex, component): a component that is both pinned and loaded. Pinning one
        # component while a traction drives a different one (a roller carrying a
        # tangential load) is well-posed and allowed; the fixed component is eliminated
        # by `DiscreteSystem`, dropping its traction, and the free ones keep theirs.
        conflicts = [
            int(v) for v, values in dirichlet.items()
            if np.any(~np.isnan(values) & (neumann[v] != 0.0))
        ]
        if conflicts:
            raise ValueError(
                'vertices carry a Dirichlet and a Neumann condition on the same '
                f'component: {sorted(conflicts)}'
            )

        # Per (vertex, component): a NaN entry is a component a condition left free
        # (a roller's tangential direction, say), so it contributes no fixed DOF;
        # free_idxs, being the complement over the whole DOF range, picks it up
        # without needing to know that is why.
        fixed_idxs = np.array(
            [n_components*v + d for v in sorted(dirichlet) for d in range(n_components)
             if not np.isnan(dirichlet[v][d])],
            dtype=int,
        )
        fixed_values = np.array(
            [dirichlet[v][d] for v in sorted(dirichlet) for d in range(n_components)
             if not np.isnan(dirichlet[v][d])],
            dtype=float,
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
            robin=tuple(robin),
            neumann=tuple(neumann_contributions),
        )
