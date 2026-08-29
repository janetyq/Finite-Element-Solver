"""Boundary conditions, specified against geometry and resolved against a node set.

A condition is a frozen object: `Dirichlet(region, value)`, `Neumann(region, value)`,
or `Robin(region, kappa, g)`, each a `Condition` that knows how to resolve itself into
its contribution on one node set at one number of DOFs per node. `fem.conditions`
collects them (with the loads) into a `Conditions` and reduces the collection to what a
solver indexes into.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from fem.regions import TimeDependent, _coerce_components, evaluate_field, field_at, is_mesh_bound, on_tag
from fem.typing import (
    BoolArray,
    Elements,
    FieldValue,
    FloatArray,
    IntArray,
    Region,
    VertexIndices,
    NodalValues,
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
    @property
    def boundary_tags(self) -> IntArray | None: ...


def _evaluate_dirichlet_value(value: FieldValue, points: Vertices, n_components: int) -> FloatArray:
    '''Like `evaluate_field`, but a component may be `None`, left as `NaN`, meaning
    "this DOF stays free" rather than "pinned to this value".
    '''
    values = _coerce_components(value, points, n_components)
    if values.shape != (len(points), n_components):
        raise ValueError(
            f'field must give {n_components} component(s) per point, got shape {values.shape} '
            f'for {len(points)} point(s)'
        )
    return values


def _has_free_component(value: FieldValue) -> bool:
    '''Whether a constant value names a free component with `None`.'''
    if isinstance(value, (list, tuple)):
        return any(c is None for c in value)
    return False


def _facet_mask(nodes: NodeGeometry, idxs: VertexIndices) -> BoolArray:
    '''The boundary facets whose every node is in `idxs`: the all-nodes rule that keeps
    a boundary integral on its own region and off a neighbour through a shared corner.'''
    return np.asarray(np.isin(nodes.boundary, idxs).all(axis=1), dtype=bool)


# -- the resolved contributions --------------------------------------------------


@dataclass(frozen=True)
class DirichletContribution:
    '''One resolved Dirichlet condition: the nodes it selects and the value at each,
    `NaN` for a component the condition leaves free.'''
    node_idxs: VertexIndices    # the selected boundary nodes
    points: Vertices            # their coordinates, for re-evaluation at a time
    value: FieldValue           # the spec's value, possibly TimeDependent
    values: FloatArray          # (n_selected, n_components) at the resolution time

    def at(self, t: float) -> 'DirichletContribution':
        if not isinstance(self.value, TimeDependent):
            return self
        values = _evaluate_dirichlet_value(field_at(self.value, t), self.points, self.values.shape[1])
        return DirichletContribution(self.node_idxs, self.points, self.value, values)


@dataclass(frozen=True)
class NeumannContribution:
    '''One resolved Neumann condition: a flux ∫_Γ g·v over a region's facets.

    `facet_mask` marks the boundary facets in the region; a `fem.loads.BoundaryLoad`
    over them integrates `value`, re-evaluated per time, across those facets alone.
    `nodal_values` is the value at the resolution time as a field over every node,
    which the residual estimator reads.
    '''
    facet_mask: BoolArray       # one entry per boundary facet
    node_idxs: VertexIndices    # the region's boundary nodes
    value: FieldValue           # the spec's value, possibly TimeDependent
    nodal_values: NodalValues   # (n_nodes, n_components) at the resolution time
    loaded: BoolArray           # (n_nodes, n_components) the components it drives


@dataclass(frozen=True)
class RobinContribution:
    '''One resolved Robin condition (∂u/∂n + κu = g on a region).

    It contributes to both sides of the system: the boundary term κ∫_Γ u·v to the
    operator (`kappa * BoundaryMassForm(facet_mask)`) and ∫_Γ g·v to the load (a
    `fem.loads.BoundaryLoad` over `value`). The assembly itself waits for a `FunctionSpace`,
    so this carries only the data, keyed to the node set.
    '''
    facet_mask: BoolArray       # one entry per boundary facet
    kappa: float
    node_idxs: VertexIndices    # the region's boundary nodes
    value: FieldValue           # the spec's g, possibly TimeDependent


# -- the conditions -----------------------------------------------------------------


@dataclass(frozen=True)
class Condition(ABC):
    '''One boundary condition on `region`, a callable over point coordinates (see
    `fem.regions`). Resolved lazily, so it means the same thing on any mesh.'''
    kind: ClassVar[str]
    region: Region

    def __post_init__(self) -> None:
        if not callable(self.region):
            raise TypeError(
                'region must be a callable over point coordinates; pass a helper '
                'from fem.regions (e.g. on_plane(0, 0.0)), or at_indices([...]) for '
                f'specific nodes. Got {type(self.region).__name__}.'
            )

    @property
    @abstractmethod
    def prescribed(self) -> FieldValue:
        '''The value the condition prescribes (the Dirichlet and Neumann `value`, the
        Robin `g`), for inspection and plotting.'''

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.prescribed, TimeDependent)

    @property
    def is_mesh_bound(self) -> bool:
        '''Whether the region names vertices of one mesh, so the condition cannot
        survive a remesh.'''
        return is_mesh_bound(self.region)

    def select(self, nodes: NodeGeometry) -> VertexIndices:
        '''Boundary nodes of `nodes` inside the region.

        Regions are evaluated over every node and then intersected with the
        boundary, which makes "a boundary condition on an interior node"
        unrepresentable rather than something to diagnose afterwards. For a P2 node
        set this picks up the edge-midpoint nodes on the boundary automatically,
        since they satisfy the same geometric region their endpoints do.
        '''
        if isinstance(self.region, on_tag):
            # Named by outline rather than by place: the facets say which nodes.
            return self.region.select_nodes(nodes.boundary, nodes.boundary_tags)
        selected = np.flatnonzero(self.region(nodes.vertices))
        boundary = np.asarray(nodes.boundary_idxs, dtype=int)
        if self.is_mesh_bound:
            # A named interior node is a mistake to report; a geometric region is a
            # filter, so its intersection with the boundary is the intent.
            interior = np.setdiff1d(selected, boundary)
            if len(interior):
                raise ValueError(
                    f'boundary conditions on non-boundary nodes: {sorted(interior)}'
                )
            return selected
        return np.intersect1d(selected, boundary)

    @abstractmethod
    def resolve(
        self, nodes: NodeGeometry, n_components: int, t: float = 0.0,
    ) -> 'DirichletContribution | NeumannContribution | RobinContribution':
        '''This condition's contribution on `nodes` at `n_components` DOFs per node,
        with a `TimeDependent` value taken at `t`.'''


@dataclass(frozen=True)
class Dirichlet(Condition):
    '''u = `value` on `region`: an essential condition, eliminated from the system.

    On a vector field a component may be `None` to leave it free rather than pinned:
    `[0, None]` pins x and leaves y natural, a roller rather than a clamp. A node may
    pick up its remaining component from a second, overlapping condition (one point
    elsewhere pinning y to remove the last rigid-body mode); the two merge rather than
    conflict as long as they agree on any component both specify.
    '''
    kind: ClassVar[str] = 'dirichlet'
    region: Region
    value: FieldValue

    @property
    def prescribed(self) -> FieldValue:
        return self.value

    def resolve(self, nodes: NodeGeometry, n_components: int, t: float = 0.0) -> DirichletContribution:
        idxs = self.select(nodes)
        points = nodes.vertices[idxs]
        values = _evaluate_dirichlet_value(field_at(self.value, t), points, n_components)
        return DirichletContribution(idxs, points, self.value, values)


@dataclass(frozen=True)
class Neumann(Condition):
    '''κ ∂u/∂n = `value` on `region`: the normal flux, a traction on an elastic
    boundary. A natural condition, integrated over the region's facets as a load.

    On a vector field a component may be `None` to say the traction does not drive
    it: `[g, None]` loads x and leaves y alone, integrating as zero there. That is
    how a traction shares a roller's nodes with the pinned component (see
    `Conditions.resolve`): a number on a pinned component conflicts unless it is
    known to vanish, which no `TimeDependent` value is.
    '''
    kind: ClassVar[str] = 'neumann'
    region: Region
    value: FieldValue

    @property
    def prescribed(self) -> FieldValue:
        return self.value

    def resolve(self, nodes: NodeGeometry, n_components: int, t: float = 0.0) -> NeumannContribution:
        idxs = self.select(nodes)
        mask = _facet_mask(nodes, idxs)
        if len(idxs) and not mask.any():
            raise ValueError(
                'a Neumann condition selects nodes but no boundary facet, so it '
                'integrates to nothing; a force at a node is a fem.loads.PointLoad'
            )
        raw = _coerce_components(field_at(self.value, t), nodes.vertices[idxs], n_components)
        if raw.shape != (len(idxs), n_components):
            raise ValueError(
                f'field must give {n_components} component(s) per point, got shape {raw.shape} '
                f'for {len(idxs)} point(s)'
            )
        values = np.zeros((len(nodes.vertices), n_components))
        values[idxs] = np.nan_to_num(raw, nan=0.0)
        return NeumannContribution(mask, idxs, self.value, values,
                                   _loaded_components(self.value, raw, idxs, values.shape))


def _loaded_components(value: FieldValue, raw: FloatArray, idxs: VertexIndices,
                       shape: tuple[int, ...]) -> BoolArray:
    '''(n_nodes, n_components): the components a Neumann value drives, read off the
    specification. A `None` component (NaN in `raw`) is never loaded. A constant or a
    callable of position loads where it is nonzero at the nodes, which is exact
    there; a `TimeDependent` loads every component it names, since a value that
    vanishes at one instant is no statement about the rest.'''
    loaded = np.zeros(shape, dtype=bool)
    named = ~np.isnan(raw)
    loaded[idxs] = named if isinstance(value, TimeDependent) else named & (raw != 0.0)
    return loaded


@dataclass(frozen=True)
class Robin(Condition):
    '''∂u/∂n + `kappa` u = `g` on `region`: a condition on both sides of the system,
    the boundary term κ∫u·v on the operator and ∫g·v on the load, both over the
    region's facets. `kappa` is a constant coefficient.'''
    kind: ClassVar[str] = 'robin'
    region: Region
    kappa: float
    g: FieldValue

    def __post_init__(self) -> None:
        super().__post_init__()
        if _has_free_component(self.g):
            raise ValueError('a Robin value has no free component; every component must be a number (None given)')

    @property
    def prescribed(self) -> FieldValue:
        return self.g

    def resolve(self, nodes: NodeGeometry, n_components: int, t: float = 0.0) -> RobinContribution:
        idxs = self.select(nodes)
        mask = _facet_mask(nodes, idxs)
        if len(idxs) and not mask.any():
            raise ValueError(
                'a Robin condition selects nodes but no boundary facet, so its '
                'boundary integral is empty'
            )
        # Evaluate now so a value that is wrong for this field fails at resolve time.
        evaluate_field(field_at(self.g, t), nodes.vertices[idxs], n_components)
        return RobinContribution(mask, float(self.kappa), idxs, self.g)
