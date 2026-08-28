"""Boundary conditions, specified against geometry and resolved against a mesh.

A condition is a frozen object: `Dirichlet(region, value)`, `Neumann(region, value)`,
or `Robin(region, kappa, g)`, each a `Condition` that knows how to resolve itself. A
`BoundaryConditions` is the frozen collection of them, a mesh-independent specification
("the left edge is pinned") that means the same thing on any mesh. A `ResolvedBC` is what
a solver needs: the Dirichlet DOF partition and one contribution per load-bearing
condition, for one node set and one number of DOFs per node. Resolution has two steps:
the geometry (which nodes and facets a region selects) is done once, and the values are
evaluated at a time, so `ResolvedBC.at(t)` re-evaluates a `TimeDependent` value without
selecting again.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

from fem.regions import (
    TimeDependent,
    as_field,
    evaluate_field,
    field_at,
    is_mesh_bound,
    sample_natural_width,
)
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
    return as_field(value, n_components, allow_free=True).sample(points)


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
    nodal_values: VertexField   # (n_nodes, n_components) at the resolution time


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
    Every component must be a number; a load has no free component.'''
    kind: ClassVar[str] = 'neumann'
    region: Region
    value: FieldValue

    def __post_init__(self) -> None:
        super().__post_init__()
        if _has_free_component(self.value):
            raise ValueError('a flux has no free component; every component must be a number (None given)')

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
        values = np.zeros((len(nodes.vertices), n_components))
        values[idxs] = evaluate_field(field_at(self.value, t), nodes.vertices[idxs], n_components)
        return NeumannContribution(mask, idxs, self.value, values)


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


# -- the resolution ------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBC:
    '''Boundary conditions reduced to what a solver indexes into: the Dirichlet
    partition and one contribution per condition.

    Frozen and built per (node set, n_components) so it cannot drift out of step with
    either. `at(t)` is the same resolution with every time-dependent Dirichlet value
    re-evaluated at `t`; the load-bearing contributions carry their own values and are
    re-evaluated by the `BoundaryLoad` terms a `Problem` builds from them.
    '''
    n_vertices: int
    n_components: int
    fixed_idxs: DofIndices      # DOF indices held by Dirichlet conditions
    free_idxs: DofIndices       # the complement
    fixed_values: FloatArray    # values at fixed_idxs, same order
    dirichlet: tuple[DirichletContribution, ...] = ()
    neumann: tuple[NeumannContribution, ...] = ()
    robin: tuple[RobinContribution, ...] = ()

    @property
    def neumann_load(self) -> VertexField:
        '''`(n_nodes, n_components)` the Neumann values summed as one nodal field, at
        the resolution time.'''
        total = np.zeros((self.n_vertices, self.n_components))
        for neumann in self.neumann:
            total += neumann.nodal_values
        return total

    def at(self, t: float) -> 'ResolvedBC':
        '''This resolution with the time-dependent Dirichlet values taken at `t`.'''
        if not any(isinstance(d.value, TimeDependent) for d in self.dirichlet):
            return self
        dirichlet = tuple(d.at(t) for d in self.dirichlet)
        fixed_idxs, fixed_values, free_idxs = _partition(self.n_vertices, self.n_components, dirichlet)
        return ResolvedBC(self.n_vertices, self.n_components, fixed_idxs, free_idxs, fixed_values,
                          dirichlet, self.neumann, self.robin)


def _merge_dirichlet(
    contributions: tuple[DirichletContribution, ...],
) -> dict[int, FloatArray]:
    '''Per-node Dirichlet values, merged across overlapping conditions.

    Overlapping regions are normal (a corner belongs to two edges, or, for a roller,
    an edge and the one point that pins its other component); a component that both
    conditions specify but disagree on is a real conflict, and last-write-wins would
    bury it. A component either side leaves free (NaN) never conflicts; the other
    side's value (fixed or itself free) wins.
    '''
    merged: dict[int, FloatArray] = {}
    for contribution in contributions:
        for v_idx, v in zip(contribution.node_idxs, contribution.values):
            v_idx = int(v_idx)
            if v_idx in merged:
                existing = merged[v_idx]
                both_given = ~np.isnan(existing) & ~np.isnan(v)
                if both_given.any() and not np.allclose(existing[both_given], v[both_given]):
                    raise ValueError(
                        f'conflicting Dirichlet values at vertex {v_idx}: {existing} and {v}'
                    )
                v = np.where(np.isnan(v), existing, v)
            merged[v_idx] = v
    return merged


def _partition(
    n: int, n_components: int, contributions: tuple[DirichletContribution, ...],
) -> tuple[DofIndices, FloatArray, DofIndices]:
    '''`(fixed_idxs, fixed_values, free_idxs)` from the merged Dirichlet values.

    Per (node, component): a NaN entry is a component a condition left free (a
    roller's tangential direction, say), so it contributes no fixed DOF; free_idxs,
    being the complement over the whole DOF range, picks it up.
    '''
    merged = _merge_dirichlet(contributions)
    fixed_idxs = np.array(
        [n_components * v + d for v in sorted(merged) for d in range(n_components)
         if not np.isnan(merged[v][d])],
        dtype=int,
    )
    fixed_values = np.array(
        [merged[v][d] for v in sorted(merged) for d in range(n_components)
         if not np.isnan(merged[v][d])],
        dtype=float,
    )
    free_idxs = np.setdiff1d(np.arange(n * n_components), fixed_idxs)
    return fixed_idxs, fixed_values, free_idxs


@dataclass(frozen=True, init=False)
class BoundaryConditions:
    '''A mesh-independent specification of the conditions on a domain boundary: a
    frozen tuple of `Condition`s built with the variadic constructor. Because it is
    iterable, extend or merge specs by unpacking: `BoundaryConditions(*existing, extra)`
    or `BoundaryConditions(*case_a, *case_b)`.'''
    conditions: tuple[Condition, ...]

    def __init__(self, *conditions: Condition) -> None:
        for condition in conditions:
            if not isinstance(condition, Condition):
                raise TypeError(
                    f'expected a Dirichlet, Neumann, or Robin condition, got {type(condition).__name__}'
                )
        object.__setattr__(self, 'conditions', tuple(conditions))

    def __iter__(self):
        return iter(self.conditions)

    def __len__(self) -> int:
        return len(self.conditions)

    @property
    def is_time_dependent(self) -> bool:
        '''Whether any condition's value is a `TimeDependent` field.'''
        return any(c.is_time_dependent for c in self.conditions)

    @property
    def is_mesh_bound(self) -> bool:
        '''Whether any condition is tied to one mesh's vertex numbering, and so
        cannot be carried across a remesh.'''
        return any(c.is_mesh_bound for c in self.conditions)

    def check_remeshable(self) -> None:
        if self.is_mesh_bound:
            raise NotImplementedError(
                'this specification uses at_indices, which names vertices of one '
                'specific mesh and cannot survive the renumbering a remesh does. '
                'Describe the region geometrically (see fem.regions) to make it '
                'remeshable.'
            )

    def entries(self, nodes: NodeGeometry) -> list[tuple[Condition, VertexIndices, FloatArray]]:
        '''[(condition, node_idxs, values), ...] resolved against `nodes`.

        Region resolution only, no DOF numbering, so this needs no `n_components` and is
        what inspection and plotting use. Values are shown one component per column as
        given, a free component as NaN, and a time-dependent value at t = 0.
        '''
        out = []
        for condition in self.conditions:
            idxs = condition.select(nodes)
            values = sample_natural_width(field_at(condition.prescribed, 0.0), nodes.vertices[idxs])
            out.append((condition, idxs, values))
        return out

    def resolve(self, nodes: NodeGeometry, n_components: int, t: float = 0.0) -> ResolvedBC:
        '''Reduce this specification to a `ResolvedBC` for `nodes` at `n_components` DOFs
        per node, with any `TimeDependent` value taken at time `t`.'''
        n = len(nodes.vertices)
        dirichlet: list[DirichletContribution] = []
        neumann: list[NeumannContribution] = []
        robin: list[RobinContribution] = []
        for condition in self.conditions:
            contribution = condition.resolve(nodes, n_components, t)
            if isinstance(contribution, DirichletContribution):
                dirichlet.append(contribution)
            elif isinstance(contribution, NeumannContribution):
                neumann.append(contribution)
            else:
                robin.append(contribution)

        merged = _merge_dirichlet(tuple(dirichlet))
        # A fixed DOF ignores any traction on it, so the ambiguity to reject is per
        # (node, component): a component that is both pinned and loaded. Pinning one
        # component while a traction drives a different one (a roller carrying a
        # tangential load) is well-posed and allowed; the fixed component is eliminated
        # by `DiscreteSystem`, dropping its traction, and the free ones keep theirs.
        loaded = np.zeros((n, n_components))
        for contribution in neumann:
            loaded += contribution.nodal_values
        conflicts = [
            v for v, values in merged.items()
            if np.any(~np.isnan(values) & (loaded[v] != 0.0))
        ]
        if conflicts:
            raise ValueError(
                'vertices carry a Dirichlet and a Neumann condition on the same '
                f'component: {sorted(conflicts)}'
            )

        fixed_idxs, fixed_values, free_idxs = _partition(n, n_components, tuple(dirichlet))
        return ResolvedBC(
            n_vertices=n,
            n_components=n_components,
            fixed_idxs=fixed_idxs,
            free_idxs=free_idxs,
            fixed_values=fixed_values,
            dirichlet=tuple(dirichlet),
            neumann=tuple(neumann),
            robin=tuple(robin),
        )
