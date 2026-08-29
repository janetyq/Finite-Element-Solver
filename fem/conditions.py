"""Everything applied to a domain, specified against geometry and resolved on a space.

A `Conditions` is the frozen, mesh-independent collection of what acts on a problem:
the boundary conditions (`Dirichlet`, `Neumann`, `Robin`, each on a region), the volume
`Source`, and any `PointLoad`. It means the same thing on any mesh ("the left edge is
pinned, gravity acts, a force at the tip"), so a driver can restate it on a refined
mesh, and it is what `equation.problem(mesh, conditions)` takes.

`resolve(space)` reduces it to a `ResolvedConditions`, what a solver indexes into on
one space: the Dirichlet DOF partition, the operator terms a Robin condition adds, and
the load terms (the source, one boundary integral per Neumann or Robin value, the point
loads). Resolution has two steps: the geometry (which nodes and facets a region selects)
is done once, and the values are evaluated at a time, so `at(t)` re-evaluates a
`TimeDependent` Dirichlet value without selecting again; the load terms re-evaluate
themselves per time (`Load.vector(space, t)`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from fem.boundary import (
    Condition,
    Dirichlet,
    DirichletContribution,
    Neumann,
    NeumannContribution,
    NodeGeometry,
    Robin,
    RobinContribution,
    _coerce_components,
    field_at,
)
from fem.loads import BoundaryLoad, Load, PointLoad, Source
from fem.physics.forms import BoundaryMassForm, Form
from fem.regions import TimeDependent
from fem.typing import DofIndices, FloatArray, VertexField, VertexIndices

if TYPE_CHECKING:
    from fem.space import FunctionSpace

__all__ = ['Conditions', 'ResolvedConditions']


@dataclass(frozen=True, init=False)
class Conditions:
    '''The conditions and loads on a domain: a frozen tuple of `Condition`s and `Load`s
    built with the variadic constructor, or by `conditions + item`.

    At most one volume `Source`, since the residual estimator reads *the* pointwise
    source; a second one is refused rather than summed.
    '''
    items: tuple[Condition | Load, ...]

    def __init__(self, *items: Condition | Load | Conditions) -> None:
        flat: list[Condition | Load] = []
        for item in items:
            if isinstance(item, Conditions):
                flat.extend(item.items)
            elif isinstance(item, (Condition, Source, PointLoad)):
                flat.append(item)
            else:
                raise TypeError(
                    'expected a Dirichlet, Neumann, or Robin condition, a Source, or a '
                    f'PointLoad, got {type(item).__name__}'
                )
        sources = [item for item in flat if isinstance(item, Source)]
        if len(sources) > 1:
            raise ValueError(f'one volume source at most; got {sources[0]} and {sources[1]}')
        object.__setattr__(self, 'items', tuple(flat))

    def __add__(self, other: Condition | Load | Conditions) -> Conditions:
        return Conditions(*self.items, other)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    # -- views -------------------------------------------------------------------------

    @property
    def boundary(self) -> tuple[Condition, ...]:
        return tuple(item for item in self.items if isinstance(item, Condition))

    @property
    def dirichlet(self) -> tuple[Dirichlet, ...]:
        return tuple(item for item in self.items if isinstance(item, Dirichlet))

    @property
    def neumann(self) -> tuple[Neumann, ...]:
        return tuple(item for item in self.items if isinstance(item, Neumann))

    @property
    def robin(self) -> tuple[Robin, ...]:
        return tuple(item for item in self.items if isinstance(item, Robin))

    @property
    def source(self) -> Source | None:
        '''The volume source, or None.'''
        return next((item for item in self.items if isinstance(item, Source)), None)

    @property
    def loads(self) -> tuple[Load, ...]:
        '''The point loads: the load terms beyond the source and the boundary values.'''
        return tuple(item for item in self.items if isinstance(item, PointLoad))

    @property
    def is_time_dependent(self) -> bool:
        '''Whether any condition or load carries a `TimeDependent` field.'''
        return any(item.is_time_dependent for item in self.items)

    @property
    def has_time_dependent_dirichlet(self) -> bool:
        '''Whether a prescribed value moves in time, which needs its velocity and
        acceleration too (what `NewmarkMethod` cannot take); a time-dependent load is
        re-evaluated per step and is fine.'''
        return any(d.is_time_dependent for d in self.dirichlet)

    @property
    def is_mesh_bound(self) -> bool:
        '''Whether any item is tied to one mesh's vertex numbering, and so cannot be
        carried across a remesh.'''
        return any(c.is_mesh_bound for c in self.boundary)

    def check_remeshable(self) -> None:
        if self.is_mesh_bound:
            raise NotImplementedError(
                'this specification uses at_indices, which names vertices of one '
                'specific mesh and cannot survive the renumbering a remesh does. '
                'Describe the region geometrically (see fem.regions) to make it '
                'remeshable.'
            )

    def entries(self, nodes: NodeGeometry) -> list[tuple[Condition, VertexIndices, FloatArray]]:
        '''[(condition, node_idxs, values), ...] for the boundary conditions, resolved
        against `nodes`.

        Region resolution only, no DOF numbering, so this needs no component count and
        is what inspection and plotting use. Values are shown one component per column
        as given, a free component as NaN, and a time-dependent value at t = 0.
        '''
        out = []
        for condition in self.boundary:
            idxs = condition.select(nodes)
            values = (_coerce_components(field_at(condition.prescribed, 0.0), nodes.vertices[idxs], 1)
                      if len(idxs) else np.zeros((0, 1)))
            out.append((condition, idxs, values))
        return out

    # -- resolution ----------------------------------------------------------------------

    def resolve(self, space: FunctionSpace, t: float = 0.0) -> ResolvedConditions:
        '''Reduce this specification to what a solver on `space` indexes into, with any
        `TimeDependent` value taken at time `t`.'''
        nodes, n_components = space.nodes, space.n_components
        n = len(nodes.vertices)
        dirichlet: list[DirichletContribution] = []
        neumann: list[NeumannContribution] = []
        robin: list[RobinContribution] = []
        for condition in self.boundary:
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

        # A Robin condition contributes to both sides: κ∫_Γ u·v on the operator and
        # ∫_Γ g·v on the load, each over the region's facets. A Neumann value is a load
        # over its own facets, one masked boundary integral per condition so a traction
        # stays on its edge instead of spreading through a shared corner node.
        operator_terms = tuple(
            r.kappa * BoundaryMassForm(n_components, r.facet_mask) for r in robin)
        source = self.source
        if source is not None:
            source = replace(source, n_components=n_components)
        loads: list[Load] = [] if source is None else [source]
        loads += [BoundaryLoad.over(space, c.facet_mask, c.node_idxs, c.value) for c in neumann]
        loads += [BoundaryLoad.over(space, c.facet_mask, c.node_idxs, c.value) for c in robin]
        loads += list(self.loads)
        return ResolvedConditions(
            n_vertices=n, n_components=n_components,
            fixed_idxs=fixed_idxs, free_idxs=free_idxs, fixed_values=fixed_values,
            dirichlet=tuple(dirichlet), neumann=tuple(neumann), robin=tuple(robin),
            operator_terms=operator_terms, source=source, loads=tuple(loads),
        )


@dataclass(frozen=True)
class ResolvedConditions:
    '''Conditions reduced to what a solver on one space indexes into: the Dirichlet
    partition, the operator terms, the load terms, and one contribution per boundary
    condition.

    Frozen and built per space so it cannot drift out of step with it. `at(t)` is the
    same resolution with every time-dependent Dirichlet value re-evaluated at `t`; the
    load terms carry their own values and are re-evaluated by `Problem.load_at`.
    '''
    n_vertices: int
    n_components: int
    fixed_idxs: DofIndices      # DOF indices held by Dirichlet conditions
    free_idxs: DofIndices       # the complement
    fixed_values: FloatArray    # values at fixed_idxs, same order
    dirichlet: tuple[DirichletContribution, ...] = ()
    neumann: tuple[NeumannContribution, ...] = ()
    robin: tuple[RobinContribution, ...] = ()
    operator_terms: tuple[Form, ...] = ()   # κ boundary mass per Robin condition
    source: Source | None = None            # the volume source, sized to the space
    loads: tuple[Load, ...] = ()            # source, boundary integrals, point loads

    @property
    def constraints(self) -> tuple[DofIndices, DofIndices, FloatArray]:
        '''`(free_idxs, fixed_idxs, fixed_values)`, the partition `DiscreteSystem` takes.'''
        return self.free_idxs, self.fixed_idxs, self.fixed_values

    @property
    def neumann_load(self) -> VertexField:
        '''`(n_nodes, n_components)` the Neumann values summed as one nodal field, at
        the resolution time.'''
        total = np.zeros((self.n_vertices, self.n_components))
        for neumann in self.neumann:
            total += neumann.nodal_values
        return total

    def at(self, t: float) -> ResolvedConditions:
        '''This resolution with the time-dependent Dirichlet values taken at `t`.'''
        if not any(isinstance(d.value, TimeDependent) for d in self.dirichlet):
            return self
        dirichlet = tuple(d.at(t) for d in self.dirichlet)
        fixed_idxs, fixed_values, free_idxs = _partition(self.n_vertices, self.n_components, dirichlet)
        return replace(self, fixed_idxs=fixed_idxs, free_idxs=free_idxs,
                       fixed_values=fixed_values, dirichlet=dirichlet)


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
