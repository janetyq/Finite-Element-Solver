"""Load terms: the linear form L(v), each assembled as a vector.

A `Load` answers `vector(space, t)`, the DOF vector of `∫ f·v` for its own `f` at time `t`,
and `is_time_dependent`. A `ResolvedConditions` holds a tuple of them and the problem's load is their sum, so a
body force, a boundary flux, a Robin value, and a point force are four terms of one shape
rather than four branches in one function.

- `Source`: the volume load `∫ f·v`. A constant or a nodal array is integrated exactly
  through the mass matrix; a callable is sampled at the quadrature points, which captures
  variation within an element, or, with `nodal=True`, read at the nodes only and
  integrated as its interpolant (the comparison the convergence study draws).
- `BoundaryLoad`: a boundary integral over a region's facets, for a Neumann value or a
  Robin `g`, masked to those facets so a load stays on its own edge. Built by
  `Conditions.resolve`, not by a caller.
- `PointLoad`: a force applied at every node a region selects, no integral.

A field may be `TimeDependent`; each term fixes it at `t` before evaluating.

The space is imported lazily where a load resolves against one: `space` already imports
`loads` at top level, so this side of the edge stays function-local.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from fem.elements import ElementGeometry
from fem.physics.forms import BoundaryMassForm, sample_field
from fem.regions import TimeDependent, evaluate_field, field_at
from fem.typing import BoolArray, DofVector, FieldValue, FloatArray, IntArray, Operator, Region, NodalValues

if TYPE_CHECKING:
    from fem.space import FunctionSpace


class Load(Protocol):
    '''One term of the load: `∫ f·v` (or a point force) as a DOF vector at time `t`.'''

    @property
    def is_time_dependent(self) -> bool: ...

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector: ...


@dataclass(frozen=True)
class Source:
    '''Volume load L(v) = ∫ f(x)·v.

    A constant, a per-component constant, or a nodal array integrates exactly through
    the mass matrix; a callable of position is sampled at the quadrature points of a
    rule of `quadrature_degree`, one element vector per element that
    `FunctionSpace.assemble_load` scatters, so variation within an element is kept.
    `nodal=True` reads a callable at the nodes instead and integrates its interpolant,
    an approximation kept for comparison against the sampled path. `field` may be
    `TimeDependent`. The component count is the space's, read at assembly.
    '''
    field: FieldValue
    quadrature_degree: int = 2
    nodal: bool = False

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.field, TimeDependent)

    @property
    def is_sampled(self) -> bool:
        '''Whether `field` is read at the quadrature points (a callable, unless
        `nodal`) rather than integrated as its interpolant.'''
        return not self.nodal and (callable(self.field) or self.is_time_dependent)

    def at(self, t: float) -> 'Source':
        '''This source with a time-dependent field fixed at `t`; itself otherwise.'''
        if not self.is_time_dependent:
            return self
        return Source(field_at(self.field, t), self.quadrature_degree, self.nodal)

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        if not self.is_sampled:
            nodal = space.interpolate(field_at(self.field, t)).dofs
            return np.asarray(space.mass_matrix @ nodal).flatten()
        return space.assemble_load(self.at(t))

    def element_vectors(self, geometry: ElementGeometry, n_components: int) -> FloatArray:
        '''(n_elements, N*n_components) element load vectors, DOFs interleaved per node,
        with `field` sampled at `geometry`'s points.'''
        if self.is_time_dependent:
            raise TypeError('a time-dependent Source has no vectors without a time; use at(t)')
        f = sample_field(self.field, geometry, n_components)   # (n_el, n_qp, c)
        # b[e, n, c] = sum_q weight_detJ[e,q] * shape[q,n] * f[e,q,c]
        b = np.einsum('eq,qn,eqc->enc', geometry.weight_detJ, geometry.shape, f)
        return b.reshape(geometry.n_elements, -1)


@dataclass(frozen=True, eq=False)
class BoundaryLoad:
    '''A boundary load ∫_Γ g·v through a region-restricted boundary mass matrix, with
    `g` given by `value` on the nodes in `node_idxs` and zero elsewhere.

    A Neumann value (a flux, a traction) and a Robin `g` are both this term.
    `boundary_mass` is the assembled `BoundaryMassForm` over the region's facets, so it
    belongs to one space; a time-dependent value re-evaluates only the nodal values,
    never the integral.
    '''
    boundary_mass: Operator     # (n_dofs, n_dofs) masked boundary mass of the space
    node_idxs: IntArray         # the nodes the value is evaluated on
    value: FieldValue

    @classmethod
    def over(cls, space: 'FunctionSpace', facet_mask: BoolArray, node_idxs: IntArray,
             value: FieldValue) -> 'BoundaryLoad':
        '''The term over the facets `facet_mask` marks on `space`.'''
        mass = space.assemble(BoundaryMassForm(space.n_components, facet_mask))
        return cls(mass, node_idxs, value)

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.value, TimeDependent)

    def nodal_values(self, space: 'FunctionSpace', t: float = 0.0) -> NodalValues:
        '''`(n_nodes, n_components)` value of `g` at time `t`, zero off the region.'''
        g = np.zeros((space.n_nodes, space.n_components))
        if len(self.node_idxs):
            g[self.node_idxs] = evaluate_field(
                field_at(self.value, t), space.node_coords[self.node_idxs], space.n_components)
        return g

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        g = self.nodal_values(space, t).flatten()
        return np.asarray(self.boundary_mass @ g).flatten()


@dataclass(frozen=True)
class PointLoad:
    '''A force `force` applied at every node of the space that `region` selects.

    A nodal force, not an integral: each selected node's DOFs receive the force's
    components as they are, so a tip load on a beam is `PointLoad(at_indices([tip]),
    [0, -F])` or a geometric region selecting the one node. On a P2 space a region
    selects edge nodes too, so name the node rather than the edge for a single force.
    '''
    region: Region
    force: FieldValue

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.force, TimeDependent)

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        idxs = np.flatnonzero(self.region(space.node_coords))
        if len(idxs) == 0:
            raise ValueError('the point load region selects no node of the space')
        load = np.zeros((space.n_nodes, space.n_components))
        load[idxs] = evaluate_field(field_at(self.force, t), space.node_coords[idxs], space.n_components)
        return load.flatten()


@dataclass(frozen=True)
class _EvaluatedLoad:
    '''A load vector already evaluated: the snapshot of a time-dependent term at one time,
    which `ResolvedConditions.at` builds.'''
    values: DofVector

    @property
    def is_time_dependent(self) -> bool:
        return False

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        return self.values


def total_load(terms: 'tuple[Load, ...]', space: 'FunctionSpace', t: float = 0.0) -> DofVector:
    '''The sum of the terms' vectors at time `t`; zero for no terms.'''
    load = np.zeros(space.n_dofs)
    for term in terms:
        load = load + term.vector(space, t)
    return load
