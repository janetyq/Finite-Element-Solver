"""Load terms: the linear form L(v), each assembled as a vector.

A `Load` answers `vector(space, t)`, the DOF vector of `∫ f·v` for its own `f` at time `t`,
and `is_time_dependent`. A `Problem` holds a tuple of them and its load is their sum, so a
body force, a boundary flux, a Robin value, and a point force are four terms of one shape
rather than four branches in one function.

- `Source`: the volume load `∫ f·v`. A constant or a nodal array is integrated exactly
  through the mass matrix; a callable is sampled at the quadrature points, which captures
  variation within an element.
- `NodalSource`: the volume load of `f`'s nodal interpolant, for a callable that should
  be read at the nodes only (a comparison against the sampled path).
- `BoundaryLoad`: a boundary integral over a region's facets, for a Neumann value or a
  Robin `g`, masked to those facets so a load stays on its own edge.
- `PointLoad`: a force applied at every node a region selects, no integral.

A field may be `TimeDependent`; each term fixes it at `t` before evaluating.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from fem.elements import ElementGeometry
from fem.forms import BoundaryMassForm, sample_field
from fem.regions import TimeDependent, evaluate_field, field_at
from fem.typing import BoolArray, DofVector, FieldValue, FloatArray, IntArray, Operator, Region, VertexField

if TYPE_CHECKING:
    from fem.space import FunctionSpace


class Load(Protocol):
    '''One term of the load: `∫ f·v` (or a point force) as a DOF vector at time `t`.'''

    @property
    def is_time_dependent(self) -> bool: ...

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector: ...


@dataclass(frozen=True)
class NodalSource:
    '''Volume load L(v) = ∫ f·v integrated as f's nodal interpolant through the mass
    matrix: exact for a constant or a nodal array, an approximation for a callable.'''
    field: FieldValue = None

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.field, TimeDependent)

    def at(self, t: float) -> 'NodalSource':
        return NodalSource(field_at(self.field, t)) if self.is_time_dependent else self

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        nodal = space.interpolate(field_at(self.field, t))
        return np.asarray(space.mass_matrix @ nodal).flatten()


@dataclass(frozen=True)
class Source:
    '''Volume load L(v) = ∫ f(x)·v.

    A constant, a per-component constant, or a nodal array integrates exactly through
    the mass matrix (`NodalSource`); a callable of position is sampled at the quadrature
    points of a rule of `quadrature_degree`, one element vector per element that
    `FunctionSpace.assemble_load` scatters, so variation within an element is kept.
    `field` may be `TimeDependent`.
    '''
    field: FieldValue
    n_components: int = 1
    quadrature_degree: int = 2

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.field, TimeDependent)

    @property
    def is_sampled(self) -> bool:
        '''Whether `field` is read at the quadrature points (a callable) rather than
        integrated as its interpolant.'''
        return callable(self.field) or self.is_time_dependent

    def at(self, t: float) -> 'Source':
        '''This source with a time-dependent field fixed at `t`; itself otherwise.'''
        if not self.is_time_dependent:
            return self
        return Source(field_at(self.field, t), self.n_components, self.quadrature_degree)

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        if not self.is_sampled:
            return NodalSource(self.field).vector(space, t)
        return space.assemble_load(self.at(t))

    def element_vectors(self, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, N*n_components) element load vectors, DOFs interleaved per node,
        with `field` sampled at `geometry`'s points.'''
        if self.is_time_dependent:
            raise TypeError('a time-dependent Source has no vectors without a time; use at(t)')
        f = sample_field(self.field, geometry, self.n_components)   # (n_el, n_qp, c)
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

    def nodal_values(self, space: 'FunctionSpace', t: float = 0.0) -> VertexField:
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
class EvaluatedLoad:
    '''A load vector already evaluated: the snapshot of a time-dependent term at one time.'''
    values: DofVector

    @property
    def is_time_dependent(self) -> bool:
        return False

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        return self.values


VolumeSource = Source | NodalSource


def as_source(source: 'FieldValue | VolumeSource', n_components: int) -> 'VolumeSource | None':
    '''Normalize a `Problem` source into its volume load term, or None for no source.

    A `Source` or `NodalSource` is taken as it is; any other value is wrapped as a
    `Source`, which integrates a constant exactly and samples a callable. Any other load
    term (a `PointLoad`) is a `Problem` `loads` entry, not a source.
    '''
    if source is None:
        return None
    if isinstance(source, (Source, NodalSource)):
        return source
    return Source(source, n_components=n_components)


def total_load(terms: 'tuple[Load, ...]', space: 'FunctionSpace', t: float = 0.0) -> DofVector:
    '''The sum of the terms' vectors at time `t`; zero for no terms.'''
    load = np.zeros(space.n_dofs)
    for term in terms:
        load = load + term.vector(space, t)
    return load
