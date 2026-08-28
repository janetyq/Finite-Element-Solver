"""Load terms: the linear form L(v), each assembled as a vector.

A `Load` answers `vector(space, t)`, the DOF vector of `∫ f·v` for its own `f` at time `t`,
and `is_time_dependent`. A `Problem` holds a tuple of them and its load is their sum, so a
body force, a traction, a Robin value, and a point force are four terms of one shape rather
than four branches in one function.

- `Source`: a volume load given at the nodes (a constant or a nodal array), integrated as
  its interpolant through the mass matrix.
- `LinearForm` (in `fem.forms`): a volume load sampled at the quadrature points, which
  captures variation within an element.
- `Traction`: a boundary integral over a region's facets, for a Neumann traction or a
  Robin value `g`, masked to those facets so a load stays on its own edge.
- `PointLoad`: a force applied at every node a region selects, no integral.

A field may be `TimeDependent`; each term fixes it at `t` before evaluating.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from fem.forms import LinearForm, MaskedMassForm
from fem.regions import TimeDependent, evaluate_field, field_at
from fem.typing import BoolArray, DofVector, FieldValue, IntArray, Region

if TYPE_CHECKING:
    from fem.space import FunctionSpace


class Load(Protocol):
    '''One term of the load: `∫ f·v` (or a point force) as a DOF vector at time `t`.'''

    @property
    def is_time_dependent(self) -> bool: ...

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector: ...


@dataclass(frozen=True)
class Source:
    '''Volume load L(v) = ∫ f·v integrated as f's nodal interpolant, f a constant, a
    per-component constant, or a nodal array. Pass one to `Problem` to ask for that
    path explicitly; a bare callable is sampled at the quadrature points instead
    (`LinearForm`).'''
    field: FieldValue = None

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.field, TimeDependent)

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        nodal = space.interpolate(field_at(self.field, t))
        return np.asarray(space.mass_matrix @ nodal).flatten()


@dataclass(frozen=True, eq=False)
class Traction:
    '''A boundary load ∫_Γ g·v over the facets in `facet_mask`, with `g` given by
    `value` on the nodes in `node_idxs` and zero elsewhere.

    The Neumann traction and the Robin `g` are both this term. The masked boundary mass
    is assembled on first use and held, so a time-dependent value re-evaluates only
    the nodal values, never the integral.
    '''
    facet_mask: BoolArray   # one entry per boundary facet of the space
    node_idxs: IntArray     # the nodes the value is evaluated on
    value: FieldValue
    _mass: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_time_dependent(self) -> bool:
        return isinstance(self.value, TimeDependent)

    def boundary_mass(self, space: 'FunctionSpace'):
        '''The region-restricted boundary mass matrix on `space`, held after the first call.'''
        if self._mass.get('space') is not space:
            self._mass['space'] = space
            self._mass['matrix'] = space.assemble(MaskedMassForm(space.n_components, self.facet_mask))
        return self._mass['matrix']

    def nodal_values(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        '''`(n_nodes, n_components)` value of `g` at time `t`, zero off the region.'''
        g = np.zeros((space.n_nodes, space.n_components))
        if len(self.node_idxs):
            g[self.node_idxs] = evaluate_field(
                field_at(self.value, t), space.node_coords[self.node_idxs], space.n_components)
        return g

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        g = self.nodal_values(space, t).flatten()
        return np.asarray(self.boundary_mass(space) @ g).flatten()


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
class FixedLoad:
    '''A load vector already evaluated: the snapshot of a time-dependent term at one time.'''
    values: DofVector

    @property
    def is_time_dependent(self) -> bool:
        return False

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        return self.values


def as_load(source: 'FieldValue | LinearForm | Load', n_components: int) -> 'Load | None':
    '''Normalize a `Problem` source into one load term, or None for no source.

    A `Load` (a `Source`, `LinearForm`, `Traction`, `PointLoad`) is taken as it is. A
    callable of position, or a `TimeDependent`, is sampled at the quadrature points as a
    `LinearForm`; a constant or a nodal array is a `Source`.
    '''
    if source is None:
        return None
    if isinstance(source, (Source, LinearForm, Traction, PointLoad, FixedLoad)):
        return source
    if hasattr(source, 'vector') and hasattr(source, 'is_time_dependent'):
        return cast(Load, source)
    field = cast(FieldValue, source)
    if callable(field) or isinstance(field, TimeDependent):
        return LinearForm(field, n_components=n_components)
    return Source(field)


def total_load(terms: 'tuple[Load, ...]', space: 'FunctionSpace', t: float = 0.0) -> DofVector:
    '''The sum of the terms' vectors at time `t`; zero for no terms.'''
    load = np.zeros(space.n_dofs)
    for term in terms:
        load = load + term.vector(space, t)
    return load
