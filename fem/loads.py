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
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from fem.elements import ElementGeometry
from fem.physics.forms import BoundaryMassForm, sample_field
from fem.regions import TimeDependent, evaluate_field, field_at
from fem.typing import BoolArray, DofVector, FieldValue, FloatArray, IntArray, NodalValues, Operator, Region

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

    Integrated element by element, one element vector per element that
    `FunctionSpace.assemble_load` scatters. A constant or a per-component constant
    multiplies the integral of each shape function over its element, exact at any rule;
    a callable of position is sampled at the quadrature points of a rule of
    `quadrature_degree`, so variation within an element is kept. `nodal=True` reads a
    callable at the nodes instead and integrates its interpolant through the mass
    matrix, an approximation kept for comparison against the sampled path. `field` may
    be `TimeDependent`. The component count is the space's, read at assembly.
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
        `nodal`) rather than taken as a constant or integrated as its interpolant.'''
        return not self.nodal and (callable(self.field) or self.is_time_dependent)

    @property
    def is_interpolated(self) -> bool:
        '''Whether `field` is integrated as its nodal interpolant through the mass
        matrix: a callable read with `nodal=True`. A constant needs neither sampling nor
        the mass matrix and integrates element by element.'''
        return self.nodal and (callable(self.field) or self.is_time_dependent)

    def at(self, t: float) -> 'Source':
        '''This source with a time-dependent field fixed at `t`; itself otherwise.'''
        if not self.is_time_dependent:
            return self
        return Source(field_at(self.field, t), self.quadrature_degree, self.nodal)

    def vector(self, space: 'FunctionSpace', t: float = 0.0) -> DofVector:
        if self.is_interpolated:
            nodal = space.interpolate(field_at(self.field, t)).dofs
            return np.asarray(space.mass_matrix @ nodal).flatten()
        return space.assemble_load(self.at(t))

    def element_vectors(self, geometry: ElementGeometry, n_components: int) -> FloatArray:
        '''(n_elements, N*n_components) element load vectors, DOFs interleaved per node,
        with a callable `field` sampled at `geometry`'s points.'''
        if self.is_time_dependent:
            raise TypeError('a time-dependent Source has no vectors without a time; use at(t)')
        if not callable(self.field):
            # A constant factors out of the integral: each shape function's integral
            # over the element, `sum_q weight_detJ[e,q] shape[q,n]`, times the value.
            # Exact at any rule, and it never touches the mass matrix, whose assembly
            # would cost a full k x k block scatter for one vector.
            value = evaluate_field(self.field, geometry.points[0, :1], n_components)[0]   # (c,)
            shape_integral = geometry.weight_detJ @ geometry.shape                        # (n_el, N)
            return (shape_integral[:, :, None] * value).reshape(geometry.n_elements, -1)
        f = sample_field(self.field, geometry, n_components)   # (n_el, n_qp, c)
        # b[e, n, c] = sum_q weight_detJ[e,q] * shape[q,n] * f[e,q,c], weighted per point
        # first so the contraction is one matrix product rather than a three-way loop.
        b = np.einsum('eqc,qn->enc', geometry.weight_detJ[..., None] * f, geometry.shape)
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
                field_at(self.value, t), space.node_coords[self.node_idxs], space.n_components,
                free_as_zero=True)
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
