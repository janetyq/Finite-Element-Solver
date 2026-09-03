"""`NodalField`: DOF values that belong to a `FunctionSpace`.

A DOF vector on its own is a bare array; which node each entry belongs to, how many
components a node has, and where the nodes sit are all facts of the space. A
`NodalField` pairs the two, so a solved field, an initial condition, a comparison
field, and a mode shape are one kind of thing, and every operation that needs the
space beside the values (reading them by node or component, integrating, evaluating
at a point, warping the mesh) lives here rather than at each call site.

The values are read-only, as a `Mesh`'s arrays are: a solution, a plot, and a series
step can share one field without a defensive copy. `np.asarray(field)` is its DOF
vector, so every numeric consumer (a `DiscreteSystem`, an integrator's initial state,
a residual) takes a field or a raw vector alike.

Per-element data (a stress, a flux, a density) is not a `NodalField`; it stays an
`ElementValues` array, one row per element, and is recovered onto the nodes by
`fem.post.recovery` when a continuous field is wanted.

`boundary_integral` imports `fem.physics.forms` lazily for its boundary mass form: the
forms sit above the field, so the edge points up and stays function-local.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fem.typing import DofVector, ElementValues, FloatArray, NodalValues, Region, Vertices

if TYPE_CHECKING:
    from fem.elements import Element
    from fem.mesh.mesh import Mesh
    from fem.space import FunctionSpace


@dataclass(frozen=True, eq=False)
class NodalField:
    '''DOF values on a `FunctionSpace`: `dofs[n_components * node + c]` is component
    `c` at the space's node `node` (vertices first, then any edge nodes).

    `dofs` is checked against the space's size on construction, so a P1 vector handed
    to a P2 space, or a scalar vector to a vector space, fails here rather than as a
    reshape error downstream. The array is copied and frozen.
    '''
    space: 'FunctionSpace'
    dofs: DofVector

    def __post_init__(self) -> None:
        dofs = np.array(self.dofs, dtype=float)
        if dofs.ndim != 1 or len(dofs) != self.space.n_dofs:
            raise ValueError(
                f'a field on {self.space!r} needs {self.space.n_dofs} DOFs, '
                f'got shape {dofs.shape}'
            )
        dofs.setflags(write=False)
        object.__setattr__(self, 'dofs', dofs)

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        return self.dofs if dtype is None else self.dofs.astype(dtype)

    def __len__(self) -> int:
        return len(self.dofs)

    # -- what the space knows ------------------------------------------------

    @property
    def mesh(self) -> 'Mesh':
        return self.space.mesh

    @property
    def n_components(self) -> int:
        return self.space.n_components

    @property
    def n_nodes(self) -> int:
        return self.space.n_nodes

    @property
    def element_type(self) -> 'type[Element]':
        return self.space.element_type

    # -- the values by node ---------------------------------------------------

    @property
    def nodal_values(self) -> NodalValues:
        '''The values by node: `(n_nodes,)` for a scalar field, `(n_nodes, n_components)`
        for a vector one.'''
        values = self.dofs.reshape(-1, self.n_components)
        return values[:, 0] if self.n_components == 1 else values

    def component(self, c: int) -> NodalValues:
        '''`(n_nodes,)` component `c` at every node.'''
        if not 0 <= c < self.n_components:
            raise IndexError(f'component {c} of a field with {self.n_components} components')
        return self.dofs[c::self.n_components]

    @property
    def element_values(self) -> FloatArray:
        '''The values gathered per element: `(n_elements, N)` for a scalar field,
        `(n_elements, N, n_components)` for a vector one, the layout a `Form` and
        `ElementGeometry.gradients` take.'''
        return self.nodal_values[self.space.element_nodes]

    # -- integrals and derivatives -------------------------------------------

    def integrate(self) -> float | FloatArray:
        '''The integral over the domain, exact for the discrete field: a float for a
        scalar field, `(n_components,)` for a vector one.'''
        integrals = np.asarray(self.space.nodal_mass_matrix @ self.nodal_values).sum(axis=0)
        return float(integrals) if self.n_components == 1 else integrals

    def mean(self) -> float | FloatArray:
        '''The volume-weighted mean; see `integrate`.'''
        return self.integrate() / self.space.total_volume

    def boundary_integral(self, region: 'Region | None' = None) -> float | FloatArray:
        '''The integral over the boundary, or over the boundary facets in `region`,
        exact for the discrete field: a float for a scalar field, `(n_components,)`
        for a vector one.

        A facet is in the region when all its nodes are, the rule a Neumann or Robin
        condition integrates by, so the integral over a condition's region is the
        integral the condition sees.
        '''
        from fem.boundary import boundary_facet_mask
        from fem.physics.forms import BoundaryMassForm

        n_facets = len(self.space.boundary_nodes)
        mask = (np.ones(n_facets, dtype=bool) if region is None
                else boundary_facet_mask(region, self.space.nodes))
        mass = self.space.assemble(BoundaryMassForm(self.n_components, mask), boundary=True)
        integrals = np.asarray(mass @ self.dofs).reshape(-1, self.n_components).sum(axis=0)
        return float(integrals[0]) if self.n_components == 1 else integrals

    def gradient(self) -> ElementValues:
        '''One gradient per element, `(n_elements, spatial_dim)` for a scalar field and
        `(n_elements, n_components, spatial_dim)` for a vector one: the volume-weighted
        mean over the element's rule, exact for P1 and the centroid value of a straight
        P2 element's linear gradient.'''
        geometry = self.space.geometry
        weights = geometry.weight_detJ / geometry.weight_detJ.sum(axis=1, keepdims=True)
        return np.einsum('eq,eq...->e...', weights, geometry.gradients(self.element_values))

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, points: Vertices) -> FloatArray:
        '''The field at `points`, `(n_points, spatial_dim)`: `(n_points,)` for a scalar
        field, `(n_points, n_components)` for a vector one.

        Each point is located in its element (`Mesh.locate`) and the shape functions
        are evaluated at its reference coordinates. A point outside the mesh is an
        error. Straight-sided elements only: a curved (isoparametric) element's
        reference coordinates need the inverse of its quadratic map.
        '''
        if self.element_type.GEOMETRY_DEGREE > 1:
            raise NotImplementedError(
                f'{self.element_type.__name__} is curved; evaluating a field on it needs '
                'the inverse of the isoparametric map'
            )
        points = np.atleast_2d(np.asarray(points, dtype=float))
        elements, reference = self.mesh.locate(points)
        phi = self.element_type.shape_values(reference)               # (n_points, N)
        values = np.einsum('pn,pn...->p...', phi, self.element_values[elements])
        return values

    def deformed_mesh(self, scale: float = 1.0) -> 'Mesh':
        '''The mesh displaced by `scale` times this field, for a displacement field:
        one component per spatial dimension.

        Only the leading vertex DOFs move the geometry: a P2 field's edge-midpoint
        DOFs have no mesh vertices, so the warp is the field's P1 restriction.
        '''
        if self.n_components != self.mesh.spatial_dim:
            raise ValueError(
                f'a displacement has one component per spatial dimension '
                f'({self.mesh.spatial_dim}); this field has {self.n_components}'
            )
        return self.mesh.displaced(self.dofs.reshape(-1, self.n_components), scale)
