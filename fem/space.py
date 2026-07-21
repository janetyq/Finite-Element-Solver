"""The discrete function space: a mesh plus a choice of element and component count.

In FEM notation a problem reads "find u in V_h such that a(u, v) = L(v) for all
v in V_h". The package has an object for the domain (`Mesh`) and objects for the
physics (`Equation`, the assembly routines), but no object for V_h.

`FunctionSpace` is that object. It **has** a mesh rather than being one: a
discretization is not a kind of geometry, it is a pairing of geometry with an
element choice and a component count. Two spaces can therefore share one domain
-- P1 and P2, scalar and vector -- over a single copy of the geometry.

`n_components` is taken as an explicit low-level argument rather than an
`Equation`, so a mixed formulation can build spaces the equation taxonomy has no
name for. Deriving it from `Equation.field` happens one layer up, in the solver.

Immutability is assumed, not enforced: the cached operators are only valid while
the mesh is not mutated underneath them. Build a new space instead of editing one
-- the same contract `ResolvedBC` has with `BoundaryConditions`.
"""
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Callable

import numpy as np

from fem.elements import (
    LinearElement,
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.mesh.mesh import Mesh
from fem.typing import DofIndices, Elements, FloatArray, IntArray, Matrix, VertexField


def dof_indices(element: IntArray | Sequence[int], n_components: int) -> DofIndices:
    '''Global DOF indices for an element's nodes, interleaved per node.

    For node indices [n0, n1, ...] and `n_components` DOFs per node, returns
    [n_components*n0, n_components*n0+1, ..., n_components*n1, n_components*n1+1, ...].
    '''
    element = np.asarray(element)
    return np.array([n_components*element + i for i in range(n_components)]).T.flatten()


_SIMPLEX_ELEMENTS: dict[int, type[LinearElement]] = {
    2: LinearLineElement,
    3: LinearTriangleElement,
    4: LinearTetrahedralElement,
}


def element_type_for(mesh: Mesh) -> type[LinearElement]:
    '''The linear element matching the mesh's node count.

    Unambiguous rather than a guess: `Mesh` rejects anything but linear simplices,
    so a 3-node element *is* a triangle and a 4-node element *is* a tet. Callers
    therefore no longer have to restate what the connectivity already says.
    '''
    n_nodes = mesh.elements.shape[1]
    if n_nodes not in _SIMPLEX_ELEMENTS:
        raise NotImplementedError(
            f'no linear element for {n_nodes}-node elements'
        )
    return _SIMPLEX_ELEMENTS[n_nodes]


class FunctionSpace:
    '''P1 finite element space over `mesh`, with `n_components` DOFs per node.'''

    def __init__(
        self,
        mesh: Mesh,
        element_type: type[LinearElement] | None = None,
        n_components: int = 1,
    ) -> None:
        element_type = element_type if element_type is not None else element_type_for(mesh)
        if element_type.SUB_TYPE is None:
            # Only reachable for line elements, whose facets would be points --
            # the 1D path the SUB_TYPE TODO tracks. Raising here beats a
            # "NoneType is not callable" from the boundary comprehension.
            raise NotImplementedError(
                f'{element_type.__name__} has no boundary element type, so boundary '
                f'integrals (and hence a FunctionSpace) are not defined for it yet'
            )
        if n_components < 1:
            raise ValueError(f'n_components must be at least 1, got {n_components}')

        self.mesh = mesh
        self.element_type = element_type
        self.boundary_type = element_type.SUB_TYPE
        self.n_components = n_components

    def __repr__(self) -> str:
        return (
            f'FunctionSpace({self.element_type.__name__}, '
            f'n_components={self.n_components}, n_dofs={self.n_dofs})'
        )

    # -- sizing and numbering -----------------------------------------------

    @property
    def spatial_dim(self) -> int:
        '''Dimension of the space the nodes live in. Distinct from n_components.'''
        return self.mesh.spatial_dim

    @property
    def n_dofs(self) -> int:
        return len(self.mesh.vertices) * self.n_components

    def dof_indices(self, element: IntArray | Sequence[int]) -> DofIndices:
        '''Global DOF indices for one element's nodes.'''
        return dof_indices(element, self.n_components)

    # -- element geometry ---------------------------------------------------

    @cached_property
    def element_objs(self) -> list[LinearElement]:
        return [self.element_type(self.mesh.vertices[e]) for e in self.mesh.elements]

    @cached_property
    def boundary_objs(self) -> list[LinearElement]:
        return [self.boundary_type(self.mesh.vertices[f]) for f in self.mesh.boundary]

    # Per-element geometry is reached through these rather than through
    # `element_objs`, so that replacing the per-element objects with batched
    # `(n_elements, ...)` arrays stays a change inside this class.

    @cached_property
    def element_volumes(self) -> FloatArray:
        '''(n_elements,) element measure -- length, area, or volume.'''
        return np.array([e.volume for e in self.element_objs])

    def element_gradient(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        '''Gradient of a field over one element, from its nodal values.'''
        return self.element_objs[e_idx].calculate_gradient(u_element)

    def element_dF_dx(self, e_idx: int) -> FloatArray:
        '''d(deformation gradient)/d(nodal position) for one element.'''
        return self.element_objs[e_idx].dF_dx

    # -- integrals ----------------------------------------------------------

    @property
    def total_volume(self) -> float:
        return float(self.element_volumes.sum())

    def integrate(self, u: VertexField) -> float:
        '''Integral of a nodal field over the domain.

        `M @ u` sums to exactly the integral of a P1 field, so no separate
        quadrature is needed. Nodal fields only -- the old mesh-level version
        guessed between nodal and per-element data by comparing lengths, which
        picks wrong whenever n_elements == n_vertices.
        '''
        return float((self.mass_matrix @ u).sum())

    def mean_value(self, u: VertexField) -> float:
        '''Volume-weighted mean of a nodal field.'''
        return self.integrate(u) / self.total_volume

    def gradient(self, u: VertexField) -> FloatArray:
        '''(n_elements, spatial_dim) gradient of a nodal field, one value per element.

        Constant per element for P1, which is why it is an element field.
        '''
        return np.array([
            self.element_gradient(e_idx, u[element])
            for e_idx, element in enumerate(self.mesh.elements)
        ])

    # -- operators ----------------------------------------------------------

    @cached_property
    def mass_matrix(self) -> Matrix:
        '''The consistent mass matrix. Depends only on geometry, so it caches.'''
        return self._assemble(
            self.mesh.elements,
            lambda e_idx: self.element_objs[e_idx].calculate_mass_matrix(self.n_components),
        )

    @cached_property
    def boundary_mass_matrix(self) -> Matrix:
        '''Mass matrix over boundary facets, for integrating tractions.'''
        return self._assemble(
            self.mesh.boundary,
            lambda f_idx: self.boundary_objs[f_idx].calculate_mass_matrix(self.n_components),
        )

    def assemble_stiffness(self, **material: Any) -> Matrix:
        '''The stiffness matrix, which is *not* cached -- it depends on material data.

        That asymmetry with the mass matrix is the whole reason assembly cannot
        simply live on the space: the scalar case is a material-free Laplacian,
        but the elastic case needs per-element `mu` and `lamb`. Passing them as
        keywords is the interim; a `Form` owning its own material is where this
        goes, and is what lets the result be cached against something meaningful.
        '''
        return self._assemble(
            self.mesh.elements,
            lambda e_idx: self.element_objs[e_idx].calculate_stiffness_matrix(
                self.n_components, idx=e_idx, **material
            ),
        )

    def _assemble(
        self,
        elements: Elements,
        element_matrix: Callable[[int], Matrix],
    ) -> Matrix:
        '''Scatter per-element matrices into a global one over `elements`.'''
        A = np.zeros((self.n_dofs, self.n_dofs))
        for e_idx, element in enumerate(elements):
            idxs = self.dof_indices(element)
            A[np.ix_(idxs, idxs)] += element_matrix(e_idx)
        return A
