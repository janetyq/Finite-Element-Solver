"""The discrete function space: a mesh plus a choice of element and component count.

In FEM notation a problem reads "find u in V_h such that a(u, v) = L(v) for all
v in V_h". The package has an object for the domain (`Mesh`) and objects for the
physics (`Equation`, the assembly routines), but none for V_h -- so `FEMesh`
answers V_h's questions by accident, being the only object that knows both the
element type and the DOF numbering.

`FunctionSpace` is that missing object. It **has** a mesh rather than being one:
a discretization is not a kind of geometry, it is a pairing of geometry with an
element choice and a component count. Two spaces can share one domain -- P1 and
P2, scalar and vector -- which is exactly what `FEMesh` cannot represent, since
`prepare_matrices` rebuilds its operators in place and the last caller wins.

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

from fem.elements import LinearElement, LinearTriangleElement
from fem.mesh.mesh import Mesh
from fem.typing import DofIndices, Elements, IntArray, Matrix


def dof_indices(element: IntArray | Sequence[int], n_components: int) -> DofIndices:
    '''Global DOF indices for an element's nodes, interleaved per node.

    For node indices [n0, n1, ...] and `n_components` DOFs per node, returns
    [n_components*n0, n_components*n0+1, ..., n_components*n1, n_components*n1+1, ...].
    '''
    element = np.asarray(element)
    return np.array([n_components*element + i for i in range(n_components)]).T.flatten()


class FunctionSpace:
    '''P1 finite element space over `mesh`, with `n_components` DOFs per node.'''

    def __init__(
        self,
        mesh: Mesh,
        element_type: type[LinearElement] = LinearTriangleElement,
        n_components: int = 1,
    ) -> None:
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
