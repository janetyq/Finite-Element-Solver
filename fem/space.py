"""The discrete function space: a mesh plus a choice of element and component count.

In FEM notation a problem reads "find u in V_h such that a(u, v) = L(v) for all
v in V_h". The package has an object for the domain (`Mesh`) and objects for the
physics (`Equation`, the assembly routines), but no object for V_h.

`FunctionSpace` is that object. It **has** a mesh rather than being one: a
discretization is not a kind of geometry, it is a pairing of geometry with an
element choice and a component count. Two spaces can therefore share one domain
-- P1 and P2, scalar and vector -- over a single copy of the geometry.

P1 is the piecewise-linear space: one DOF per vertex, linear over each element
and continuous across element boundaries. P2 adds edge-midpoint nodes for
quadratic interpolation. Only P1 is implemented here.

`n_components` is taken as an explicit low-level argument rather than an
`Equation`, so a mixed formulation can build spaces the equation taxonomy has no
name for. Deriving it from `Equation.field` happens one layer up, in the solver.

Immutability is assumed, not enforced: the cached operators are only valid while
the mesh is not mutated underneath them. Build a new space instead of editing one
-- the same contract `ResolvedBC` has with `BoundaryConditions`.
"""
from collections.abc import Sequence
from functools import cached_property
from typing import Callable

import numpy as np

from fem.elements import (
    ElementGeometry,
    LinearElement,
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.forms import EnergyForm, Form, MassForm
from fem.mesh.mesh import Mesh
from fem.typing import (
    DofIndices,
    DofVector,
    Elements,
    FloatArray,
    IntArray,
    Matrix,
    SparseMatrix,
    VertexField,
)

from scipy.sparse import csr_array


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
    def geometry(self) -> ElementGeometry:
        '''Batched geometry of the volume elements: one array of grad_phi, one of measures.'''
        return self.element_type.geometry(self.mesh.vertices[self.mesh.elements])

    @cached_property
    def boundary_geometry(self) -> ElementGeometry:
        '''The same, for the boundary facets -- embedded elements, so a wider grad_phi.'''
        return self.boundary_type.geometry(self.mesh.vertices[self.mesh.boundary])

    @property
    def element_volumes(self) -> FloatArray:
        '''(n_elements,) element measure -- length, area, or volume.'''
        return self.geometry.volumes

    def element_gradient(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        '''Gradient of a field over one element, from its nodal values.'''
        return self.geometry.at(e_idx).calculate_gradient(u_element)

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
        return self.geometry.gradients(u[self.mesh.elements])

    # -- operators ----------------------------------------------------------

    @cached_property
    def mass_matrix(self) -> SparseMatrix:
        '''The consistent mass matrix. Depends only on geometry, so it caches.'''
        return self.assemble(MassForm(self.n_components))

    @cached_property
    def boundary_mass_matrix(self) -> SparseMatrix:
        '''Mass matrix over boundary facets, for integrating tractions.'''
        return self.assemble(MassForm(self.n_components), boundary=True)

    def assemble(self, form: Form, boundary: bool = False) -> SparseMatrix:
        '''Scatter `form`'s element matrices into a global matrix.

        The space owns the loop; the form owns the integrand, so the space stays
        free of any physics. `boundary=True` integrates over the boundary facets
        instead of the volume elements -- the same scatter, a different mesh of
        elements. Not cached: a form may carry material data that changes (a
        topology-optimization iteration rescales the modulus). The geometry-only
        results the callers *want* cached, the mass matrices, cache themselves.
        '''
        elements = self.mesh.boundary if boundary else self.mesh.elements
        geometry = self.boundary_geometry if boundary else self.geometry
        return self._assemble(
            elements,
            lambda idx: form.element_matrix(geometry.at(idx), idx),
        )

    # -- nonlinear assembly -------------------------------------------------
    #
    # The bilinear `assemble` above scatters a state-independent matrix. An
    # EnergyForm's element quantities depend on the current displacement, so these
    # take `u` and evaluate the form at each element's slice of it. The tangent
    # reuses the same scatter loop as `assemble`; the residual is a vector scatter
    # and the energy a scalar reduction. Constraints stay with the caller
    # (EnergySolver's Newton loop), exactly as boundary conditions stay with the
    # caller for the bilinear path.

    def total_energy(self, form: EnergyForm, u: DofVector) -> float:
        '''Sum an EnergyForm's element energies at state `u`: the scalar Pi(u).'''
        u_nodal = u.reshape(-1, self.n_components)  # (n_vertices, n_components)
        return sum(
            # u_nodal[element] is the element's (N, n_components) local state.
            form.element_energy(self.geometry.at(e_idx), u_nodal[element])
            for e_idx, element in enumerate(self.mesh.elements)
        )

    def assemble_residual(self, form: EnergyForm, u: DofVector) -> DofVector:
        '''Scatter element residuals at `u` into grad Pi(u), shape (n_dofs,).'''
        u_nodal = u.reshape(-1, self.n_components)  # (n_vertices, n_components)
        r = np.zeros(self.n_dofs)
        for e_idx, element in enumerate(self.mesh.elements):
            # (N, n_components) -> flatten to (N*n_components,), added into the
            # element's global DOF slots.
            contribution = form.element_residual(self.geometry.at(e_idx), u_nodal[element])
            r[self.dof_indices(element)] += contribution.flatten()
        return r

    def assemble_tangent(self, form: EnergyForm, u: DofVector) -> SparseMatrix:
        '''Scatter element tangents at `u` into grad^2 Pi(u), shape (n_dofs, n_dofs).'''
        u_nodal = u.reshape(-1, self.n_components)  # (n_vertices, n_components)

        def element_matrix(e_idx: int) -> Matrix:
            # (N, n_components, N, n_components) -> (k, k) local stiffness, ordered
            # to match dof_indices for _assemble's scatter.
            tangent = form.element_tangent(
                self.geometry.at(e_idx), u_nodal[self.mesh.elements[e_idx]]
            )
            k = self.element_type.N * self.n_components
            return tangent.reshape(k, k)

        return self._assemble(self.mesh.elements, element_matrix)

    def _assemble(
        self,
        elements: Elements,
        element_matrix: Callable[[int], Matrix],
    ) -> SparseMatrix:
        '''Scatter per-element (k, k) matrices into the global (n_dofs, n_dofs) one.

        Emitted as COO triplets (row, col, value) and built into a CSR matrix,
        which sums entries at repeated (row, col) -- exactly the scatter-add that
        `A[np.ix_(idxs, idxs)] += block` did densely, now in O(nonzeros) memory.
        '''
        rows: list[IntArray] = []
        cols: list[IntArray] = []
        data: list[FloatArray] = []
        for e_idx, element in enumerate(elements):
            # idxs: the element's k = N*n_components global DOF positions; the
            # (k, k) index grid pairs each block entry with its global (row, col).
            idxs = self.dof_indices(element)
            grid_rows, grid_cols = np.meshgrid(idxs, idxs, indexing='ij')
            rows.append(grid_rows.ravel())
            cols.append(grid_cols.ravel())
            data.append(element_matrix(e_idx).ravel())
        return csr_array(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_dofs, self.n_dofs),
        )
