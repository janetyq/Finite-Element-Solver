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
from dataclasses import dataclass
from functools import cached_property

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
    ElementField,
    Elements,
    FloatArray,
    IntArray,
    SparseMatrix,
    VertexField,
)

from scipy.sparse import csr_array


def dof_indices(element: IntArray | Sequence[int], n_components: int) -> DofIndices:
    '''Global DOF indices for an element's nodes, interleaved per node.

    For node indices [n0, n1, ...] and `n_components` DOFs per node, returns
    [n_components*n0, n_components*n0+1, ..., n_components*n1, n_components*n1+1, ...].

    Batched over the leading axis, so an `(n_elements, N)` connectivity array
    gives `(n_elements, N*n_components)` -- one row of DOF slots per element.
    '''
    element = np.asarray(element)
    interleaved = n_components * element[..., None] + np.arange(n_components)
    return interleaved.reshape(*element.shape[:-1], -1)


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


@dataclass(frozen=True, eq=False)
class _ScatterPlan:
    '''Where the entries of a batch of element matrices land in the global matrix.

    Assembly sums every element-matrix entry into the global slot its (row, col)
    names, so entries from elements sharing a node land together. Which slots exist
    and which entries share one is fixed by the connectivity, not by the form or
    the geometry -- so it is resolved once, here, and each assembly is then a
    weighted `bincount` into a CSR matrix whose index arrays are already built.

    That matters for a driver that assembles the same elements repeatedly: a
    topology optimization iteration rescales the element matrices and reassembles,
    and would otherwise re-sort the same COO triplets into CSR every time.
    '''
    n_entries: int      # element-matrix entries this plan expects
    order: IntArray     # sorts those entries by destination slot
    group: IntArray     # destination slot of each entry, once sorted
    indices: IntArray   # CSR column indices, one per slot
    indptr: IntArray
    shape: tuple[int, int]

    @classmethod
    def build(cls, rows: IntArray, cols: IntArray, n_dofs: int) -> '_ScatterPlan':
        '''Resolve flat (row, col) COO coordinates into a reusable scatter.'''
        # One integer per coordinate, ordered as CSR wants its entries: by row,
        # then by column within the row.
        destination = rows * n_dofs + cols
        order = np.argsort(destination, kind='stable')
        in_order = destination[order]

        # Equal neighbours in the sorted run are entries summing into one slot, so
        # a running count of the distinct ones numbers the slots.
        starts_slot = np.ones(len(in_order), dtype=bool)
        starts_slot[1:] = in_order[1:] != in_order[:-1]
        group = np.cumsum(starts_slot) - 1

        slots = in_order[starts_slot]
        per_row = np.bincount(slots // n_dofs, minlength=n_dofs)
        return cls(
            n_entries=len(destination),
            order=order,
            group=group,
            indices=slots % n_dofs,
            indptr=np.concatenate([[0], np.cumsum(per_row)]),
            shape=(n_dofs, n_dofs),
        )

    def scatter(self, element_matrices: FloatArray) -> SparseMatrix:
        '''Sum `element_matrices` into the global matrix this plan was built for.'''
        if element_matrices.size != self.n_entries:
            raise ValueError(
                f'expected element matrices covering {self.n_entries} entries, got '
                f'{element_matrices.size} (shape {element_matrices.shape})'
            )
        data = np.bincount(
            self.group,
            weights=element_matrices.ravel()[self.order],
            minlength=len(self.indices),
        )
        return csr_array((data, self.indices, self.indptr), shape=self.shape)


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
        '''Gradient of a field over one element, from its nodal values.

        At the first quadrature point -- constant over the element for P1.
        '''
        return self.geometry.grad_phi[e_idx, 0].T @ u_element

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

        Taken at the first quadrature point; constant per element for P1, which is
        why it is an element field.
        '''
        return self.geometry.gradients(u[self.mesh.elements])[:, 0]

    # -- projections between element and nodal fields -----------------------

    def element_to_vertex(self, values: ElementField) -> VertexField:
        '''Project a per-element field onto the nodes, weighted by element volume.

        A P1 solve produces element-constant derived quantities (stress, an error
        estimate, a density) while plotting and nodal output want a value per
        vertex, so something has to combine the values of the elements meeting at
        a node.

        Weighted by volume rather than counted evenly: on a graded mesh a sliver
        and the large element beside it are not equally good evidence about the
        field near their shared node, and an unweighted mean gives them the same
        say. On a uniform mesh the two agree exactly.

        This lives on the space rather than on `Mesh` because it is a
        discretization operation, not a geometric one -- it needs the element
        measures, which the space owns and the mesh does not.
        '''
        values = np.asarray(values, dtype=float)
        if len(values) != len(self.mesh.elements):
            raise ValueError(
                f'expected one value per element ({len(self.mesh.elements)}), '
                f'got {len(values)}'
            )
        nodes = self.mesh.elements
        weights = self.element_volumes
        flat = nodes.ravel()
        weighted = np.repeat(values * weights, nodes.shape[1])
        totals = np.repeat(weights, nodes.shape[1])

        sums = np.bincount(flat, weights=weighted, minlength=len(self.mesh.vertices))
        norms = np.bincount(flat, weights=totals, minlength=len(self.mesh.vertices))
        # Every meshed vertex belongs to at least one element; an unreferenced one
        # would divide by zero, so it keeps 0 instead.
        return sums / np.where(norms > 0, norms, 1.0)

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
        geometry = self.boundary_geometry if boundary else self.geometry
        return self._assemble(form.element_matrices(geometry), boundary=boundary)

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
        u_elements = u.reshape(-1, self.n_components)[self.mesh.elements]
        return float(form.element_energies(self.geometry, u_elements).sum())

    def assemble_residual(self, form: EnergyForm, u: DofVector) -> DofVector:
        '''Scatter element residuals at `u` into grad Pi(u), shape (n_dofs,).'''
        u_elements = u.reshape(-1, self.n_components)[self.mesh.elements]
        residuals = form.element_residuals(self.geometry, u_elements)
        dofs = dof_indices(self.mesh.elements, self.n_components)
        r = np.zeros(self.n_dofs)
        np.add.at(r, dofs, residuals.reshape(len(self.mesh.elements), -1))
        return r

    def assemble_tangent(self, form: EnergyForm, u: DofVector) -> SparseMatrix:
        '''Scatter element tangents at `u` into grad^2 Pi(u), shape (n_dofs, n_dofs).'''
        u_elements = u.reshape(-1, self.n_components)[self.mesh.elements]
        k = self.element_type.N * self.n_components
        tangents = form.element_tangents(self.geometry, u_elements).reshape(-1, k, k)
        return self._assemble(tangents)

    # -- the scatter -------------------------------------------------------

    def _scatter_plan(self, elements: Elements) -> _ScatterPlan:
        '''Where every entry of every element block lands in the global matrix.

        Depends only on connectivity and `n_components`, never on the form or the
        geometry, so it is computed once per element set and reused by every
        operator assembled over it -- mass, stiffness, and each topology
        optimization iteration's rebuilt stiffness alike.
        '''
        # (n_elements, k): each element's global DOF positions, interleaved per node.
        dofs = self.dof_indices(elements)
        k = dofs.shape[1]
        # Row index varies down the block, column index across it -- the vectorized
        # form of the (k, k) index grid, one block per element.
        rows = np.repeat(dofs, k, axis=1).ravel()
        cols = np.tile(dofs, (1, k)).ravel()
        return _ScatterPlan.build(rows, cols, self.n_dofs)

    @cached_property
    def _volume_scatter(self) -> _ScatterPlan:
        return self._scatter_plan(self.mesh.elements)

    @cached_property
    def _boundary_scatter(self) -> _ScatterPlan:
        return self._scatter_plan(self.mesh.boundary)

    def _assemble(
        self,
        element_matrices: FloatArray,
        boundary: bool = False,
    ) -> SparseMatrix:
        '''Scatter (n_elements, k, k) element matrices into the global operator.

        The scatter-add that `A[np.ix_(idxs, idxs)] += block` did densely, in
        O(nonzeros) memory: entries sharing a (row, col) are summed, and the
        destinations come from the cached `_ScatterPlan`.
        '''
        plan = self._boundary_scatter if boundary else self._volume_scatter
        return plan.scatter(element_matrices)
