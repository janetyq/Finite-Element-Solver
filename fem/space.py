"""The discrete function space: a mesh plus a choice of element and component count.

A problem in FEM reads "find u in V_h such that a(u, v) = L(v) for all v in V_h".
`Mesh` is the domain, `Equation` the physics, and `FunctionSpace` is V_h. A space
has a mesh rather than being one, so two spaces (P1 and P2, scalar and vector) can
share one copy of the domain.

P1 has one DOF per vertex, linear over each element. P2 adds an edge-midpoint node
per edge for quadratic interpolation. `n_components` is an explicit argument, so a
mixed formulation can build spaces the equation taxonomy has no name for; the solver
derives it from the equation one layer up.

The cached operators are valid only while the mesh is not mutated underneath them.
Build a new space instead of editing one.

`assemble_load` imports `fem.loads` lazily: a `Source` is written against the space, so the
edge points up and stays function-local (`loads` imports the space the same way).
"""
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from fem.elements import (
    Element,
    ElementGeometry,
    LinearElement,
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
)
from fem.field import NodalField
from fem.physics.forms import Form, MassForm
from fem.mesh.mesh import Mesh
from fem.regions import evaluate_field
from fem.typing import (
    DofIndices,
    DofVector,
    Elements,
    FieldValue,
    FloatArray,
    IntArray,
    SparseMatrix,
    Vertices,
)

if TYPE_CHECKING:
    from fem.loads import Source

from scipy.sparse import csr_array


def dof_indices(element: IntArray | Sequence[int], n_components: int) -> DofIndices:
    '''Global DOF indices for an element's nodes, interleaved per node.

    For node indices [n0, n1, ...] and `n_components` DOFs per node, returns
    [n_components*n0, n_components*n0+1, ..., n_components*n1, n_components*n1+1, ...].

    Batched over the leading axis, so an `(n_elements, N)` connectivity array
    gives `(n_elements, N*n_components)`, one row of DOF slots per element.
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

    `Mesh` rejects anything but linear simplices, so a 3-node element is a triangle
    and a 4-node element is a tet. The node count alone fixes the type.
    '''
    n_nodes = mesh.elements.shape[1]
    if n_nodes not in _SIMPLEX_ELEMENTS:
        raise NotImplementedError(
            f'no linear element for {n_nodes}-node elements'
        )
    return _SIMPLEX_ELEMENTS[n_nodes]


@dataclass(frozen=True, eq=False)
class _ScatterPlan:
    '''Where a batch of element matrices lands in the global matrix.

    The (row, col) slot each entry sums into is fixed by the connectivity, so it is
    resolved once. Each assembly is then a weighted `bincount` into a CSR matrix
    whose index arrays are already built.
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


@dataclass(frozen=True, eq=False)
class _VectorScatterPlan:
    '''Where a batch of element vectors lands in the global vector.

    The vector counterpart of `_ScatterPlan`: one weighted `bincount` over a flat
    destination map, resolved once. Several times faster than a per-call `np.add.at`
    for a Newton loop reassembling the residual each iteration.
    '''
    n_entries: int          # element-vector entries this plan expects
    destination: IntArray   # global DOF each entry sums into, one per entry
    n_dofs: int

    @classmethod
    def build(cls, dofs: DofIndices, n_dofs: int) -> '_VectorScatterPlan':
        '''Resolve an `(n_elements, k)` DOF map into a flat vector scatter.'''
        destination = np.asarray(dofs).ravel()
        return cls(n_entries=len(destination), destination=destination, n_dofs=n_dofs)

    def scatter(self, element_vectors: FloatArray) -> DofVector:
        '''Sum `element_vectors` into the global vector, matching entries to
        destinations in row-major order, the pairing `np.add.at` made.'''
        if element_vectors.size != self.n_entries:
            raise ValueError(
                f'expected element vectors covering {self.n_entries} entries, got '
                f'{element_vectors.size} (shape {element_vectors.shape})'
            )
        # bincount with float weights is float64, but its annotation doesn't say
        # so; the asarray tells the type checker and copies nothing.
        summed = np.bincount(
            self.destination,
            weights=np.asarray(element_vectors).ravel(),
            minlength=self.n_dofs,
        )
        return np.asarray(summed, dtype=np.float64)


@dataclass(frozen=True)
class NodeSet:
    '''The node geometry boundary-condition resolution resolves against.

    For P1 the mesh itself serves. For P2 the space builds one of these whose
    `vertices` include the edge-midpoint nodes and whose `boundary` facets carry them,
    so a condition written against coordinates pins the edge DOFs too. Duck-types
    `Mesh` for the attributes `Conditions.resolve` reads.
    '''
    vertices: Vertices          # (n_nodes, spatial) all node coordinates
    boundary: Elements          # (n_boundary_facets, facet_N) boundary facets as node indices
    boundary_idxs: IntArray     # unique boundary node indices
    spatial_dim: int
    boundary_tags: IntArray | None = None   # the mesh's; facet order shared with `boundary`


def p2_connectivity(
    mesh: Mesh, project_boundary: bool = False,
) -> tuple[Elements, Vertices, Elements]:
    '''Build the P2 node set for `mesh`: (element_nodes, node_coords, boundary_nodes).

    Nodes are the mesh vertices (indices unchanged) followed by one node per edge,
    placed at the edge midpoint: node `n_vertices + i` for `mesh.edges[i]`. An
    element's six nodes are its three corners then its three edge nodes, ordered so
    the edge opposite corner k comes k-th, matching `QuadraticTriangleElement`'s hats.
    A boundary facet gains its own edge's node as a third entry.

    With `project_boundary` and a mesh carrying `boundary_curves`, a boundary edge's
    midside node is projected onto its curve instead of the chord midpoint, so an
    isoparametric element's boundary edge bends to follow the true curve. Interior edge
    nodes stay at chord midpoints (curved boundary, straight interior). A straight P2
    element passes `project_boundary=False`, so its node placement is unchanged.
    '''
    n_vertices = len(mesh.vertices)
    edges = mesh.edges
    # `mesh.edges` is sorted lexicographically, so an edge's index is where its
    # single-integer key lands among the sorted keys.
    edge_keys = edges[:, 0] * n_vertices + edges[:, 1]

    def edge_index(pairs: IntArray) -> IntArray:
        '''Index into `mesh.edges` of each (a, b) row of `pairs`, in either orientation.'''
        pairs = np.sort(pairs, axis=1)
        return np.searchsorted(edge_keys, pairs[:, 0] * n_vertices + pairs[:, 1])

    elements = np.asarray(mesh.elements)
    element_nodes = np.empty((len(elements), 6), dtype=int)
    element_nodes[:, :3] = elements
    # The edge opposite corner k comes k-th: (1, 2), (0, 2), (0, 1).
    for k, (i, j) in enumerate([(1, 2), (0, 2), (0, 1)]):
        element_nodes[:, 3 + k] = n_vertices + edge_index(elements[:, [i, j]])

    boundary = np.asarray(mesh.boundary)
    boundary_edges = edge_index(boundary)
    edge_midpoints = mesh.vertices[edges].mean(axis=1)
    if project_boundary and mesh.boundary_curves is not None:
        # Each curve projects the midpoints of all its facets in one call.
        curves = list(mesh.boundary_curves)
        for curve in {id(c): c for c in curves if c is not None}.values():
            on_curve = boundary_edges[[c is curve for c in curves]]
            edge_midpoints[on_curve] = curve.project(edge_midpoints[on_curve])
    node_coords = np.vstack([mesh.vertices, edge_midpoints])

    boundary_nodes = np.empty((len(boundary), 3), dtype=int)
    boundary_nodes[:, :2] = boundary
    boundary_nodes[:, 2] = n_vertices + boundary_edges

    return element_nodes, node_coords, boundary_nodes


class FunctionSpace:
    '''A finite element space over `mesh`: an element, its node numbering, and
    `n_components` DOFs per node. P1 numbers DOFs on the mesh vertices; P2 adds a
    node per edge, so the node set is vertices then edge midpoints.'''

    def __init__(
        self,
        mesh: Mesh,
        element_type: type[Element] | None = None,
        n_components: int = 1,
    ) -> None:
        element_type = element_type if element_type is not None else element_type_for(mesh)
        if element_type.SUB_TYPE is None:
            # Only reachable for line elements, whose facets would be points,
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
        # Volume geometry keyed by the rule's exactness degree: the default 1-point
        # rule for constant-coefficient P1 forms, higher rules for variable
        # coefficients and higher-order elements, each built once and shared.
        self._geometry_cache: dict[int, ElementGeometry] = {}

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

    @cached_property
    def _connectivity(self) -> tuple[Elements, Vertices, Elements]:
        '''(element_nodes, node_coords, boundary_nodes) for this space's element.

        For P1 the mesh's own arrays (nodes are vertices); for P2 the enlarged set
        with an edge-midpoint node per edge. Everything below reads DOF numbering
        through here rather than off the mesh, so the mesh stays pure geometry.
        '''
        if self.element_type.SHAPE_DEGREE == 1:
            return self.mesh.elements, self.mesh.vertices, self.mesh.boundary
        # A curved element places its boundary nodes on the mesh's curves; a straight
        # P2 element keeps them at chord midpoints, so the two switches (node positions
        # and the Jacobian) always move together.
        return p2_connectivity(
            self.mesh, project_boundary=self.element_type.GEOMETRY_DEGREE > 1)

    @property
    def element_nodes(self) -> Elements:
        '''(n_elements, N) global node index of each element's local nodes.'''
        return self._connectivity[0]

    @property
    def node_coords(self) -> Vertices:
        '''(n_nodes, spatial) coordinates of every node the DOFs live on.'''
        return self._connectivity[1]

    @property
    def boundary_nodes(self) -> Elements:
        '''(n_boundary_facets, boundary_N) global node index of each facet's nodes.'''
        return self._connectivity[2]

    @property
    def n_nodes(self) -> int:
        return len(self.node_coords)

    @cached_property
    def nodes(self) -> Mesh | NodeSet:
        '''The node geometry BC resolution resolves against (see `NodeSet`).'''
        if self.element_type.SHAPE_DEGREE == 1:
            return self.mesh
        return NodeSet(
            vertices=self.node_coords,
            boundary=self.boundary_nodes,
            boundary_idxs=np.unique(self.boundary_nodes),
            spatial_dim=self.spatial_dim,
            boundary_tags=self.mesh.boundary_tags,
        )

    @property
    def n_dofs(self) -> int:
        return self.n_nodes * self.n_components

    def dof_indices(self, element: IntArray | Sequence[int]) -> DofIndices:
        '''Global DOF indices for one element's nodes.'''
        return dof_indices(element, self.n_components)

    # -- element geometry ---------------------------------------------------

    def geometry_at(self, min_degree: int) -> ElementGeometry:
        '''Volume-element geometry integrated at a rule of at least `min_degree`.

        Cached per rule: `geometry` is this at degree 1 (a single point, exact for
        the constant integrand of a P1 stiffness), and a variable-coefficient or
        higher-order form asks for the degree its integrand needs. Built once per
        distinct rule and shared by every form that wants it.
        '''
        rule = self.element_type.quadrature(min_degree)
        cached = self._geometry_cache.get(rule.degree)
        if cached is None:
            cached = self.element_type.geometry(self.node_coords[self.element_nodes], rule)
            self._geometry_cache[rule.degree] = cached
        return cached

    @property
    def geometry(self) -> ElementGeometry:
        '''Batched geometry at the element's default rule: a single point for P1,
        three for the degree-2 integrand of a P2 stiffness.'''
        return self.geometry_at(self.element_type.default_quadrature_degree())

    @cached_property
    def geometry_at_nodes(self) -> ElementGeometry:
        '''Geometry whose "quadrature points" are each element's own nodes.

        For reading a field's gradient (a flux, a stress) at the nodes rather than at
        interior integration points, as a nodal recovery of a P2 field needs. Not for
        integrating: see `Element.nodal_rule`.
        '''
        return self.element_type.geometry(self.node_coords[self.element_nodes],
                                          self.element_type.nodal_rule())

    @cached_property
    def boundary_geometry(self) -> ElementGeometry:
        '''The same, for the boundary facets: embedded elements, so a wider grad_phi.'''
        return self.boundary_type.geometry(self.node_coords[self.boundary_nodes])

    @property
    def element_volumes(self) -> FloatArray:
        '''(n_elements,) element measure: length, area, or volume.'''
        return self.geometry.volumes

    def element_gradient(self, e_idx: int, u_element: FloatArray) -> FloatArray:
        '''Gradient of a field over one element, from its nodal values.

        The element mean over the rule, as `gradient`; constant over the element for P1.
        '''
        w = self.geometry.weight_detJ[e_idx]
        grad_phi = np.einsum('q,qni->ni', w / w.sum(), self.geometry.grad_phi[e_idx])
        return grad_phi.T @ u_element

    # -- integrals ----------------------------------------------------------

    @property
    def total_volume(self) -> float:
        return float(self.element_volumes.sum())

    def interpolate(self, value: FieldValue) -> NodalField:
        '''The nodal interpolant of a field: `value` (a constant, a per-component
        constant, or a callable of position) evaluated at every node of the space.

        The way to build an initial condition or a comparison field. A load that must
        resolve variation within an element is a `Source` over a callable.
        '''
        return NodalField(self, evaluate_field(value, self.node_coords, self.n_components).flatten())

    def gradient(self, u: DofVector | NodalField) -> FloatArray:
        '''(n_elements, spatial_dim) gradient of a scalar DOF vector, one value per
        element; `NodalField.gradient` for a field, which also takes a vector one.'''
        return NodalField(self, np.asarray(u)).gradient()

    def element_hessian(self, u_elements: FloatArray) -> FloatArray:
        '''(n_elements, spatial, spatial[, n_components]) physical Hessian of a field.

        The second derivatives `d2u / dx_i dx_j`, constant per element for a straight
        (affine) element: the reference Hessian of the shape functions mapped through the
        constant inverse Jacobian, `H_phys = J^-T H_ref J^-1`. Zero for P1, the curvature
        a P2 field carries, and what a strong-form residual (a Laplacian or a stress
        divergence) needs. `u_elements` is `(n_elements, N)` for a scalar field or
        `(n_elements, N, n_components)` for a vector one.

        Straight-sided only: a curved (isoparametric) element has a varying Jacobian, so
        its physical Hessian picks up a first-derivative term this omits.
        '''
        d = self.spatial_dim
        corners = self.node_coords[self.element_nodes[:, :d + 1]]   # (n_el, d+1, d)
        # J[e, i, r] = d x_i / d xi_r: the columns are the edge vectors from corner 0.
        jacobian = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        jac_inv = np.linalg.inv(jacobian)                          # jac_inv[e, r, i] = d xi_r / d x_i
        h_ref = self.element_type.shape_hessians(np.zeros((1, d)))[0]   # (N, r, r), constant
        # H_phys[e, a, i, j] = J^-1[e, r, i] H_ref[a, r, s] J^-1[e, s, j]
        h_phys = np.einsum('eri,ars,esj->eaij', jac_inv, h_ref, jac_inv)
        return np.einsum('eaij,ea...->eij...', h_phys, np.asarray(u_elements, dtype=float))

    # -- the scalar mass matrix nodal recovery projects against ------------

    @cached_property
    def nodal_mass_matrix(self) -> SparseMatrix:
        '''The scalar (n_nodes x n_nodes) consistent mass matrix, for L2 nodal recovery
        (`fem.post.recovery`).

        A vector space's `mass_matrix` is block-interleaved over its components, but
        recovery projects each scalar component on its own, so it needs the one-component
        matrix on the same nodes. Identical to `mass_matrix` when the space is scalar.
        '''
        if self.n_components == 1:
            return self.mass_matrix
        return FunctionSpace(self.mesh, self.element_type, n_components=1).mass_matrix

    # -- operators ----------------------------------------------------------

    @cached_property
    def mass_matrix(self) -> SparseMatrix:
        '''The consistent mass matrix. Depends only on geometry, so it caches.'''
        return self.assemble(MassForm(self.n_components))

    @cached_property
    def boundary_mass_matrix(self) -> SparseMatrix:
        '''Mass matrix over boundary facets, for integrating tractions.'''
        return self.assemble(MassForm(self.n_components), boundary=True)

    def assemble(self, form: Form, boundary: bool | None = None) -> SparseMatrix:
        '''Scatter a constant-tangent form's element matrices into a global matrix.

        Each of the form's `terms` integrates over its own `domain` and the results are
        added; `boundary` overrides the domain for every term (a volume `MassForm`
        integrated over the facets is the boundary mass). Not cached, since a form may
        carry material data that changes between calls.
        '''
        total = None
        for term in form.terms:
            on_boundary = term.domain == 'boundary' if boundary is None else boundary
            if not term.constant_tangent:
                raise TypeError(
                    f'{type(term).__name__} has a state-dependent tangent; use assemble_tangent'
                )
            blocks = term.element_matrices(self._term_geometry(term, on_boundary))
            matrix = self._term_matrix_scatter(on_boundary).scatter(blocks)
            total = matrix if total is None else total + matrix
        assert total is not None
        return total

    def assemble_load(self, source: 'Source') -> DofVector:
        '''Scatter a sampled `Source`'s element vectors into the global load vector.

        The vector counterpart of `assemble`: element load vectors summed into the
        DOFs their nodes own, the same scatter `assemble_residual` runs for the
        nonlinear residual. The source is sampled at the quadrature points of a rule
        of its `quadrature_degree`.
        '''
        geometry = self.geometry_at(source.quadrature_degree)
        vectors = source.element_vectors(geometry, self.n_components)
        return self._volume_vector_scatter.scatter(vectors)

    def assemble_loads(self, form: Form) -> DofVector | None:
        '''The global vector of the loads a form's terms carry through
        `Form.element_loads`, or None when none does. The vector counterpart of
        `assemble_residual`.'''
        total = None
        for term in form.terms:
            on_boundary = term.domain == 'boundary'
            vectors = term.element_loads(self._term_geometry(term, on_boundary))
            if vectors is None:
                continue
            load = self._term_vector_scatter(on_boundary).scatter(vectors)
            total = load if total is None else total + load
        return total

    # -- state-dependent assembly -------------------------------------------
    #
    # A form's element quantities may depend on the current state, so these take
    # `u` and evaluate each term at its elements' slice of it. Constraints stay with
    # the caller (the Problem and its solve strategy).

    def _term_geometry(self, term: Form, on_boundary: bool) -> ElementGeometry:
        '''Geometry at a rule that integrates `term` over its domain.

        On the volume, the larger of the element's default and what the term asks
        for: a quartic St-Venant-Kirchhoff energy wants degree 4 on P2. One rule serves
        the energy, residual, and tangent, so the residual is the exact gradient of the
        quadrature energy and Newton sees a matching tangent.
        '''
        if on_boundary:
            return self.boundary_geometry
        degree = max(self.element_type.default_quadrature_degree(),
                     term.quadrature_degree(self.element_type.SHAPE_DEGREE))
        return self.geometry_at(degree)

    def _term_state(self, u: DofVector, on_boundary: bool) -> FloatArray:
        '''(n_elements, N, n_components): each element's (or facet's) slice of the state.'''
        nodes = self.boundary_nodes if on_boundary else self.element_nodes
        return np.asarray(u, dtype=float).reshape(-1, self.n_components)[nodes]

    def _term_matrix_scatter(self, on_boundary: bool) -> _ScatterPlan:
        return self._boundary_scatter if on_boundary else self._volume_scatter

    def _term_vector_scatter(self, on_boundary: bool) -> _VectorScatterPlan:
        return self._boundary_vector_scatter if on_boundary else self._volume_vector_scatter

    def total_energy(self, form: Form, u: DofVector) -> float:
        '''Sum a form's element energies at state `u` over its terms: the scalar Pi(u).'''
        total = 0.0
        for term in form.terms:
            on_boundary = term.domain == 'boundary'
            energies = term.element_energies(
                self._term_geometry(term, on_boundary), self._term_state(u, on_boundary))
            total += float(energies.sum())
        return total

    def assemble_residual(self, form: Form, u: DofVector) -> DofVector:
        '''Scatter element residuals at `u` into the global residual, shape (n_dofs,).'''
        total = np.zeros(self.n_dofs)
        for term in form.terms:
            on_boundary = term.domain == 'boundary'
            residuals = term.element_residuals(
                self._term_geometry(term, on_boundary), self._term_state(u, on_boundary))
            total = total + self._term_vector_scatter(on_boundary).scatter(residuals)
        return total

    def assemble_tangent(self, form: Form, u: DofVector) -> SparseMatrix:
        '''Scatter element tangents at `u` into the global tangent, shape (n_dofs, n_dofs).'''
        total = None
        for term in form.terms:
            on_boundary = term.domain == 'boundary'
            tangents = term.element_tangents(
                self._term_geometry(term, on_boundary), self._term_state(u, on_boundary))
            matrix = self._term_matrix_scatter(on_boundary).scatter(tangents)
            total = matrix if total is None else total + matrix
        assert total is not None
        return total

    # -- the scatter -------------------------------------------------------

    def _scatter_plan(self, elements: Elements) -> _ScatterPlan:
        '''Where every entry of every element block lands in the global matrix.

        Depends only on connectivity and `n_components`, never on the form or the
        geometry, so it is computed once per element set and reused by every
        operator assembled over it: mass, stiffness, and each topology optimization
        iteration's rebuilt stiffness alike.
        '''
        # (n_elements, k): each element's global DOF positions, interleaved per node.
        dofs = self.dof_indices(elements)
        k = dofs.shape[1]
        # Row index varies down the block, column index across it: the vectorized
        # form of the (k, k) index grid, one block per element.
        rows = np.repeat(dofs, k, axis=1).ravel()
        cols = np.tile(dofs, (1, k)).ravel()
        return _ScatterPlan.build(rows, cols, self.n_dofs)

    @cached_property
    def _volume_scatter(self) -> _ScatterPlan:
        return self._scatter_plan(self.element_nodes)

    @cached_property
    def _volume_vector_scatter(self) -> _VectorScatterPlan:
        '''The shared load/residual scatter: `assemble_load` and `assemble_residual`
        both sum an element vector into the same volume-element DOFs, so one plan
        serves both.'''
        return _VectorScatterPlan.build(
            self.dof_indices(self.element_nodes), self.n_dofs
        )

    @cached_property
    def _boundary_scatter(self) -> _ScatterPlan:
        return self._scatter_plan(self.boundary_nodes)

    @cached_property
    def _boundary_vector_scatter(self) -> _VectorScatterPlan:
        return _VectorScatterPlan.build(self.dof_indices(self.boundary_nodes), self.n_dofs)
