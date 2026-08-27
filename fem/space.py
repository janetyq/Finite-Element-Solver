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
"""
from collections.abc import Sequence
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
from fem.forms import BilinearForm, Form, LinearForm, MassForm
from fem.mesh.mesh import Mesh
from fem.regions import evaluate_field
from fem.typing import (
    DofIndices,
    DofVector,
    ElementField,
    Elements,
    FieldValue,
    FloatArray,
    IntArray,
    SparseMatrix,
    VertexField,
    Vertices,
)

from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve


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
    `Mesh` for the attributes `BoundaryConditions.resolve` reads.
    '''
    vertices: Vertices          # (n_nodes, spatial) all node coordinates
    boundary: Elements          # (n_boundary_facets, facet_N) boundary facets as node indices
    boundary_idxs: IntArray     # unique boundary node indices
    spatial_dim: int


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
    edge_index = {(int(a), int(b)): i for i, (a, b) in enumerate(mesh.edges)}

    def edge_node(a: int, b: int) -> int:
        return n_vertices + edge_index[(a, b) if a < b else (b, a)]

    elements = np.asarray(mesh.elements)
    element_nodes = np.empty((len(elements), 6), dtype=int)
    element_nodes[:, :3] = elements
    for e, (a, b, c) in enumerate(elements):
        element_nodes[e, 3] = edge_node(b, c)   # opposite corner 0
        element_nodes[e, 4] = edge_node(a, c)   # opposite corner 1
        element_nodes[e, 5] = edge_node(a, b)   # opposite corner 2

    edge_midpoints = mesh.vertices[mesh.edges].mean(axis=1)
    if project_boundary and mesh.boundary_curves is not None:
        for facet, curve in zip(mesh.boundary, mesh.boundary_curves):
            if curve is None:
                continue
            a, b = int(facet[0]), int(facet[1])
            e = edge_index[(a, b) if a < b else (b, a)]
            edge_midpoints[e] = curve.project(edge_midpoints[e])
    node_coords = np.vstack([mesh.vertices, edge_midpoints])

    boundary = np.asarray(mesh.boundary)
    boundary_nodes = np.empty((len(boundary), 3), dtype=int)
    boundary_nodes[:, :2] = boundary
    for i, (a, b) in enumerate(boundary):
        boundary_nodes[i, 2] = edge_node(a, b)

    return element_nodes, node_coords, boundary_nodes


def _reference_subtriangulation(subdivisions: int) -> tuple[FloatArray, IntArray]:
    '''Uniform split of the reference triangle into `subdivisions**2` sub-triangles.

    Returns `(points, triangles)`: the barycentric lattice `{(i/k, j/k): i + j <= k}`
    in reference `(xi, eta)` coordinates, and a structured triangulation of it. Sampling
    a P2 element's shape functions at this lattice traces the element's true (curved)
    image; the triangles tessellate it into the flat pieces matplotlib can draw.
    '''
    if subdivisions < 1:
        raise ValueError(f'subdivisions must be at least 1, got {subdivisions}')
    k = subdivisions
    index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    for j in range(k + 1):
        for i in range(k - j + 1):
            index[(i, j)] = len(points)
            points.append((i / k, j / k))
    triangles: list[list[int]] = []
    for j in range(k):
        for i in range(k - j):
            triangles.append([index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]])
            if i < k - j - 1:   # the downward triangle filling the gap above
                triangles.append(
                    [index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]])
    return np.array(points, dtype=float), np.array(triangles, dtype=int)


@dataclass(frozen=True)
class PlotTessellation:
    '''A curved element space sampled into flat sub-triangles for display.

    `points`/`triangles` are a fine straight-sided triangulation whose vertices sit on
    the true element geometry, so matplotlib draws a curved boundary as a chord chain
    fine enough to read as smooth. `interpolate` samples a per-node field at the same
    points, so a P2 field shows its within-element curvature instead of being flattened
    to one triangle per element. A display tessellation only: it adds no error to the
    solve, it just controls how faithfully the computed geometry and field are drawn.
    '''
    points: FloatArray            # (n_el * n_sub, spatial)
    triangles: IntArray           # (n_el * n_ref_tris, 3) into `points`
    _sample: FloatArray           # (n_sub, N) shape functions at the reference sub-points
    _element_nodes: IntArray      # (n_el, N) global node index per element

    def interpolate(self, nodal: FloatArray) -> FloatArray:
        '''Sample a per-node field at the tessellation points, aligned with `points`.

        `nodal` is one value (or component vector) per space node; the result is the
        field evaluated at every sub-point through the same shape functions the geometry
        used, so a quadratic field is drawn quadratically within each element.
        '''
        vals = np.asarray(nodal)[self._element_nodes]              # (n_el, N[, comp])
        sampled = np.einsum('sn,en...->es...', self._sample, vals)  # (n_el, n_sub[, comp])
        return sampled.reshape(-1, *sampled.shape[2:])


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

    # -- display tessellation -----------------------------------------------

    def tessellation(self, subdivisions: int = 3,
                     node_coords: FloatArray | None = None) -> PlotTessellation:
        '''Sample every element into `subdivisions**2` flat sub-triangles on its true
        geometry, for a plot that follows a curved boundary and a P2 field.

        Each element's shape functions are evaluated at a reference sub-lattice and
        mapped through the element's own nodes, the same geometry map `Element.geometry`
        integrates over, so boundary sub-points land on the true curve and interior ones
        stay flat. See `PlotTessellation`.

        `node_coords` overrides the node positions the sub-points are mapped through,
        `(n_nodes, spatial)` like `self.node_coords`. Passing the deformed positions
        (node coords plus a nodal displacement) tessellates the deformed configuration, so
        a P2 field can be drawn on the warped shape rather than only the reference one.
        '''
        coords_of = self.node_coords if node_coords is None else np.asarray(node_coords)
        ref_points, ref_triangles = _reference_subtriangulation(subdivisions)
        sample = self.element_type.shape_values(ref_points)       # (n_sub, N)
        element_nodes = np.asarray(self.element_nodes)
        coords = coords_of[element_nodes]                         # (n_el, N, spatial)
        points = np.einsum('sn,end->esd', sample, coords)         # (n_el, n_sub, spatial)
        n_el, n_sub = points.shape[0], points.shape[1]
        points = points.reshape(n_el * n_sub, -1)
        offsets = (np.arange(n_el) * n_sub)[:, None, None]
        triangles = (ref_triangles[None] + offsets).reshape(-1, 3)
        return PlotTessellation(points, triangles, sample, element_nodes)

    def boundary_polylines(self, subdivisions: int = 3) -> FloatArray:
        '''`(n_facets, subdivisions + 1, spatial)` boundary facets on their true curve.

        Each boundary facet sampled through the boundary element's geometry map, so a
        curved facet draws as a smooth polyline; a straight facet comes back as a
        straight sampled line.
        '''
        xi = np.linspace(0.0, 1.0, subdivisions + 1)[:, None]     # (k+1, 1)
        sample = self.boundary_type.shape_values(xi)              # (k+1, boundary_N)
        facet_coords = self.node_coords[self.boundary_nodes]      # (n_facets, boundary_N, spatial)
        return np.einsum('sn,end->esd', sample, facet_coords)

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

    def interpolate(self, value: FieldValue) -> DofVector:
        '''The nodal interpolant of a field as a DOF vector: `value` (a constant, a
        per-component constant, or a callable of position) evaluated at every node of
        the space, components interleaved per node.

        The way to build an initial condition or a comparison field. A load that must
        resolve variation within an element is a `LinearForm`.
        '''
        return evaluate_field(value, self.node_coords, self.n_components).flatten()

    def integrate(self, u: VertexField) -> float:
        '''Integral of a nodal field over the domain: the entries of `M @ u` summed,
        which is exact since the shape functions sum to 1.'''
        return float((self.mass_matrix @ u).sum())

    def mean_value(self, u: VertexField) -> float:
        '''Volume-weighted mean of a nodal field.'''
        return self.integrate(u) / self.total_volume

    def gradient(self, u: VertexField) -> FloatArray:
        '''(n_elements, spatial_dim) gradient of a nodal field, one value per element.

        The volume-weighted mean over the element's rule: the exact constant for P1,
        and the centroid value of a straight P2 element's linear gradient.
        '''
        geometry = self.geometry
        weights = geometry.weight_detJ / geometry.weight_detJ.sum(axis=1, keepdims=True)
        return np.einsum('eq,eqi->ei', weights, geometry.gradients(u[self.element_nodes]))

    def nodal_gradient(self, u: VertexField, method: str = 'average') -> VertexField:
        '''(n_nodes, spatial_dim) continuous gradient of a nodal field.

        `'average'` evaluates each element's gradient at its own nodes and volume-averages
        the elements sharing a node; `'l2'` projects the gradient sampled at quadrature
        points onto the nodal space. Both read a P2 gradient's variation within the
        element, so a boundary node gets the boundary value rather than an interior one.
        For P1 both agree with `recover_nodal(gradient(u), method)`.
        '''
        u_elements = np.asarray(u)[self.element_nodes]
        if method == 'average':
            return self.average_to_nodal(self.geometry_at_nodes.gradients(u_elements))
        if method == 'l2':
            geometry = self.geometry_at(2 * self.element_type.SHAPE_DEGREE)
            return self.project_to_nodal(geometry.gradients(u_elements), geometry)
        raise ValueError(f"unknown recovery method {method!r}; use 'average' or 'l2'")

    def element_field_hessian(self, u_elements: FloatArray) -> FloatArray:
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

    # -- projections between element and nodal fields -----------------------

    def recover_nodal(self, values: ElementField, method: str = 'average') -> VertexField:
        '''Recover a continuous nodal field from a per-element one.

        Takes `(n_elements,)` or `(n_elements, *component_shape)` and returns
        `(n_nodes,)` or `(n_nodes, *component_shape)`, each component recovered
        independently. This is the smooth field nodal output and P2 plotting draw.

        `method` picks the recovery:

        - `'average'` (default): the volume-weighted nodal average. Local and cheap.
          Volume-weighted so that on a graded mesh a sliver does not count as much as
          the large element beside it.
        - `'l2'`: the global L2 projection onto the nodal space, `M q = ∫ f φ`. A mass
          solve, more accurate on a graded mesh, and it conserves the field's integral.
        '''
        values = np.asarray(values, dtype=float)
        if len(values) != len(self.element_nodes):
            raise ValueError(
                f'expected one value per element ({len(self.element_nodes)}), '
                f'got {len(values)}'
            )
        if method == 'average':
            return self._recover_nodal_average(values)
        if method == 'l2':
            return self._recover_nodal_l2(values)
        raise ValueError(f"unknown recovery method {method!r}; use 'average' or 'l2'")

    def _recover_nodal_average(self, values: FloatArray) -> VertexField:
        '''The volume-weighted nodal average of a per-element field.'''
        n_local = self.element_nodes.shape[1]
        per_node = np.repeat(values[:, None, ...], n_local, axis=1)
        return self.average_to_nodal(per_node)

    def average_to_nodal(self, values_at_nodes: FloatArray) -> VertexField:
        '''Volume-weighted nodal average of a field sampled at each element's nodes.

        `values_at_nodes` is `(n_elements, N, *component_shape)`: element e's reading
        of the field at its N nodes, as `fields_at(space.geometry_at_nodes, ...)`
        produces. A node shared by several elements gets their readings averaged,
        weighted by element volume. This is `recover_nodal('average')` for a field
        that varies within the element; for an element-constant field the two agree.
        '''
        values_at_nodes = np.asarray(values_at_nodes, dtype=float)
        nodes = self.element_nodes
        if values_at_nodes.shape[:2] != nodes.shape:
            raise ValueError(
                f'expected one value per element node {nodes.shape}, '
                f'got {values_at_nodes.shape[:2]}'
            )
        weights = self.element_volumes
        n_local = nodes.shape[1]
        flat = nodes.ravel()
        trailing = values_at_nodes.shape[2:]

        # Scatter each element's volume-weighted readings onto its own nodes; `add.at`
        # accumulates, so a shared node sums its elements' contributions.
        w = weights.reshape((-1, 1) + (1,) * len(trailing))
        weighted = (values_at_nodes * w).reshape(len(flat), *trailing)
        sums = np.zeros((self.n_nodes, *trailing))
        np.add.at(sums, flat, weighted)
        norms = np.bincount(flat, weights=np.repeat(weights, n_local), minlength=self.n_nodes)
        # Every referenced node belongs to at least one element; an unreferenced one
        # would divide by zero, so it keeps 0 instead.
        norms = np.where(norms > 0, norms, 1.0).reshape((-1,) + (1,) * len(trailing))
        return sums / norms

    def _recover_nodal_l2(self, values: FloatArray) -> VertexField:
        '''The L2 projection of a per-element field onto the nodal space: solve M q = b.

        `b_i = ∫ f φ_i`, and with `f` element-constant that is `Σ_e f_e ∫_e φ_i`, built
        from the same rule the mass matrix integrates with so `M⁻¹ b` is the exact
        projection. Each trailing component is one right-hand side against the shared
        scalar mass matrix.
        '''
        geometry = self.geometry
        # The integral of each shape function over each element: (n_elements, N).
        shape_integral = np.einsum('eq,qn->en', geometry.weight_detJ, geometry.shape)
        nodes = self.element_nodes
        n_local = nodes.shape[1]
        trailing = values.shape[1:]

        contrib = (shape_integral.reshape(len(values), n_local, *((1,) * len(trailing)))
                   * values[:, None, ...])
        load = np.zeros((self.n_nodes, *trailing))
        np.add.at(load, nodes.ravel(), contrib.reshape(len(values) * n_local, *trailing))

        projected = spsolve(self._nodal_mass_matrix, load.reshape(self.n_nodes, -1))
        return np.asarray(projected).reshape(self.n_nodes, *trailing)

    def project_to_nodal(self, values_qp: FloatArray, geometry: ElementGeometry) -> VertexField:
        '''L2-project a per-quadrature-point field onto the continuous nodal space.

        `values_qp` is `(n_elements, n_qp, *component_shape)`, a field sampled at
        `geometry`'s quadrature points. Solves `M q = b` with `b_i = Σ_e Σ_q w_eq φ_i(x_eq)
        f_eq`, the L2 projection of that field, recovering each trailing component against
        the shared scalar mass matrix.

        This generalizes `recover_nodal('l2')` from an element-constant field to one that
        varies within the element, as a P2 derived field does. `geometry` must be the space's own geometry so its shape functions
        and node numbering line up with the nodal space `M` is built on.
        '''
        values_qp = np.asarray(values_qp, dtype=float)
        trailing = values_qp.shape[2:]
        nodes = self.element_nodes
        n_local = nodes.shape[1]
        # b[e, n, ...] = Σ_q weight_detJ[e,q] shape[q,n] f[e,q,...]
        contrib = np.einsum('eq,qn,eq...->en...', geometry.weight_detJ, geometry.shape, values_qp)
        load = np.zeros((self.n_nodes, *trailing))
        np.add.at(load, nodes.ravel(), contrib.reshape(len(nodes) * n_local, *trailing))
        projected = spsolve(self._nodal_mass_matrix, load.reshape(self.n_nodes, -1))
        return np.asarray(projected).reshape(self.n_nodes, *trailing)

    @cached_property
    def _nodal_mass_matrix(self) -> SparseMatrix:
        '''The scalar (n_nodes x n_nodes) consistent mass matrix, for L2 nodal recovery.

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

    def assemble(self, form: BilinearForm, boundary: bool = False) -> SparseMatrix:
        '''Scatter a bilinear form's element matrices into a global matrix.

        `boundary=True` integrates over the boundary facets instead of the volume
        elements. Not cached, since a form may carry material data that changes
        between calls.
        '''
        geometry = self.boundary_geometry if boundary else self._geometry_for(form)
        return self._assemble(form.element_matrices(geometry), boundary=boundary)

    def assemble_load(self, form: LinearForm) -> DofVector:
        '''Scatter a `LinearForm`'s element vectors into the global load vector.

        The vector counterpart of `assemble`: element load vectors summed into the
        DOFs their nodes own, the same scatter `assemble_residual` runs for the
        nonlinear residual. This is the general load path; `problem.Source` is the
        mass-matrix special case that suffices when the source is given at the nodes.
        '''
        geometry = self.geometry_at(form.quadrature_degree)
        vectors = form.element_vectors(geometry)
        return self._volume_vector_scatter.scatter(vectors)

    # -- state-dependent assembly -------------------------------------------
    #
    # A form's element quantities may depend on the current state, so these take
    # `u` and evaluate the form at each element's slice of it. Constraints stay with
    # the caller (the Problem and its solve strategy).

    def _geometry_for(self, form: Form) -> ElementGeometry:
        '''Geometry at a rule that integrates `form` on this element.

        The larger of the element's default and what the form asks for: a quartic
        St-Venant-Kirchhoff energy wants degree 4 on P2. One rule serves the energy,
        residual, and tangent, so the residual is the exact gradient of the quadrature
        energy and Newton sees a matching tangent.
        '''
        degree = max(self.element_type.default_quadrature_degree(),
                     form.quadrature_degree(self.element_type.SHAPE_DEGREE))
        return self.geometry_at(degree)

    def _element_state(self, u: DofVector) -> FloatArray:
        '''(n_elements, N, n_components): each element's slice of the state.'''
        return np.asarray(u, dtype=float).reshape(-1, self.n_components)[self.element_nodes]

    def total_energy(self, form: Form, u: DofVector) -> float:
        '''Sum a form's element energies at state `u`: the scalar Pi(u).'''
        return float(form.element_energies(self._geometry_for(form), self._element_state(u)).sum())

    def assemble_residual(self, form: Form, u: DofVector) -> DofVector:
        '''Scatter element residuals at `u` into the global residual, shape (n_dofs,).'''
        residuals = form.element_residuals(self._geometry_for(form), self._element_state(u))
        return self._volume_vector_scatter.scatter(residuals)

    def assemble_tangent(self, form: Form, u: DofVector) -> SparseMatrix:
        '''Scatter element tangents at `u` into the global tangent, shape (n_dofs, n_dofs).'''
        return self._assemble(form.element_tangents(self._geometry_for(form), self._element_state(u)))

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

    def _assemble(
        self,
        element_matrices: FloatArray,
        boundary: bool = False,
    ) -> SparseMatrix:
        '''Scatter (n_elements, k, k) element matrices into the global operator,
        summing entries that share a (row, col), through the cached `_ScatterPlan`.'''
        plan = self._boundary_scatter if boundary else self._volume_scatter
        return plan.scatter(element_matrices)
