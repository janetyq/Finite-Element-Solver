"""Element types and the batched geometry they produce.

An element type is a *stateless* description of a shape: how many nodes it has,
what its boundary facets are, and how to turn node coordinates into shape-function
gradients and a measure. It holds no per-element data, so there is exactly one
`LinearTetrahedralElement` in a program rather than one per tet.

The per-element data lives in `ElementGeometry`, which holds it for the whole mesh
at once: an `(n_elements, n_qp, N, spatial_dim)` array of `grad_phi` -- one shape
gradient per element and quadrature point -- with the matching `(n_elements, n_qp)`
quadrature weights. That shape is what lets `fem.forms` compute every element matrix
in one vectorized pass instead of a Python loop -- assembly was the dominant cost of
a 3D solve when this was a loop over per-element objects. A linear element's gradient
is constant across the points, so P1 is the single-point (`n_qp == 1`) special case.

Two quantities are easy to confuse and are kept distinct throughout:
`reference_dim` is the dimension of the element itself (2 for a triangle), while
`spatial_dim` is the dimension it is embedded in. They differ exactly for the
boundary facets of a 3D mesh -- a triangle in 3D -- which is why the Jacobian
below is not assumed square.
"""
from dataclasses import dataclass
from math import factorial
from typing import ClassVar

import numpy as np

from fem.quadrature import QuadratureRule, quadrature_rule
from fem.typing import ElementVertices, FloatArray, Matrix


class Element:
    '''Base class for a simplex element: a shape, its nodes, and its shape functions.

    An element is defined by its node count `N`, the polynomial degree of its shape
    functions `SHAPE_DEGREE`, and the boundary-facet element `SUB_TYPE`. The shared
    machinery -- quadrature selection, the reference mass matrix, and the batched
    geometry -- lives here; a subclass supplies only `shape_values` / `shape_gradients`
    and, for a higher-order element, its own `reference_dim`.
    '''
    # Annotations without values: a concrete element type must supply these, and
    # reaching one on the base raises rather than yielding a None that would only
    # fail later inside the arithmetic.
    N: ClassVar[int]
    SHAPE_DEGREE: ClassVar[int]                    # polynomial degree of the shape functions
    SUB_TYPE: ClassVar[type['Element'] | None]     # element of a boundary facet

    @classmethod
    def reference_dim(cls) -> int:
        '''Dimension of the element itself: 1 for a line, 2 for a triangle, 3 for a tet.

        Distinct from the spatial dimension: a triangle embedded in 3D has
        reference_dim 2 and spatial_dim 3. Equals `N - 1` only for a *linear*
        simplex, so a higher-order element overrides it -- a 6-node quadratic
        triangle is still a 2D element.
        '''
        return cls.N - 1

    @classmethod
    def n_corners(cls) -> int:
        '''The simplex corners that define the affine geometry map: reference_dim + 1.

        A quadratic element has more nodes than corners; the extra ones carry field
        DOFs but do not move the (straight-sided) geometry, so only the corners enter
        the Jacobian. Nodes are ordered corners-first, which is what lets this take a
        prefix.
        '''
        return cls.reference_dim() + 1

    @classmethod
    def quadrature(cls, min_degree: int) -> QuadratureRule:
        '''The cheapest rule on this element's reference simplex exact to `min_degree`.'''
        return quadrature_rule(cls.reference_dim(), min_degree)

    @classmethod
    def default_quadrature_degree(cls) -> int:
        '''The degree a plain stiffness needs: the shape-gradient product's degree.

        A gradient drops one degree, and the stiffness pairs two of them, so the
        integrand is `2 * (SHAPE_DEGREE - 1)` -- 0 for P1 (a single point suffices),
        2 for P2. Floored at 1 so it always names a real rule.
        '''
        return max(1, 2 * (cls.SHAPE_DEGREE - 1))

    @classmethod
    def shape_values(cls, points: FloatArray) -> FloatArray:
        '''(n_points, N) shape functions at reference `points` (n_points, reference_dim).'''
        raise NotImplementedError

    @classmethod
    def shape_gradients(cls, points: FloatArray) -> FloatArray:
        '''(n_points, N, reference_dim) reference-coordinate shape gradients.'''
        raise NotImplementedError

    @classmethod
    def reference_mass_matrix(cls) -> Matrix:
        '''Consistent mass matrix per unit measure -- `int phi_i phi_j` over the
        reference simplex, divided by the simplex measure so `MassForm` recovers the
        element's mass by scaling with its volume.

        Integrated by quadrature at a degree that captures `phi_i phi_j` exactly
        (twice the shape degree). Exact for an affine element, whose Jacobian is
        constant, so the physical mass is just this scaled by the measure.
        '''
        rule = cls.quadrature(2 * cls.SHAPE_DEGREE)
        shape = cls.shape_values(rule.points)           # (n_qp, N)
        reference_measure = 1.0 / factorial(cls.reference_dim())
        return np.einsum('qi,qj,q->ij', shape, shape, rule.weights) / reference_measure

    @classmethod
    def geometry(
        cls, element_vertices: ElementVertices, rule: QuadratureRule | None = None,
    ) -> 'ElementGeometry':
        '''Batched quadrature-aware geometry for `(n_elements, N, spatial_dim)` coords.

        `rule` defaults to the degree a plain stiffness needs on this element. A
        caller integrating a higher-degree form (a mass matrix, a variable
        coefficient) passes a rule of the degree it needs.

        The geometry map is affine -- subparametric: the Jacobian is built from the
        `n_corners` simplex corners alone, while the field's shape functions may be
        higher order. So the Jacobian is constant per element and only the reference
        shape gradients differ between quadrature points. P1 falls out as the case
        where the corners *are* all the nodes and the gradients are constant.
        '''
        X = np.asarray(element_vertices, dtype=np.float64)
        if X.ndim != 3 or X.shape[1] != cls.N:
            raise ValueError(
                f'{cls.__name__}.geometry expects (n_elements, {cls.N}, spatial_dim) '
                f'coordinates, got shape {X.shape}'
            )
        rule = rule if rule is not None else cls.quadrature(cls.default_quadrature_degree())

        # Columns of J are the edge vectors from corner 0, so J maps the reference
        # simplex onto the element: (n_elements, spatial_dim, reference_dim). Only the
        # corners enter -- the affine map does not see the higher-order nodes.
        corners = X[:, :cls.n_corners()]
        J = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        spatial_dim, reference_dim = J.shape[1], J.shape[2]

        if spatial_dim == reference_dim:
            # The element fills its ambient space: J is invertible and its
            # determinant is the volume scaling directly.
            J_inv = np.linalg.inv(J)
            scale = np.abs(np.linalg.det(J))
        else:
            # An embedded element (a triangular facet of a tet mesh) has a tall J
            # with no inverse. The pseudo-inverse gives the gradient *within* the
            # element's own plane, and the Gram determinant gives its measure --
            # sqrt(det(J^T J)) is |a x b| for a triangle in 3D. Both reduce to the
            # square case above, which is preferred where it applies because it
            # avoids squaring the condition number.
            J_inv = np.linalg.pinv(J)
            gram = np.swapaxes(J, 1, 2) @ J
            scale = np.sqrt(np.abs(np.linalg.det(gram)))

        dshape = cls.shape_gradients(rule.points)   # (n_qp, N, reference_dim)
        shape = cls.shape_values(rule.points)       # (n_qp, N)
        return ElementGeometry(
            element_type=cls,
            rule=rule,
            shape=shape,
            # (n_qp, N, r) @ (n_el, r, s) -> (n_el, n_qp, N, s): the reference shape
            # gradients mapped through each element's inverse Jacobian.
            grad_phi=np.einsum('qnr,ers->eqns', dshape, J_inv),
            # (n_el, n_qp): the reference weight at each point times the element's
            # measure scale. The reference weights sum to 1/d!, so this sums over
            # points to `scale / d!` -- the closed-form element volume.
            weight_detJ=scale[:, None] * rule.weights[None, :],
            # (n_el, n_qp, spatial): where each quadrature point lands in space,
            # interpolated through the shape functions -- for a form or load that
            # samples a coefficient or source there.
            points=np.einsum('qn,ens->eqs', shape, X),
        )


class LinearElement(Element):
    '''Base class for linear (P1) simplex elements.

    Shape function phi(x) = a + b*x_1 + c*x_2 + ... + z*x_{N-1}, so the gradient
    is constant over the element -- the reason a P1 assembly reduces to one
    Jacobian per element and a single quadrature point.
    '''
    SHAPE_DEGREE = 1

    @classmethod
    def _dshape(cls) -> Matrix:
        '''(N, N-1) shape-function gradients on the reference simplex.

        Constant per element type -- the reference simplex does not move -- so the
        only per-element work is mapping these through the inverse Jacobian.
        '''
        return np.vstack([-np.ones(cls.N - 1), np.eye(cls.N - 1)])

    @classmethod
    def reference_mass_matrix(cls) -> Matrix:
        '''Consistent P1 mass matrix per unit measure, `(1 + delta_ij) / (N (N+1))`.

        Overrides the quadrature-integrated base with the exact rational closed form,
        so the P1 mass stays bit-identical to what it was before the quadrature layer.
        '''
        return (np.ones((cls.N, cls.N)) + np.eye(cls.N)) / (cls.N * (cls.N + 1))

    @classmethod
    def shape_values(cls, points: FloatArray) -> FloatArray:
        '''(n_points, N) shape functions evaluated at reference `points` (n_points, N-1).

        Barycentric: phi_0 = 1 - sum(xi) and phi_i = xi_i, so the first column is
        the node-0 hat and the rest are the reference coordinates themselves. Nodal
        (1 at its own node, 0 at the others), which is what makes a DOF the value at
        its node.
        '''
        P = np.atleast_2d(np.asarray(points, dtype=float))
        first = 1.0 - P.sum(axis=1, keepdims=True)
        return np.concatenate([first, P], axis=1)

    @classmethod
    def shape_gradients(cls, points: FloatArray) -> FloatArray:
        '''(n_points, N, N-1) reference-coordinate shape gradients.

        Constant for a linear element -- the same `_dshape` at every point -- so
        this broadcasts it over the requested points. `geometry` maps these through
        the inverse Jacobian to get physical gradients.
        '''
        n_points = len(np.atleast_2d(np.asarray(points, dtype=float)))
        return np.broadcast_to(cls._dshape(), (n_points, cls.N, cls.N - 1))


class LinearLineElement(LinearElement):
    '''1D linear element. Shape function phi(x) = a + b*x.'''
    N = 2
    SUB_TYPE = None # TODO: add subtype point element? need to test 1D solve


class LinearTriangleElement(LinearElement):
    '''2D linear triangle element. Shape function phi(x) = a + b*x + c*y.'''
    N = 3
    SUB_TYPE = LinearLineElement


class LinearTetrahedralElement(LinearElement):
    '''3D linear tetrahedral element.'''
    N = 4
    SUB_TYPE = LinearTriangleElement


class QuadraticLineElement(Element):
    '''1D quadratic element: two endpoint nodes and one midpoint, ordered [start, end, mid].

    The boundary facet of a quadratic triangle. Its shape functions are the P2
    Lagrange basis on [0, 1], quadratic rather than linear, so their gradients are
    no longer constant.
    '''
    N = 3
    SHAPE_DEGREE = 2
    SUB_TYPE = None

    @classmethod
    def reference_dim(cls) -> int:
        return 1

    @classmethod
    def shape_values(cls, points: FloatArray) -> FloatArray:
        P = np.atleast_2d(np.asarray(points, dtype=float))
        xi = P[:, 0]
        l0, l1 = 1.0 - xi, xi   # barycentric: start, end
        return np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), 4 * l0 * l1], axis=1)

    @classmethod
    def shape_gradients(cls, points: FloatArray) -> FloatArray:
        P = np.atleast_2d(np.asarray(points, dtype=float))
        xi = P[:, 0]
        g = np.zeros((len(P), 3, 1))
        g[:, 0, 0] = 4 * xi - 3
        g[:, 1, 0] = 4 * xi - 1
        g[:, 2, 0] = 4 - 8 * xi
        return g


class QuadraticTriangleElement(Element):
    '''2D quadratic triangle: three corner nodes and three edge-midpoint nodes.

    Nodes are ordered corners-first -- [c0, c1, c2, m12, m02, m01] -- where mij is the
    midpoint of the edge between corners i and j, and each edge-midpoint hat is the one
    opposite the corner it is *not* named by. That ordering is what lets the affine map
    read the first three nodes as the simplex corners, and it must match the (element ->
    global node) map `FunctionSpace` builds. The field is quadratic (O(h^3) in L2) while
    the geometry stays straight-sided.
    '''
    N = 6
    SHAPE_DEGREE = 2
    SUB_TYPE = QuadraticLineElement

    @classmethod
    def reference_dim(cls) -> int:
        return 2

    @classmethod
    def shape_values(cls, points: FloatArray) -> FloatArray:
        P = np.atleast_2d(np.asarray(points, dtype=float))
        xi, eta = P[:, 0], P[:, 1]
        l0, l1, l2 = 1.0 - xi - eta, xi, eta   # barycentric of the three corners
        return np.stack([
            l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
            4 * l1 * l2, 4 * l0 * l2, 4 * l0 * l1,
        ], axis=1)

    @classmethod
    def shape_gradients(cls, points: FloatArray) -> FloatArray:
        P = np.atleast_2d(np.asarray(points, dtype=float))
        xi, eta = P[:, 0], P[:, 1]
        l0, l1, l2 = 1.0 - xi - eta, xi, eta
        g = np.zeros((len(P), 6, 2))
        # Corner hats phi_i = l_i(2 l_i - 1); grad l0 = (-1, -1), l1 = (1, 0), l2 = (0, 1).
        g[:, 0, 0] = g[:, 0, 1] = -(4 * l0 - 1)
        g[:, 1, 0] = 4 * l1 - 1
        g[:, 2, 1] = 4 * l2 - 1
        # Edge hats: m12 = 4 l1 l2, m02 = 4 l0 l2, m01 = 4 l0 l1.
        g[:, 3, 0] = 4 * l2
        g[:, 3, 1] = 4 * l1
        g[:, 4, 0] = -4 * l2
        g[:, 4, 1] = 4 * l0 - 4 * l2
        g[:, 5, 0] = 4 * l0 - 4 * l1
        g[:, 5, 1] = -4 * l1
        return g


@dataclass(frozen=True)
class ElementGeometry:
    '''Shape values, shape-function gradients, and quadrature weights for one mesh.

    The batched, quadrature-aware geometry every form integrates against. `grad_phi`
    carries a gradient per (element, quadrature point); a linear element's is
    constant across the points, so P1 assembly is the single-point special case --
    which is why one assembly path serves P1 and higher orders alike.

    Immutable, and cached on the `FunctionSpace` that built it: it is valid only
    while the mesh underneath it is not mutated, the same contract the space's
    operators have.
    '''
    element_type: type[Element]
    rule: QuadratureRule
    # (n_qp, N) -- shape functions at the quadrature points, for mass and load
    # integrals; the gradients alone are not enough once the integrand samples the
    # field's value rather than only its slope.
    shape: FloatArray
    # (n_elements, n_qp, N, spatial_dim) -- gradient of each shape function at each
    # quadrature point. The last axis is the *spatial* dimension, so for an embedded
    # facet it is wider than the element's own reference_dim.
    grad_phi: FloatArray
    # (n_elements, n_qp) -- the quadrature weight times |det J| at each point; the
    # coefficient every integrand is summed against. Replaces the old scalar
    # `volumes`, which is now these summed over the points.
    weight_detJ: FloatArray
    # (n_elements, n_qp, spatial_dim) -- physical coordinates of the quadrature
    # points. Empty of meaning for a constant-coefficient form, which never samples
    # anything; carried so a variable coefficient or source can be evaluated there.
    points: FloatArray

    @property
    def n_elements(self) -> int:
        return self.grad_phi.shape[0]

    @property
    def n_qp(self) -> int:
        return self.grad_phi.shape[1]

    @property
    def reference_dim(self) -> int:
        return self.element_type.reference_dim()

    @property
    def spatial_dim(self) -> int:
        return self.grad_phi.shape[-1]

    @property
    def volumes(self) -> FloatArray:
        '''(n_elements,) element measure -- the quadrature weights summed per element.'''
        return self.weight_detJ.sum(axis=1)

    @property
    def total_volume(self) -> float:
        return float(self.volumes.sum())

    def gradients(self, u_elements: FloatArray) -> FloatArray:
        '''Field gradient at every quadrature point, from per-element nodal values.

        `u_elements` is `(n_elements, N)` for a scalar field or
        `(n_elements, N, n_components)` for a vector one; the result carries a
        leading (element, quadrature point) pair, then the spatial axis in the same
        position `calculate_gradient` used to put it. Constant across the points for
        P1, so a P1 caller reads any point and gets the element's one value.
        '''
        return np.einsum('eqni,en...->eqi...', self.grad_phi, u_elements)

