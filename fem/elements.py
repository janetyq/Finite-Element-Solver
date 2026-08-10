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
from typing import ClassVar

import numpy as np

from fem.quadrature import QuadratureRule, quadrature_rule
from fem.typing import ElementVertices, FloatArray, Matrix


class Element:
    '''Base class for elements with N nodes.'''
    # Annotation without a value: a concrete element type must supply its node
    # count, and reaching this attribute on the base raises rather than yielding
    # a None that would only fail later inside the shape-function arithmetic.
    N: ClassVar[int]

    @classmethod
    def reference_dim(cls) -> int:
        '''Dimension of the element itself: 1 for a line, 2 for a triangle, 3 for a tet.

        Equals `N - 1` for a simplex. Distinct from the spatial dimension: a
        triangle embedded in 3D has reference_dim 2 and spatial_dim 3.
        '''
        return cls.N - 1


class LinearElement(Element):
    '''Base class for linear (P1) simplex elements.

    Shape function phi(x) = a + b*x_1 + c*x_2 + ... + z*x_{N-1}, so the gradient
    is constant over the element and the geometry reduces to one Jacobian per
    element.
    '''
    SUB_TYPE: ClassVar[type['LinearElement'] | None]

    @classmethod
    def _dshape(cls) -> Matrix:
        '''(N, N-1) shape-function gradients on the reference simplex.

        Constant per element type -- the reference simplex does not move -- so the
        only per-element work is mapping these through the inverse Jacobian.
        '''
        return np.vstack([-np.ones(cls.N - 1), np.eye(cls.N - 1)])

    @classmethod
    def geometry(
        cls, element_vertices: ElementVertices, rule: QuadratureRule | None = None,
    ) -> 'ElementGeometry':
        '''Batched quadrature-aware geometry for `(n_elements, N, spatial_dim)` coords.

        `rule` defaults to the cheapest rule that integrates this element's own
        stiffness exactly -- a single point for a linear simplex, whose gradients
        are constant. A caller integrating a higher-degree form (a consistent mass
        matrix, a variable coefficient, a higher-order field) passes a rule of the
        degree it needs.

        The geometry map is affine -- straight-sided simplices -- so the Jacobian
        is constant per element and only the reference shape gradients differ
        between quadrature points. P1 falls out as the single-point case: one point,
        one constant gradient, and `weight_detJ` summing to the closed-form measure.
        '''
        X = np.asarray(element_vertices, dtype=np.float64)
        if X.ndim != 3 or X.shape[1] != cls.N:
            raise ValueError(
                f'{cls.__name__}.geometry expects (n_elements, {cls.N}, spatial_dim) '
                f'coordinates, got shape {X.shape}'
            )
        rule = rule if rule is not None else cls.quadrature(1)

        # Columns of J are the edge vectors from node 0, so J maps the reference
        # simplex onto the element: (n_elements, spatial_dim, N-1).
        J = np.swapaxes(X[:, 1:] - X[:, :1], 1, 2)
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
        return ElementGeometry(
            element_type=cls,
            rule=rule,
            shape=cls.shape_values(rule.points),        # (n_qp, N)
            # (n_qp, N, r) @ (n_el, r, s) -> (n_el, n_qp, N, s): the reference shape
            # gradients mapped through each element's inverse Jacobian.
            grad_phi=np.einsum('qnr,ers->eqns', dshape, J_inv),
            # (n_el, n_qp): the reference weight at each point times the element's
            # measure scale. The reference weights sum to 1/d!, so this sums over
            # points to `scale / d!` -- the closed-form element volume.
            weight_detJ=scale[:, None] * rule.weights[None, :],
        )

    @classmethod
    def reference_mass_matrix(cls) -> Matrix:
        '''Consistent scalar P1 mass matrix per unit measure, `(1 + delta_ij) / (N (N+1))`.

        The `int phi_i phi_j` integral divided out by the element's measure. Pure
        geometry and identical for every element of a type, so it is computed once
        and scaled by `ElementGeometry.volumes`. A vector field replicates it per
        component; that is `MassForm`'s job.
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

    @classmethod
    def quadrature(cls, min_degree: int) -> QuadratureRule:
        '''The cheapest rule on this element's reference simplex exact to `min_degree`.'''
        return quadrature_rule(cls.reference_dim(), min_degree)


class LinearLineElement(LinearElement):
    '''1D linear element. Shape function phi(x) = a + b*x.'''
    N = 2
    SUB_TYPE = None # TODO: add subtype point element? need to test 1D solve


class LinearTriangleElement(LinearElement): # TODO: perhaps put quadrature in here too?
    '''2D linear triangle element. Shape function phi(x) = a + b*x + c*y.'''
    N = 3
    SUB_TYPE = LinearLineElement
    # d2F_dx2 = 0


class LinearTetrahedralElement(LinearElement):
    '''3D linear tetrahedral element.'''
    N = 4
    SUB_TYPE = LinearTriangleElement


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
    element_type: type[LinearElement]
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

