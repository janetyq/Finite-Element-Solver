"""Element types and the batched geometry they produce.

An element type is a *stateless* description of a shape: how many nodes it has,
what its boundary facets are, and how to turn node coordinates into shape-function
gradients and a measure. It holds no per-element data, so there is exactly one
`LinearTetrahedralElement` in a program rather than one per tet.

The per-element data lives in `ElementGeometry`, which holds it for the whole mesh
at once: a single `(n_elements, N, spatial_dim)` array of `grad_phi` and a single
`(n_elements,)` array of measures. That shape is what lets `fem.forms` compute
every element matrix in one vectorized pass instead of a Python loop -- assembly
was the dominant cost of a 3D solve when this was a loop over per-element objects.

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
    def geometry(cls, element_vertices: ElementVertices) -> 'ElementGeometry':
        '''Batched geometry for `(n_elements, N, spatial_dim)` node coordinates.'''
        X = np.asarray(element_vertices, dtype=np.float64)
        if X.ndim != 3 or X.shape[1] != cls.N:
            raise ValueError(
                f'{cls.__name__}.geometry expects (n_elements, {cls.N}, spatial_dim) '
                f'coordinates, got shape {X.shape}'
            )
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

        return ElementGeometry(
            element_type=cls,
            # (N, N-1) @ (n_elements, N-1, spatial_dim) -> (n_elements, N, spatial_dim)
            grad_phi=cls._dshape() @ J_inv,
            # The reference simplex has measure 1/d!, and J scales it by `scale`.
            volumes=scale / factorial(reference_dim),
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
    '''Shape-function gradients and measures for every element of one mesh.

    The batched counterpart of what used to be a list of per-element objects.
    Immutable, and cached on the `FunctionSpace` that built it: it is valid only
    while the mesh underneath it is not mutated, the same contract the space's
    operators have.
    '''
    element_type: type[LinearElement]
    # (n_elements, N, spatial_dim) -- gradient of each shape function, constant
    # over the element for P1. The last axis is the *spatial* dimension, so for an
    # embedded facet it is wider than the element's own reference_dim.
    grad_phi: FloatArray
    # (n_elements,) element measure -- length, area, or volume.
    volumes: FloatArray

    @property
    def n_elements(self) -> int:
        return len(self.volumes)

    @property
    def reference_dim(self) -> int:
        return self.element_type.reference_dim()

    @property
    def spatial_dim(self) -> int:
        return self.grad_phi.shape[-1]

    @property
    def total_volume(self) -> float:
        return float(self.volumes.sum())

    def gradients(self, u_elements: FloatArray) -> FloatArray:
        '''Gradient of a field over every element, from its per-element nodal values.

        `u_elements` is `(n_elements, N)` for a scalar field or
        `(n_elements, N, n_components)` for a vector one; the result carries the
        spatial axis in the same position `calculate_gradient` used to put it.
        '''
        return np.einsum('eni,en...->ei...', self.grad_phi, u_elements)

