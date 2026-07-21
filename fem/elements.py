from typing import Any, ClassVar

import numpy as np

from fem.geometry import calculate_polygon_area, calculate_tetrahedron_volume
from fem.typing import FloatArray, Matrix, Vertices


class Element:
    '''
    Base class for elements with N nodes

    '''
    # Annotation without a value: a concrete element type must supply its node
    # count, and reaching this attribute on the base raises rather than yielding
    # a None that would only fail later inside the shape-function arithmetic.
    N: ClassVar[int]
    volume: float

    def __init__(self, vertices: Vertices) -> None:
        self.vertices = vertices

    @property
    def reference_dim(self) -> int:
        '''Dimension of the element itself: 1 for a line, 2 for a triangle, 3 for a tet.

        Equals `N - 1` for a simplex, which is what the arithmetic below spells
        out longhand. Distinct from `Mesh.spatial_dim`: a triangle embedded in 3D
        has reference_dim 2 and spatial_dim 3. They coincide only when the element
        fills its ambient space, so code using one to mean the other happens to
        work for planar triangle and tet meshes and nowhere else.
        '''
        return self.N - 1


class LinearElement(Element):
    '''
    Base class for linear elements

    N nodes

    Shape function phi(x) = a + b*x_1 + c*x_2 + ... + z * x_{N-1}
    '''
    SUB_TYPE: ClassVar[type['LinearElement'] | None]

    def __init__(self, vertices: Vertices) -> None:
        super().__init__(vertices)

        dshape_dphi = np.vstack([-np.ones(self.N-1), np.eye(self.N-1)])
        J = (self.vertices[1:] - self.vertices[0]).T
        self.grad_phi: FloatArray = dshape_dphi @ np.linalg.pinv(J)

        self.dF_dx: FloatArray = self.calculate_dF_dx()

    def calculate_mass_matrix(self, n_components: int, **kwargs: Any) -> Matrix:
        '''Consistent P1 mass matrix, `volume * (1 + delta_ij) / (N (N+1))`.

        A vector unknown repeats the scalar matrix once per component, which is
        the Kronecker product with the identity: DOFs are interleaved per node,
        so entry (n*a + d, n*b + e) is M[a, b] when d == e and zero otherwise.
        '''
        M = (np.ones((self.N, self.N)) + np.eye(self.N)) * self.volume / (self.N * (self.N + 1))
        return np.kron(M, np.eye(n_components)).astype(np.float64)

    def calculate_stiffness_matrix(self, n_components: int, **kwargs: Any) -> Matrix:
        if n_components == 1:
            return self.grad_phi @ self.grad_phi.T * self.volume
        # otherwise, the equation is linear elastic
        idx = kwargs['idx']
        B, D = self.calculate_B(), self.calculate_D(kwargs['mu'][idx], kwargs['lamb'][idx])
        return B.T @ D @ B * self.volume

    def calculate_B(self) -> FloatArray:
        '''Strain-displacement matrix, mapping nodal DOFs to element strain.

        Only the area/volume elements define one: there is no 1D elasticity path
        yet, so a line element reaching here is a bug rather than a missing case.
        '''
        raise NotImplementedError(
            f'{type(self).__name__} defines no strain-displacement matrix'
        )

    def calculate_D(self, mu: float, lamb: float) -> Matrix:
        '''Constitutive matrix relating strain to stress, in Lame parameters.'''
        raise NotImplementedError(
            f'{type(self).__name__} defines no constitutive matrix'
        )

    # TODO: haven't checked if these make sense for 1D, 3D
    def deformation_gradient(self, u_element: FloatArray) -> FloatArray:
        # F = I + grad_u = I + grad_phi^T @ u
        return np.eye(self.N-1) + self.grad_phi.T @ u_element

    def calculate_dF_dx(self) -> FloatArray:
        # dF_dx = I x grad_phi^T, TODO: figure out kronecker product
        dF_dx = np.zeros((self.N-1, self.N-1, self.N, self.N-1))
        for i in range(self.N-1):
            for j in range(self.N-1):
                for m in range(self.N):
                    for n in range(self.N-1):
                        if j == n:
                            dF_dx[i, j, m, n] = self.grad_phi[m, i]
        return dF_dx

    def calculate_gradient(self, u_element: FloatArray) -> FloatArray:
        # grad_u = grad_phi @ u
        return self.grad_phi.T @ u_element


class LinearLineElement(LinearElement):
    '''
    1D linear element

    Shape function phi(x) = a + b*x
    '''
    N = 2
    SUB_TYPE = None # TODO: add subtype point element? need to test 1D solve

    def __init__(self, vertices: Vertices) -> None:
        self.volume = float(np.linalg.norm(vertices[1] - vertices[0]))
        super().__init__(vertices)



class LinearTriangleElement(LinearElement): # TODO: perhaps put quadrature in here too?
    '''
    2D linear triangle element

    Shape function phi(x) = a + b*x + c*y
    '''
    N = 3
    SUB_TYPE = LinearLineElement

    def __init__(self, vertices: Vertices) -> None:
        self.volume = calculate_polygon_area(vertices)
        super().__init__(vertices)
        # d2F_dx2 = 0

    def calculate_B(self) -> FloatArray:
        b, c = self.grad_phi.T
        return np.array([[b[0],   0 , b[1],   0 , b[2],   0 ],
                         [  0 , c[0],   0 , c[1],   0 , c[2]],
                         [c[0], b[0], c[1], b[1], c[2], b[2]]])
    
    def calculate_D(self, mu: float, lamb: float) -> Matrix:
        return np.array([
            [2*mu + lamb, lamb, 0],
            [lamb, 2*mu + lamb, 0],
            [0, 0, mu]
        ])


class LinearTetrahedralElement(LinearElement):
    '''
    3D linear tetrahedral element
    '''
    N = 4
    SUB_TYPE = LinearTriangleElement

    def __init__(self, vertices: Vertices) -> None:
        self.volume = calculate_tetrahedron_volume(vertices)
        super().__init__(vertices)

    def calculate_B(self) -> FloatArray:
        a, b, c = self.grad_phi.T
        return np.array([
            [a[0], 0, 0, a[1], 0, 0, a[2], 0, 0, a[3], 0, 0],
            [0, b[0], 0, 0, b[1], 0, 0, b[2], 0, 0, b[3], 0],
            [0, 0, c[0], 0, 0, c[1], 0, 0, c[2], 0, 0, c[3]],
            [b[0], a[0], 0, b[1], a[1], 0, b[2], a[2], 0, b[3], a[3], 0],
            [0, c[0], b[0], 0, c[1], b[1], 0, c[2], b[2], 0, c[3], b[3]],
            [c[0], 0, a[0], c[1], 0, a[1], c[2], 0, a[2], c[3], 0, a[3]]
        ])
    
    def calculate_D(self, mu: float, lamb: float) -> Matrix:
        return np.array([
            [2*mu + lamb, lamb, lamb, 0, 0, 0],
            [lamb, 2*mu + lamb, lamb, 0, 0, 0],
            [lamb, lamb, 2*mu + lamb, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])

