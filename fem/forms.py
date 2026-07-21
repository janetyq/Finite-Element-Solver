"""Bilinear forms: the integrand a finite-element assembly scatters.

A `Form` is the assembly-ready view of a bilinear form `a(u, v)`, the way
`ResolvedBC` is the assembly-ready view of a `BoundaryConditions`. It answers one
question -- "what is the element matrix for element `e_idx`?" -- and
`FunctionSpace.assemble` scatters the results into the global matrix. Every
matrix the linear solvers assemble -- mass, stiffness, boundary mass -- is a
`Form`, so nothing reaches into element internals with an ad-hoc loop.

Every element matrix here has the shape `Gᵀ C G · volume`, where G is a
gradient-like operator built from the element's shape-function gradients and C is
the material. The Laplacian is the case G = grad_phi, C = I (no material). Linear
elasticity is G = B (the strain-displacement matrix), C = D (the material's Hooke
matrix). Splitting G from C is what lets `Element` be pure geometry: it supplies
`grad_phi`, and the form knows what physics to build from it.

`strain_displacement` fixes the Voigt ordering of the strain vector, which must
match `fem.materials.hooke_matrix`; the two are contracted together.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fem.elements import LinearElement
from fem.materials import LinearElasticMaterial
from fem.typing import FloatArray, Matrix


def strain_displacement(grad_phi: FloatArray) -> Matrix:
    '''Voigt strain-displacement matrix B: nodal DOFs -> element strain vector.

    Strain is ordered [xx, yy, (zz,) engineering shears] to match the rows and
    columns of `fem.materials.hooke_matrix`. DOFs are interleaved per node, so
    column `reference_dim*n + d` is node n's displacement component d.
    '''
    n_nodes, reference_dim = grad_phi.shape
    if reference_dim == 2:
        b, c = grad_phi.T
        B = np.zeros((3, 2 * n_nodes))
        B[0, 0::2] = b
        B[1, 1::2] = c
        B[2, 0::2] = c
        B[2, 1::2] = b
        return B
    if reference_dim == 3:
        a, b, c = grad_phi.T
        B = np.zeros((6, 3 * n_nodes))
        B[0, 0::3] = a
        B[1, 1::3] = b
        B[2, 2::3] = c
        B[3, 0::3] = b
        B[3, 1::3] = a
        B[4, 1::3] = c
        B[4, 2::3] = b
        B[5, 0::3] = c
        B[5, 2::3] = a
        return B
    raise NotImplementedError(
        f'no strain-displacement matrix for reference_dim={reference_dim}'
    )


class Form(Protocol):
    '''The element-matrix integrand for a bilinear form.'''

    def element_matrix(self, element: LinearElement, e_idx: int) -> Matrix:
        '''The dense element stiffness for `element`, index `e_idx` in the mesh.'''
        ...


@dataclass(frozen=True)
class MassForm:
    '''The mass form ∫ u·v -- the consistent P1 mass matrix.

    The scalar `∫ phi_i phi_j` is element geometry; a k-component field repeats it
    once per component, which is the Kronecker product with the k×k identity: DOFs
    are interleaved per node, so entry (k*a + d, k*b + e) is the scalar M[a, b]
    when d == e and zero otherwise. Used both as an operator (the mass terms of
    the time-steppers) and as a system matrix (an L2 projection solves M u = b).
    '''
    n_components: int = 1

    def element_matrix(self, element: LinearElement, e_idx: int) -> Matrix:
        M = np.kron(element.calculate_mass_matrix(), np.eye(self.n_components))
        return M.astype(np.float64)


@dataclass(frozen=True)
class LaplacianForm:
    '''The scalar Laplacian ∫ ∇u·∇v -- material-free, so G = grad_phi, C = I.'''

    def element_matrix(self, element: LinearElement, e_idx: int) -> Matrix:
        return element.grad_phi @ element.grad_phi.T * element.volume


@dataclass(frozen=True)
class LinearElasticForm:
    '''Small-strain linear elasticity ∫ ε(u):D:ε(v), so G = B, C = D.'''
    material: LinearElasticMaterial

    def element_matrix(self, element: LinearElement, e_idx: int) -> Matrix:
        B = strain_displacement(element.grad_phi)
        D = self.material.constitutive_matrix(element.reference_dim, e_idx)
        return B.T @ D @ B * element.volume
