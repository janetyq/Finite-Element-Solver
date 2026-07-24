"""Bilinear and nonlinear forms: the integrands a finite-element assembly scatters.

A `Form` is the assembly-ready view of a bilinear form `a(u, v)`, the way
`ResolvedBC` is the assembly-ready view of a `BoundaryConditions`. It answers one
question -- "what are the element matrices for this mesh?" -- and
`FunctionSpace.assemble` scatters the results into the global matrix. Every
matrix the linear solvers assemble -- mass, stiffness, boundary mass -- is a
`Form`, so nothing reaches into element internals with an ad-hoc loop.

`EnergyForm` is the nonlinear sibling: same batched geometry, but the integrand
depends on the current displacement through an energy density whose derivative
chain is evaluated once for the whole mesh rather than element-at-a-time.

Every element matrix here has the shape `Gᵀ C G · volume`, where G is a
gradient-like operator built from the element's shape-function gradients and C is
the material. The Laplacian is the case G = grad_phi, C = I (no material). Linear
elasticity is G = B (the strain-displacement matrix), C = D (the material's Hooke
matrix). Splitting G from C is what lets element types be pure geometry: they
supply `grad_phi`, and the form knows what physics to build from it.

`strain_displacement` fixes the Voigt ordering of the strain vector, which must
match `fem.materials.hooke_matrix`; the two are contracted together.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fem.elements import ElementGeometry
from fem.energies import StrainEnergyDerivatives
from fem.materials import LinearElasticMaterial
from fem.typing import FloatArray


def strain_displacement(grad_phi: FloatArray) -> FloatArray:
    '''Voigt strain-displacement matrices B: nodal DOFs -> element strain vector.

    Batched: takes `(n_elements, n_nodes, dim)` shape-function gradients and
    returns `(n_elements, n_strains, n_nodes*dim)`. Strain is ordered
    [xx, yy, (zz,) engineering shears] to match the rows and columns of
    `fem.materials.hooke_matrix`. DOFs are interleaved per node, so column
    `dim*n + d` is node n's displacement component d.
    '''
    n_elements, n_nodes, dim = grad_phi.shape
    if dim == 2:
        b, c = grad_phi[..., 0], grad_phi[..., 1]
        B = np.zeros((n_elements, 3, 2 * n_nodes))
        B[:, 0, 0::2] = b
        B[:, 1, 1::2] = c
        B[:, 2, 0::2] = c
        B[:, 2, 1::2] = b
        return B
    if dim == 3:
        a, b, c = grad_phi[..., 0], grad_phi[..., 1], grad_phi[..., 2]
        B = np.zeros((n_elements, 6, 3 * n_nodes))
        B[:, 0, 0::3] = a
        B[:, 1, 1::3] = b
        B[:, 2, 2::3] = c
        B[:, 3, 0::3] = b
        B[:, 3, 1::3] = a
        B[:, 4, 1::3] = c
        B[:, 4, 2::3] = b
        B[:, 5, 0::3] = c
        B[:, 5, 2::3] = a
        return B
    raise NotImplementedError(
        f'no strain-displacement matrix for dim={dim}'
    )


class Form(Protocol):
    '''The element-matrix integrand for a bilinear form.'''

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, k, k) dense element matrices for every element at once.

        Batched rather than one element at a time: a P1 element matrix is a
        handful of flops, so evaluating them in a Python loop spends nearly all
        of its time in per-call numpy overhead. One vectorized pass over the
        whole mesh is roughly 30x faster on a 3D solve.
        '''
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

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        # The reference matrix is the same for every element of a type, so the
        # only per-element quantity is the measure it scales by.
        reference = geometry.element_type.reference_mass_matrix()
        block = np.kron(reference, np.eye(self.n_components))
        return geometry.volumes[:, None, None] * block


@dataclass(frozen=True)
class LaplacianForm:
    '''The scalar Laplacian ∫ ∇u·∇v -- material-free, so G = grad_phi, C = I.'''

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        grad_phi = geometry.grad_phi
        return np.einsum('eid,ejd,e->eij', grad_phi, grad_phi, geometry.volumes)


@dataclass(frozen=True)
class LinearElasticForm:
    '''Small-strain linear elasticity ∫ ε(u):D:ε(v), so G = B, C = D.'''
    material: LinearElasticMaterial

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        B = strain_displacement(geometry.grad_phi)
        D = self.material.constitutive_matrices(
            geometry.reference_dim, geometry.n_elements
        )
        # B^T D B scaled by the measure, contracted per element. optimize=True is
        # load-bearing rather than cosmetic: the default left-to-right order
        # forms an (n_elements, k, s) intermediate and runs ~60x slower here.
        return np.einsum('eji,ejk,ekl,e->eil', B, D, B, geometry.volumes, optimize=True)

    def recover(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        '''Element strain, stress (both Voigt), and compliance from nodal displacements.

        The mirror of `element_matrices`: the same B and D, contracted against the
        solved displacement instead of assembled into a stiffness. Recovery lives on
        the form so the constitutive law (B, D) stays in the physics layer rather than
        being rebuilt by whatever solved the system.

        `u_elements` is `(n_elements, N*n_components)` -- each element's nodal DOFs,
        interleaved per node to match B's columns. Callers reduce the returned Voigt
        vectors to scalars (e.g. their norm) as a presentation choice; the physics is
        the full strain and stress.
        '''
        B = strain_displacement(geometry.grad_phi)
        D = self.material.constitutive_matrices(
            geometry.reference_dim, geometry.n_elements
        )
        strain = np.einsum('esk,ek->es', B, u_elements)
        stress = np.einsum('est,et->es', D, strain)
        compliance = np.einsum('es,es,e->e', stress, strain, geometry.volumes)
        return strain, stress, compliance


class EnergyDensity(Protocol):
    '''The material law an `EnergyForm` integrates: `fem.energies` implements it.'''

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        '''Derivative chain at `(n_elements, d, d)` displacement gradients.'''
        ...


@dataclass(frozen=True)
class EnergyForm:
    '''The nonlinear (hyperelastic) sibling of `Form`.

    A bilinear `Form` maps geometry to a constant matrix. An `EnergyForm` maps
    geometry *and the current nodal displacement* to three volume-weighted
    quantities, all batched over the mesh:

    - the stored energy (a scalar per element),
    - its gradient (the residual, one vector per element),
    - its Hessian (the tangent, one matrix per element).

    A quadratic energy gives a constant tangent independent of the state -- the
    linear stiffness `Form` is that special case, which is why these are siblings
    rather than one protocol taking a mostly-ignored state.

    The physics is delegated to an energy density (`fem.energies`), which
    evaluates the full derivative chain once for the whole mesh and returns a
    `StrainEnergyDerivatives` bundle -- derivatives of W, distinct from the
    derivatives of the total potential Pi that this form goes on to build. It
    contracts those against `dF_dx` (the shape-function contribution to the
    deformation gradient) to produce the assembly-ready element quantities.
    '''
    energy_density: EnergyDensity

    def _dF_dx(self, geometry: ElementGeometry) -> FloatArray:
        '''(n_el, d, d, N, d) -- dF/dx = I ⊗ grad_phiᵀ, batched.'''
        d = geometry.spatial_dim
        return np.einsum('emi,jn->eijmn', geometry.grad_phi[:, :, :d], np.eye(d))

    def element_energies(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements,) element energies at the given nodal displacements.'''
        grad_u = geometry.gradients(u_elements)
        t = self.energy_density.evaluate(grad_u)
        return t.W * geometry.volumes

    def element_residuals(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements, N, d) element residuals -- dPi/dx per element.'''
        grad_u = geometry.gradients(u_elements)
        t = self.energy_density.evaluate(grad_u)
        dF_dx = self._dF_dx(geometry)
        dW_dx = np.einsum('eij,eijmn->emn', t.dW_dF, dF_dx)
        return dW_dx * geometry.volumes[:, None, None]

    def element_tangents(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements, N, d, N, d) element tangents -- d²Pi/dx² per element.

        Reshaped to (n_elements, k, k) by the caller for scatter into the global
        matrix, where k = N * n_components.
        '''
        # d2W_dx2 = dW_dS : (d2S_dF2 : dF_dx : dF_dx) + d2W_dS2 : (dS_dx : dS_dx)
        #
        # ":" is the tensor double contraction. For two second-order tensors,
        # A : B = sum_ij A_ij B_ij -- the elementwise product summed over both
        # indices, giving a scalar. In general it contracts the last two indices
        # of the left operand against the first two of the right; each ":" above
        # is one such contraction, i.e. one "...ij,ij...->..." einsum below (with
        # a leading "e" element axis on everything that varies per element).
        grad_u = geometry.gradients(u_elements)
        t = self.energy_density.evaluate(grad_u)
        dF_dx = self._dF_dx(geometry)

        dS_dx = np.einsum('eklij,eijmn->eklmn', t.dS_dF, dF_dx)

        # term1: dW_dS : d²S_dF² : dF_dx : dF_dx
        # d2S_dF2 is constant (no element axis), broadcast over elements.
        term1 = np.einsum('abcdij,eijmn->eabcdmn', t.d2S_dF2, dF_dx)
        term1 = np.einsum('eabijcd,eijmn->eabcdmn', term1, dF_dx)
        term1 = np.einsum('eij,eijklmn->eklmn', t.dW_dS, term1)

        # term2: d²W_dS² : dS_dx : dS_dx
        # d2W_dS2 is constant (no element axis), broadcast over elements.
        term2 = np.einsum('klij,eijmn->eklmn', t.d2W_dS2, dS_dx)
        term2 = np.einsum('eijkl,eijmn->eklmn', term2, dS_dx)

        return (term1 + term2) * geometry.volumes[:, None, None, None, None]
