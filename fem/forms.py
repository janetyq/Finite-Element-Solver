"""Bilinear and nonlinear forms: the per-element matrices assembly scatters.

A `Form` builds the small dense matrices for each element. `FunctionSpace.assemble`
then scatters those element matrices into the global system matrix, so a `Form` is
the layer between element geometry and the global linear system. It answers one
thing: given this mesh, what are the per-element matrices?

Almost every element matrix follows the same pattern:

    Gᵀ C G · volume

G comes from the element's shape-function gradients, and C is the material. Keeping
them separate is the main design idea: element types stay pure geometry (they supply
G), and the `Form` supplies the physics (C). The Laplacian is G = grad_phi with
C = I; linear elasticity is G = B (strain-displacement) with C = D (the Hooke matrix).

The two families
----------------

1. Linear forms produce a constant element matrix (mass, stiffness).
2. `EnergyForm` is the nonlinear version. Its output depends on the current
   displacement, so it returns energy, residual, and tangent for a Newton solve
   (hyperelasticity).
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fem.elements import ElementGeometry
from fem.energies import StrainEnergyDerivatives
from fem.materials import LinearElasticMaterial
from fem.regions import evaluate_field
from fem.typing import BoolArray, ElementField, FieldValue, FloatArray


def strain_displacement(grad_phi: FloatArray) -> FloatArray:
    '''Voigt strain-displacement matrices B: nodal DOFs -> element strain vector.

    Operates on the trailing `(n_nodes, dim)` axes and preserves any leading batch
    axes, so it takes either `(n_elements, n_nodes, dim)` or the quadrature-aware
    `(n_elements, n_qp, n_nodes, dim)` and returns those same leading axes followed
    by `(n_strains, n_nodes*dim)`. Strain is ordered [xx, yy, (zz,) engineering
    shears] to match the rows and columns of `fem.materials.hooke_matrix`. DOFs are
    interleaved per node, so column `dim*n + d` is node n's displacement component d.
    '''
    *batch, n_nodes, dim = grad_phi.shape
    if dim == 2:
        b, c = grad_phi[..., 0], grad_phi[..., 1]
        B = np.zeros((*batch, 3, 2 * n_nodes))
        B[..., 0, 0::2] = b
        B[..., 1, 1::2] = c
        B[..., 2, 0::2] = c
        B[..., 2, 1::2] = b
        return B
    if dim == 3:
        a, b, c = grad_phi[..., 0], grad_phi[..., 1], grad_phi[..., 2]
        B = np.zeros((*batch, 6, 3 * n_nodes))
        B[..., 0, 0::3] = a
        B[..., 1, 1::3] = b
        B[..., 2, 2::3] = c
        B[..., 3, 0::3] = b
        B[..., 3, 1::3] = a
        B[..., 4, 1::3] = c
        B[..., 4, 2::3] = b
        B[..., 5, 0::3] = c
        B[..., 5, 2::3] = a
        return B
    raise NotImplementedError(
        f'no strain-displacement matrix for dim={dim}'
    )


def gradient_displacement(grad_phi: FloatArray) -> FloatArray:
    '''Gradient-displacement matrices G: nodal DOFs -> displacement-gradient vector.

    The sibling of `strain_displacement`. Where B produces the symmetric strain a
    material law contracts against, G produces the full displacement gradient
    `du_c/dx_i`. The geometric (initial-stress) stiffness contracts a prestress
    against this, since buckling is driven by the rotation part the symmetric strain
    discards.

    Operates on the trailing `(n_nodes, dim)` axes and preserves any leading batch
    axes, returning those axes followed by `(dim*dim, n_nodes*dim)`. The gradient
    vector is component-major (row `c*dim + i` is `d u_c / d x_i`), and DOFs are
    interleaved per node, so column `dim*n + e` is node n's component e. That pairing
    lines up with the block-diagonal prestress `I ⊗ σ` in `GeometricStiffnessForm`.
    '''
    *batch, n_nodes, dim = grad_phi.shape
    G = np.zeros((*batch, dim * dim, n_nodes * dim))
    for c in range(dim):       # displacement component
        for i in range(dim):   # spatial direction
            G[..., c * dim + i, c::dim] = grad_phi[..., i]
    return G


def voigt_to_tensor(voigt: FloatArray, shear_factor: float = 1.0) -> FloatArray:
    '''Unpack `(n_elements, n_strains)` Voigt vectors into `(n_elements, d, d)` tensors.

    Voigt packing stores a symmetric tensor as a vector, which makes the element
    stiffness a matrix product `B^T D B`. A norm or eigenvalue of the packed vector
    is not the tensor's, since the off-diagonal entries appear once in one and twice
    in the other.

    `shear_factor` divides the packed shear entry to recover the tensor component:
    1 for stress, 2 for strain, which packs engineering shear `gamma = 2 eps`. That
    asymmetry makes the Voigt dot product equal the tensor contraction.
    '''
    voigt = np.asarray(voigt, dtype=float)
    n_elements, n_strains = voigt.shape
    if n_strains == 3:
        d, shears = 2, [(0, 1)]
    elif n_strains == 6:
        d, shears = 3, [(0, 1), (1, 2), (0, 2)]
    else:
        raise ValueError(
            f'expected 3 (2D) or 6 (3D) Voigt components, got {n_strains}'
        )

    tensor = np.zeros((n_elements, d, d))
    for i in range(d):
        tensor[:, i, i] = voigt[:, i]
    # These pairs must match the shear rows `strain_displacement` writes above
    # (xy, then yz, then xz in 3D). The ordering is spelled out in both places
    # because B is built by direct assignment and cannot read a shared table.
    # `tests/test_invariants.py` pins the pairing; a mismatch also breaks the
    # convergence tests, so it cannot drift silently.
    for offset, (i, j) in enumerate(shears):
        tensor[:, i, j] = tensor[:, j, i] = voigt[:, d + offset] / shear_factor
    return tensor


def _with_out_of_plane(tensor: FloatArray, zz: FloatArray) -> FloatArray:
    '''Embed `(n_elements, 2, 2)` in-plane tensors into full 3x3 ones.

    A 2D solve reduces a 3D state; the third direction still carries a component,
    and which one is a property of the reduction (see `out_of_plane_stress`).
    '''
    n = len(tensor)
    full = np.zeros((n, 3, 3))
    full[:, :2, :2] = tensor
    full[:, 2, 2] = zz
    return full


@dataclass(frozen=True)
class ElasticFields:
    '''What an elastic form recovers from a solved displacement, per element.'''
    strain: FloatArray       # (n_elements, 3, 3)
    stress: FloatArray       # (n_elements, 3, 3)
    compliance: ElementField  # (n_elements,)


@runtime_checkable
class RecoversElasticFields(Protocol):
    '''A form that can recover an elastic state from a solved displacement.

    `runtime_checkable`, so a caller can branch on it. The check only tests that
    the attribute exists, not that its signature matches.
    '''

    def derived_fields(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticFields:
        '''Recover element fields from `(n_elements, N, n_components)` nodal values.

        The nested layout, matching what `FunctionSpace.assemble_residual` builds
        and what `ElementGeometry.gradients` consumes. A form wanting the flat
        interleaved `(n_elements, N*n_components)` that Voigt's B multiplies
        reshapes internally: flattening the last two axes reproduces the node-major,
        component-minor order `dof_indices` emits.
        '''
        ...


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
    '''The mass form ∫ u·v: the consistent P1 mass matrix.

    The scalar `∫ phi_i phi_j` is element geometry; a k-component field repeats it
    once per component, which is the Kronecker product with the k×k identity: DOFs
    are interleaved per node, so entry (k*a + d, k*b + e) is the scalar M[a, b]
    when d == e and zero otherwise. Used both as an operator (the mass terms of
    the time-steppers) and as a system matrix (an L2 projection solves M u = b).
    '''
    n_components: int = 1

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        if not geometry.is_affine:
            return self._curved_element_matrices(geometry)
        # The reference matrix is the same for every element of a type, so the
        # only per-element quantity is the measure it scales by.
        reference = geometry.element_type.reference_mass_matrix()
        block = np.kron(reference, np.eye(self.n_components))
        return geometry.volumes[:, None, None] * block

    def _curved_element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        '''Mass matrices integrated by quadrature, for a curved (isoparametric) element.

        `det J` varies within a curved element, so the reference-matrix-times-volume
        shortcut no longer holds; the consistent mass `int phi_i phi_j det J` is summed
        at the quadrature points instead. The n-component block is the same per-node
        interleaving as `np.kron(scalar, I)`, entry `(c*a+d, c*b+e)` carrying the scalar
        `M[a, b]` when `d == e`.
        '''
        scalar = np.einsum(
            'qi,qj,eq->eij', geometry.shape, geometry.shape, geometry.weight_detJ)
        c = self.n_components
        if c == 1:
            return scalar
        n_el, n_nodes, _ = scalar.shape
        block = np.zeros((n_el, n_nodes * c, n_nodes * c))
        for d in range(c):
            block[:, d::c, d::c] = scalar
        return block


@dataclass(frozen=True)
class MaskedMassForm:
    '''A mass form zeroed on the facets outside `mask`: a boundary mass over a
    subset of the boundary.

    Used for the Robin term ∫_∂Ω_R u·v, restricted to its region: `mask` marks the
    boundary facets that lie in the region, and the element matrices of the rest
    are zeroed before scatter, so the assembled matrix integrates over the region
    alone. `mask` is aligned with the facets `element_matrices` is called on.
    '''
    n_components: int
    mask: BoolArray  # one entry per facet

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        base = MassForm(self.n_components).element_matrices(geometry)
        return base * np.asarray(self.mask, dtype=float)[:, None, None]


@dataclass(frozen=True)
class LaplacianForm:
    '''The scalar Laplacian ∫ ∇u·∇v: material-free, so G = grad_phi, C = I.'''

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        # Sum over quadrature points q and spatial index d. For a 1-point P1 rule
        # this is the old `eid,ejd,e->eij` with a singleton q axis, identical.
        grad_phi = geometry.grad_phi
        return np.einsum('eqid,eqjd,eq->eij', grad_phi, grad_phi, geometry.weight_detJ)


def _sample_field(field: FieldValue, geometry: ElementGeometry, n_components: int) -> FloatArray:
    '''Evaluate a coefficient or source at every quadrature point: (n_el, n_qp, n_components).

    Reads a value that varies within an element at the interior points assembly
    integrates over, not only at the nodes. Flattens the (element, point) pair for
    `evaluate_field`, which works point-by-point, then restores it.
    '''
    n_el, n_qp = geometry.weight_detJ.shape
    flat = geometry.points.reshape(n_el * n_qp, geometry.spatial_dim)
    values = evaluate_field(field, flat, n_components)
    return values.reshape(n_el, n_qp, n_components)


@dataclass(frozen=True)
class DiffusionForm:
    '''Variable-coefficient diffusion ∫ κ(x) ∇u·∇v, κ sampled at the quadrature points.

    `LaplacianForm` is the κ ≡ 1 special case, kept as the cheaper constant path
    that needs no sampling. Here κ is a `FieldValue` (a constant or a callable of
    position) and scales each quadrature point's weight, so a spatially varying
    conductivity is one multiply in the sum.

    `quadrature_degree` selects the rule the space integrates this against; 2 (three
    points on a triangle) resolves a smoothly varying κ against a P1 field.
    '''
    coefficient: FieldValue
    quadrature_degree: int = 2

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        kappa = _sample_field(self.coefficient, geometry, 1)[..., 0]   # (n_el, n_qp)
        grad_phi = geometry.grad_phi
        return np.einsum(
            'eqid,eqjd,eq->eij', grad_phi, grad_phi, geometry.weight_detJ * kappa)


@dataclass(frozen=True)
class LinearForm:
    '''The linear form L(v) = ∫ f(x)·v, assembled by sampling f at the quadrature points.

    The counterpart of the bilinear `Form`: it produces one element vector per
    element, which `FunctionSpace.assemble_load` scatters into the global load.
    `problem.Source` is the cheaper special case that integrates f's P1 interpolant
    through the cached mass matrix; this samples f itself, capturing variation within
    an element the interpolant cannot.
    '''
    field: FieldValue
    n_components: int = 1
    quadrature_degree: int = 2

    def element_vectors(self, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, N*n_components) element load vectors, DOFs interleaved per node.'''
        f = _sample_field(self.field, geometry, self.n_components)   # (n_el, n_qp, c)
        # b[e, n, c] = sum_q weight_detJ[e,q] * shape[q,n] * f[e,q,c]
        b = np.einsum('eq,qn,eqc->enc', geometry.weight_detJ, geometry.shape, f)
        return b.reshape(geometry.n_elements, -1)


@dataclass(frozen=True)
class LinearElasticForm:
    '''Small-strain linear elasticity ∫ ε(u):D:ε(v), so G = B, C = D.'''
    material: LinearElasticMaterial

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        B = strain_displacement(geometry.grad_phi)   # (n_el, n_qp, n_strains, k)
        D = self.material.constitutive_matrices(
            geometry.reference_dim, geometry.n_elements
        )
        # B^T D B summed over quadrature points q and strain indices j, k, weighted
        # per point. D does not vary within an element, so it carries no q axis. For
        # a 1-point P1 rule this is the old `eji,ejk,ekl,e->eil`, identical.
        # optimize=True is load-bearing rather than cosmetic: the default
        # left-to-right order forms a large intermediate and runs far slower here.
        return np.einsum('eqji,ejk,eqkl,eq->eil', B, D, B, geometry.weight_detJ, optimize=True)

    def derived_fields(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticFields:
        '''Element strain, stress, and compliance from nodal displacements.

        `u_elements` is `(n_elements, N, n_components)`; flattening its last two
        axes gives the interleaved DOF order B's columns are written in.

        Returns full `(n_elements, 3, 3)` tensors, not the Voigt vectors assembly
        works in (see `voigt_to_tensor`); a 2D result is lifted to the 3D state
        its plane-strain assumption implies.

        Recovered per element at the first quadrature point. For P1 the strain is
        constant over the element, so any point gives its one value; a higher-order
        field varies, and reducing it to one per-element tensor is a reporting
        choice this makes explicit rather than a property of the element.
        '''
        B = strain_displacement(geometry.grad_phi[:, 0])   # (n_el, n_strains, k)
        D = self.material.constitutive_matrices(
            geometry.reference_dim, geometry.n_elements
        )
        u_flat = np.asarray(u_elements).reshape(geometry.n_elements, -1)
        strain_voigt = np.einsum('esk,ek->es', B, u_flat)
        stress_voigt = np.einsum('est,et->es', D, strain_voigt)

        strain = voigt_to_tensor(strain_voigt, shear_factor=2.0)
        stress = voigt_to_tensor(stress_voigt, shear_factor=1.0)

        if strain.shape[-1] == 2:
            # Plane strain: eps_zz is zero by definition, and the material
            # develops sigma_zz holding it there. von Mises without it is computed
            # on the wrong state.
            sigma_zz = self.material.out_of_plane_stress(strain)
            strain = _with_out_of_plane(strain, np.zeros(len(strain)))
            stress = _with_out_of_plane(stress, sigma_zz)

        # The full double contraction. eps_zz is zero under plane strain, so the
        # lift above leaves this equal to the in-plane Voigt dot product it replaces.
        compliance = np.einsum('eij,eij,e->e', stress, strain, geometry.volumes)
        return ElasticFields(strain, stress, compliance)


@dataclass(frozen=True)
class GeometricStiffnessForm:
    '''Geometric (initial-stress) stiffness ∫ Gᵀ Σ₀ G, from a per-element prestress.

    The material `C` here is not a constitutive law but the stress the structure
    already carries: `G = gradient_displacement` (the full gradient, not the
    symmetric strain `B`) and `Σ₀ = I ⊗ σ₀` block-diagonalises the prestress, so the
    quadratic form is `Σ_c (∇u_c)ᵀ σ₀ (∇u_c)`. This term stiffens a structure in
    tension and softens it in compression, to the point of buckling, where `K + λ K_g`
    loses positive-definiteness.

    It is exactly `term1` of the St-Venant–Kirchhoff consistent tangent
    (`EnergyForm.element_tangents`), where the prestress is the second Piola–Kirchhoff
    stress `dW/dS` contracted through the constant kernel `d²S/dF²`. Here the prestress
    is supplied (recovered once from a reference linear solve), so the geometric
    stiffness assembles about the undeformed configuration for a linearised buckling
    eigenproblem, without a Newton solve.

    `prestress` is the `(n_elements, d, d)` Cauchy stress in the mesh's spatial
    dimension (the in-plane 2x2 block for a 2D solve); `σ_zz` does not enter, having
    no in-plane displacement gradient to couple to.
    '''
    prestress: FloatArray   # (n_elements, d, d) prestress from the reference solve

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        d = geometry.spatial_dim
        sigma = np.asarray(self.prestress, dtype=float)
        if sigma.shape != (geometry.n_elements, d, d):
            raise ValueError(
                f'prestress must be ({geometry.n_elements}, {d}, {d}) for this mesh, '
                f'got shape {sigma.shape}'
            )
        # G maps element DOFs to the displacement-gradient vector; Σ₀ = I_d ⊗ σ₀ puts
        # the prestress on each component's diagonal block. Same Gᵀ C G contraction
        # as LinearElasticForm, with the prestress standing in for the material.
        G = gradient_displacement(geometry.grad_phi)     # (n_el, n_qp, d*d, N*d)
        Sigma = np.zeros((geometry.n_elements, d * d, d * d))
        for c in range(d):
            Sigma[:, c * d:(c + 1) * d, c * d:(c + 1) * d] = sigma
        return np.einsum(
            'eqpk,epr,eqrl,eq->ekl', G, Sigma, G, geometry.weight_detJ, optimize=True)


@dataclass(frozen=True)
class ScaledForm:
    '''A form scaled by a constant coefficient, such as c² for the wave operator.

    The wave equation's stiffness is c²K, so `problem.tangent` returns the scaled
    operator and the integrator never has to know the wave speed. Kept minimal: an
    `OperatorSum` waits for a second operator term (Robin, advection).
    '''
    factor: float
    form: Form

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        return self.factor * self.form.element_matrices(geometry)


@dataclass(frozen=True, eq=False)
class PrecomputedForm:
    '''Element matrices computed elsewhere, handed to assembly as they are.

    An escape hatch for a driver that can derive its element matrices more cheaply
    than by re-integrating them. SIMP is the case in hand: since the constitutive
    matrix is linear in E, scaling the modulus by `rho^p` scales each element matrix
    by `rho^p`, so a topology optimization iteration rescales one precomputed set
    rather than re-contracting `B^T D B` over the mesh.

    Valid only for the geometry the matrices were computed on. The element count is
    checked, the one mismatch a bare `matrices` array can reveal.
    '''
    matrices: FloatArray   # (n_elements, k, k)

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        if len(self.matrices) != geometry.n_elements:
            raise ValueError(
                f'precomputed matrices cover {len(self.matrices)} elements but the '
                f'geometry has {geometry.n_elements}'
            )
        return self.matrices


class EnergyDensity(Protocol):
    '''The material law an `EnergyForm` integrates: `fem.energies` implements it.'''

    def evaluate(self, grad_u: FloatArray) -> StrainEnergyDerivatives:
        '''Derivative chain at `(n_elements, d, d)` displacement gradients.'''
        ...

    def strain(self, grad_u: FloatArray) -> FloatArray:
        '''The strain measure this density is built on, at those gradients.'''
        ...

    def out_of_plane_stress(self, strain: FloatArray) -> FloatArray:
        '''Second Piola-Kirchhoff `S_zz` for a 2D (plane-strain) reduction.'''
        ...


@dataclass(frozen=True)
class EnergyForm:
    '''The nonlinear (hyperelastic) sibling of `Form`.

    A bilinear `Form` maps geometry to a constant matrix. An `EnergyForm` maps
    geometry and the current nodal displacement to three volume-weighted quantities,
    all batched over the mesh:

    - the stored energy (a scalar per element),
    - its gradient (the residual, one vector per element),
    - its Hessian (the tangent, one matrix per element).

    A quadratic energy gives a constant tangent independent of the state. The linear
    stiffness `Form` is that special case, which is why these are siblings.

    The physics is delegated to an energy density (`fem.energies`), which evaluates
    the full derivative chain once for the whole mesh and returns a
    `StrainEnergyDerivatives` bundle (derivatives of W, distinct from those of the
    total potential Pi this form builds). It contracts those against `dF_dx` (the
    shape-function contribution to the deformation gradient) to produce the
    assembly-ready element quantities.
    '''
    energy_density: EnergyDensity

    def _dF_dx(self, geometry: ElementGeometry, q: int) -> FloatArray:
        '''(n_el, d, d, N, d): dF/dx at quadrature point q = I ⊗ grad_phi_qᵀ.'''
        d = geometry.spatial_dim
        return np.einsum('emi,jn->eijmn', geometry.grad_phi[:, q, :, :d], np.eye(d))

    def element_energies(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements,) element energies at the given nodal displacements.

        Summed over the quadrature points, each contributing its density evaluated
        at that point weighted by `weight_detJ`. One iteration for a 1-point P1
        rule. The density's derivative chain is reused across orders unchanged.
        '''
        grad_u = geometry.gradients(u_elements)   # (n_el, n_qp, d, d)
        total = np.zeros(geometry.n_elements)
        for q in range(geometry.n_qp):
            t = self.energy_density.evaluate(grad_u[:, q])
            total += t.W * geometry.weight_detJ[:, q]
        return total

    def element_residuals(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements, N, d) element residuals: dPi/dx per element.'''
        grad_u = geometry.gradients(u_elements)
        n_nodes, d = geometry.grad_phi.shape[2], geometry.spatial_dim
        residual = np.zeros((geometry.n_elements, n_nodes, d))
        for q in range(geometry.n_qp):
            t = self.energy_density.evaluate(grad_u[:, q])
            dF_dx = self._dF_dx(geometry, q)
            dW_dx = np.einsum('eij,eijmn->emn', t.dW_dF, dF_dx)
            residual += dW_dx * geometry.weight_detJ[:, q][:, None, None]
        return residual

    def element_tangents(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements, N, d, N, d) element tangents: d²Pi/dx² per element.

        Reshaped to (n_elements, k, k) by the caller for scatter into the global
        matrix, where k = N * n_components.
        '''
        # d2W_dx2 = dW_dS : (d2S_dF2 : dF_dx : dF_dx) + d2W_dS2 : (dS_dx : dS_dx)
        #
        # ":" is the tensor double contraction. For two second-order tensors,
        # A : B = sum_ij A_ij B_ij, the elementwise product summed over both
        # indices, giving a scalar. In general it contracts the last two indices
        # of the left operand against the first two of the right; each ":" above
        # is one such contraction, i.e. one "...ij,ij...->..." einsum below (with
        # a leading "e" element axis on everything that varies per element).
        #
        # Summed over the quadrature points, each evaluated at its own dF_dx and
        # weighted by `weight_detJ`. The per-point contraction is exactly the old
        # single-pass one, so a 1-point P1 rule reproduces it term for term.
        grad_u = geometry.gradients(u_elements)
        n_nodes, d = geometry.grad_phi.shape[2], geometry.spatial_dim
        tangent = np.zeros((geometry.n_elements, n_nodes, d, n_nodes, d))
        for q in range(geometry.n_qp):
            t = self.energy_density.evaluate(grad_u[:, q])
            dF_dx = self._dF_dx(geometry, q)

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

            tangent += (term1 + term2) * geometry.weight_detJ[:, q][:, None, None, None, None]
        return tangent

    def derived_fields(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticFields:
        '''Element strain, Cauchy stress, and compliance at a solved displacement.

        Stress is **Cauchy**, `sigma = J^-1 P F^T`, not the first Piola-Kirchhoff
        `P = dW_dF` the energy derivative gives: P is measured per unit undeformed
        area, so it is not comparable with the small-strain path's stress. The two
        agree to O(||grad u||). Strain is the density's own measure.

        Reconciles two conventions from `fem.energies`: the gradient orientation it
        works in and the plane-strain reduction a 2D solve makes. Both are explained
        there under "Solving versus reporting".

        Recovered per element at the first quadrature point, matching the linear
        path: constant over the element for P1, one representative value otherwise.
        '''
        grad_u = geometry.gradients(u_elements)[:, 0]   # (n_el, d, d), first quad point
        t = self.energy_density.evaluate(grad_u)
        d = grad_u.shape[-1]

        # Put F and dW_dF into the standard orientation before anything contracts
        # them; fem.energies works in the transposed one, which the energy cannot
        # tell apart but a reported tensor can.
        F = np.eye(d) + np.swapaxes(grad_u, -2, -1)
        P = np.swapaxes(t.dW_dF, -2, -1)

        J = np.linalg.det(F)
        cauchy = np.einsum('e,eij,ekj->eik', 1.0 / J, P, F)

        # The density's own measure (Green-Lagrange for St-VK, eps for its
        # linearisation) is asked for rather than branched on here, so the class
        # that owns the choice is the one that answers.
        strain = self.energy_density.strain(grad_u)

        if d == 2:
            # Restore the stress in the restrained direction, which the 2D Voigt
            # vector omits. Without it the nonlinear path would report a different
            # von Mises than the linear one for the same material.
            sigma_zz = self.energy_density.out_of_plane_stress(strain) / J
            strain = _with_out_of_plane(strain, np.zeros(len(strain)))
            cauchy = _with_out_of_plane(cauchy, sigma_zz)

        # Twice the stored energy, which for a quadratic W is exactly S:E, the
        # work-conjugate pair. Contracting the *reported* Cauchy stress with E
        # instead mixes measures and runs ~30% wrong at finite strain. Going
        # through W also avoids having to pick an orientation.
        compliance = 2.0 * t.W * geometry.volumes
        return ElasticFields(strain, cauchy, compliance)
