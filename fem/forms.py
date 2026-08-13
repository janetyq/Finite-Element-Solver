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


def voigt_to_tensor(voigt: FloatArray, shear_factor: float = 1.0) -> FloatArray:
    '''Unpack `(n_elements, n_strains)` Voigt vectors into `(n_elements, d, d)` tensors.

    Voigt packing stores a symmetric tensor as a vector, which is what makes the
    element stiffness a matrix product `B^T D B`. Nothing above assembly should
    have to know it: a norm or an eigenvalue of the packed vector is not the
    tensor's, since the off-diagonal entries appear once in one and twice in the
    other.

    `shear_factor` divides the packed shear entry to recover the tensor
    component: 1 for stress, 2 for strain, which packs engineering shear
    `gamma = 2 eps`. That asymmetry is what makes the Voigt dot product equal the
    tensor contraction.
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
    # These pairs must match the shear rows `strain_displacement` writes above --
    # xy, then yz, then xz in 3D -- and the ordering is spelled out in both places
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
        reshapes internally -- flattening the last two axes reproduces exactly the
        node-major, component-minor order `dof_indices` emits.
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
class MaskedMassForm:
    '''A mass form zeroed on the facets outside `mask` -- a boundary mass over a
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
    '''The scalar Laplacian ∫ ∇u·∇v -- material-free, so G = grad_phi, C = I.'''

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        # Sum over quadrature points q and spatial index d. For a 1-point P1 rule
        # this is the old `eid,ejd,e->eij` with a singleton q axis -- identical.
        grad_phi = geometry.grad_phi
        return np.einsum('eqid,eqjd,eq->eij', grad_phi, grad_phi, geometry.weight_detJ)


def _sample_field(field: FieldValue, geometry: ElementGeometry, n_components: int) -> FloatArray:
    '''Evaluate a coefficient or source at every quadrature point: (n_el, n_qp, n_components).

    The point of a real quadrature layer: a value that varies *within* an element,
    read at the interior points assembly integrates over rather than only at the
    nodes. Flattens the (element, point) pair for `evaluate_field`, which works
    point-by-point, then restores it.
    '''
    n_el, n_qp = geometry.weight_detJ.shape
    flat = geometry.points.reshape(n_el * n_qp, geometry.spatial_dim)
    values = evaluate_field(field, flat, n_components)
    return values.reshape(n_el, n_qp, n_components)


@dataclass(frozen=True)
class DiffusionForm:
    '''Variable-coefficient diffusion ∫ κ(x) ∇u·∇v, κ sampled at the quadrature points.

    `LaplacianForm` is the κ ≡ 1 special case, kept as the cheaper constant path
    that needs no sampling and no interior points. Here κ is a `FieldValue` -- a
    constant or a callable of position -- and the only change from the Laplacian is
    that it scales each point's weight, so a spatially varying conductivity, a
    material property that changes across the domain, is one multiply in the sum.

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

    The counterpart of the bilinear `Form`: it produces one element *vector* per
    element, which `FunctionSpace.assemble_load` scatters into the global load.
    `problem.Source` is the cheaper special case that integrates f's P1 interpolant
    through the cached mass matrix; this samples f itself, so it captures variation
    within an element the interpolant cannot -- the load half of the quadrature layer.
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
        # a 1-point P1 rule this is the old `eji,ejk,ekl,e->eil` -- identical.
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
class ScaledForm:
    '''A form scaled by a constant coefficient -- c² for the wave operator.

    The first operator-side combinator, and it earns its place exactly where the
    composition-algebra design said it would: the wave equation's stiffness is
    c²K, so `problem.tangent` returns the scaled operator and the integrator never
    has to know the wave speed. Kept minimal on purpose -- an `OperatorSum` waits
    for a second operator term (Robin, advection).
    '''
    factor: float
    form: Form

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        return self.factor * self.form.element_matrices(geometry)


@dataclass(frozen=True, eq=False)
class PrecomputedForm:
    '''Element matrices computed elsewhere, handed to assembly as they are.

    The escape hatch for a driver that can derive its element matrices more cheaply
    than by re-integrating them. SIMP is the case in hand: scaling the modulus by
    `rho^p` scales each element matrix by exactly `rho^p`, since the constitutive
    matrix is linear in E, so a topology optimization iteration rescales one
    precomputed set rather than re-contracting `B^T D B` over the mesh.

    Valid only for the geometry the matrices were computed on, which is why the
    element count is checked -- `matrices` carries the geometry's imprint but no
    way to identify it, so a mismatched *shape* is the one error catchable here.
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

    def _dF_dx(self, geometry: ElementGeometry, q: int) -> FloatArray:
        '''(n_el, d, d, N, d) -- dF/dx at quadrature point q = I ⊗ grad_phi_qᵀ.'''
        d = geometry.spatial_dim
        return np.einsum('emi,jn->eijmn', geometry.grad_phi[:, q, :, :d], np.eye(d))

    def element_energies(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements,) element energies at the given nodal displacements.

        Summed over the quadrature points, each contributing its density evaluated
        at that point weighted by `weight_detJ`. One iteration for a 1-point P1
        rule -- the density's derivative chain is reused across orders unchanged.
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
        '''(n_elements, N, d) element residuals -- dPi/dx per element.'''
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

        Reconciles two conventions from `fem.energies` -- the gradient orientation
        it works in and the plane-strain reduction a 2D solve makes -- both
        explained there under "Solving versus reporting".

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

        # The density's own measure -- Green-Lagrange for St-VK, eps for its
        # linearisation -- asked for rather than branched on here, so the class
        # that owns the choice is the one that answers.
        strain = self.energy_density.strain(grad_u)

        if d == 2:
            # Restore the stress in the restrained direction, which the 2D Voigt
            # vector omits. Without it the nonlinear path would report a different
            # von Mises than the linear one for the same material.
            sigma_zz = self.energy_density.out_of_plane_stress(strain) / J
            strain = _with_out_of_plane(strain, np.zeros(len(strain)))
            cauchy = _with_out_of_plane(cauchy, sigma_zz)

        # Twice the stored energy, which for a quadratic W is exactly S:E -- the
        # work-conjugate pair. Contracting the *reported* Cauchy stress with E
        # instead mixes measures and runs ~30% wrong at finite strain. Going
        # through W also avoids having to pick an orientation.
        compliance = 2.0 * t.W * geometry.volumes
        return ElasticFields(strain, cauchy, compliance)
