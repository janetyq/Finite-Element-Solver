"""Forms: the per-element residuals and tangents assembly scatters.

A `Form` maps element geometry and a nodal state to the small dense blocks of each
element: the residual (internal force) and its tangent. `FunctionSpace` scatters those
into the global vector and matrix, so a `Form` is the layer between element geometry
and the global system. It answers one thing: given this mesh and this state, what are
the per-element blocks?

Almost every tangent follows the same pattern:

    Gᵀ C G · volume

G comes from the element's shape-function gradients, and C is the material. Keeping
them separate is the main design idea: element types stay pure geometry (they supply
G), and the `Form` supplies the physics (C). The Laplacian is G = grad_phi with
C = I; linear elasticity is G = B (strain-displacement) with C = D (the Hooke matrix).

Two ways to write one: a `BilinearForm` writes the constant matrix `K` (mass,
stiffness) and gets residual `K u`, tangent `K`, energy `½ uᵀ K u`; an `EnergyForm`
writes a stored-energy density and gets all three by differentiating it at the state.
What else a form can answer (a constant tangent, an energy, a recoverable flux, an AMG
near-kernel) is a hook on `Form` with a default answer of "no".

Forms compose: `a + b` is a `SumForm` and `c * a` a `ScaledForm`, each answering the hooks
from its terms. A form names its integration `domain` (the volume elements or the boundary
facets), so a sum may mix the two, as an operator with a Robin boundary term does, and the
space assembles each term over its own domain.
"""
from dataclasses import dataclass
from numbers import Real
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

import numpy as np

from fem.elements import ElementGeometry
from fem.physics.energies import StrainEnergyDerivatives
from fem.physics.materials import LinearElasticMaterial
from fem.physics.derived import DerivedField, GradientField, StressField
from fem.post.solution import ElasticSolution, FieldSolution, ScalarFieldSolution
from fem.regions import evaluate_field
from fem.typing import BoolArray, ElementField, FieldValue, FloatArray, Vertices

if TYPE_CHECKING:
    from fem.space import FunctionSpace

# The typed solution a form packages (`solution`): `ElasticSolution` for a form that
# recovers stress, `ScalarFieldSolution` for one naming a flux, else `FieldSolution`.
# It flows up through `Problem[S]` to `Problem.solve() -> S`.
S = TypeVar('S', bound=FieldSolution)


def strain_displacement(grad_phi: FloatArray) -> FloatArray:
    '''Voigt strain-displacement matrices B: nodal DOFs -> element strain vector.

    Operates on the trailing `(n_nodes, dim)` axes and preserves any leading batch
    axes, so it takes either `(n_elements, n_nodes, dim)` or the quadrature-aware
    `(n_elements, n_qp, n_nodes, dim)` and returns those same leading axes followed
    by `(n_strains, n_nodes*dim)`. Strain is ordered [xx, yy, (zz,) engineering
    shears] to match the rows and columns of `fem.physics.materials.hooke_matrix`. DOFs are
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
    # These pairs must match the shear rows `strain_displacement` writes (xy, then
    # yz, then xz in 3D); `tests/test_invariants.py` pins the pairing.
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


@dataclass(frozen=True)
class ElasticPointFields:
    '''Strain and stress at every point of a geometry's rule, full 3x3 tensors.'''
    strain: FloatArray       # (n_elements, n_qp, 3, 3)
    stress: FloatArray       # (n_elements, n_qp, 3, 3)


@runtime_checkable
class RecoversElasticFields(Protocol):
    '''A form that can recover an elastic state from a solved displacement.

    `runtime_checkable`, so a caller can branch on it. The check only tests that
    the attribute exists, not that its signature matches.
    '''

    def sample(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticPointFields:
        '''Strain and stress at every point of `geometry`'s rule.

        `u_elements` is `(n_elements, N, n_components)`, matching what
        `FunctionSpace.assemble_residual` builds and what `ElementGeometry.gradients`
        consumes. A form wanting the flat interleaved `(n_elements, N*n_components)`
        that Voigt's B multiplies reshapes internally: flattening the last two axes
        reproduces the node-major, component-minor order `dof_indices` emits.

        The points are whatever the geometry was built at: a quadrature rule for an
        integral or a projection, or the element's own nodes (`nodal_rule`) for a
        nodal reading.
        '''
        ...

    def derived_fields(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticFields:
        '''One strain, stress, and compliance per element, from `(n_elements, N,
        n_components)` nodal values: the element mean of `sample` over the rule.'''
        ...


def _element_mean(values: FloatArray, weight_detJ: FloatArray) -> FloatArray:
    '''Reduce `(n_elements, n_qp, ...)` point values to their volume-weighted element mean.

    Exact for P1 (the field is constant) and the centroid value of a field linear
    within the element, which a straight P2 stress is.
    '''
    weights = weight_detJ / weight_detJ.sum(axis=1, keepdims=True)
    return np.einsum('eq,eq...->e...', weights, values)


class Form(ABC, Generic[S]):
    '''Element residual and tangent at a nodal state: what a `Problem` assembles.

    A subclass writes `element_residuals` and `element_tangents`. Everything else is
    a question a `Problem` may ask, answered "no" by default:

    - `constant_tangent`: the tangent is one matrix, so a consumer may assemble it
      once (`LinearSolve`, the integrators, an eigenproblem, SIMP). `BilinearForm`.
    - `has_energy`: `element_energies` is defined and the residual is its gradient.
      Read only by the line search, which then scores a step by the energy instead
      of ½‖r‖²; the Newton iteration is the same either way.
    - `derived_field`: the flux post-processing recovers and estimators jump
      (Poisson's gradient, elasticity's stress).
    - `near_null_space`: the AMG near-kernel an iterative solve of the tangent is
      built with (the rigid-body modes of elasticity).

    `domain` is where the form integrates: `'volume'` over the elements (the default) or
    `'boundary'` over the boundary facets. `terms` is the flat tuple of forms a sum is
    made of, `(self,)` for a form that is not a sum; assembly iterates it so every term
    integrates over its own domain.

    `u_elements` is `(n_elements, N, n_components)`, each element's slice of the state.
    Batched over the mesh: a Python loop over elements spends nearly all its time in
    per-call numpy overhead, and one vectorized pass is roughly 30x faster.
    '''
    constant_tangent: bool = False
    has_energy: bool = False
    domain: ClassVar[Literal['volume', 'boundary']] = 'volume'

    @property
    def terms(self) -> tuple['Form[Any]', ...]:
        return (self,)

    def __add__(self, other: 'Form[Any]') -> 'Form[S]':
        # A sum packages through its physics term, and a boundary term added to a
        # physics form is the common sum, so the left operand's solution type is kept.
        if not isinstance(other, Form):
            return NotImplemented
        return SumForm(self.terms + other.terms)

    def __mul__(self, factor: float) -> 'Form[S]':
        if not isinstance(factor, Real):
            return NotImplemented
        scaled = tuple(ScaledForm(float(factor), term) for term in self.terms)
        return scaled[0] if len(scaled) == 1 else SumForm(scaled)

    __rmul__ = __mul__

    def quadrature_degree(self, shape_degree: int) -> int:
        '''The lowest rule degree that integrates this form on a degree-`shape_degree`
        element; the space uses the larger of it and the element's default.'''
        return 0

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, k, k) constant element matrices; defined when `constant_tangent`.'''
        raise TypeError(f'{type(self).__name__} has a state-dependent tangent and no constant matrices')

    @abstractmethod
    def element_residuals(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        '''(n_elements, k) internal-force blocks at the state, k = N * n_components.'''

    @abstractmethod
    def element_tangents(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        '''(n_elements, k, k) tangent blocks at the state.'''

    def element_energies(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        '''(n_elements,) stored energy per element at the state; defined when `has_energy`.'''
        raise NotImplementedError(f'{type(self).__name__} has no energy')

    def derived_field(self) -> DerivedField | None:
        return None

    def near_null_space(self, space: 'FunctionSpace') -> FloatArray | None:
        '''`(n_dofs, n_modes)` over every DOF of `space`; the solve restricts it to the free block.'''
        return None

    def solution(self, space: 'FunctionSpace', u: FloatArray) -> S:
        '''Package a solved DOF vector as the typed `Solution` this physics recovers:
        a bare `FieldSolution` unless a subclass recovers more (and then says so in
        its `S`; a form that keeps this default is a `Form[FieldSolution]`).'''
        return cast(S, FieldSolution(space, u))


class BilinearForm(Form[S]):
    '''A form with a constant element matrix K: residual K u, tangent K, energy ½ uᵀKu.

    Subclasses write `element_matrices`; the rest follows from it. The state is
    flattened node-major, component-minor, the DOF order `dof_indices` emits and the
    matrix's rows use.
    '''
    constant_tangent = True
    has_energy = True

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        '''(n_elements, k, k) dense element matrices for every element at once.'''
        raise NotImplementedError

    def element_residuals(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        K = self.element_matrices(geometry)
        u = np.asarray(u_elements).reshape(len(K), -1)
        return np.einsum('eij,ej->ei', K, u)

    def element_tangents(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        return self.element_matrices(geometry)

    def element_energies(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        K = self.element_matrices(geometry)
        u = np.asarray(u_elements).reshape(len(K), -1)
        return 0.5 * np.einsum('ei,eij,ej->e', u, K, u)


@dataclass(frozen=True)
class MassForm(BilinearForm[FieldSolution]):
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


@dataclass(frozen=True, eq=False)
class BoundaryMassForm(BilinearForm[FieldSolution]):
    '''The boundary mass form ∫_Γ u·v over the subset Γ of the boundary that `mask`
    marks.

    The Robin operator term κ ∫_Γ u·v is `kappa * BoundaryMassForm(n, mask)`, and the
    Neumann and Robin loads integrate through the same matrix. `mask` marks the
    boundary facets that lie in the region, and the element matrices of the rest are
    zeroed before scatter, so the assembled matrix integrates over the region alone.
    `mask` is aligned with the space's boundary facets.
    '''
    n_components: int
    mask: BoolArray  # one entry per facet
    domain: ClassVar[Literal['volume', 'boundary']] = 'boundary'

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        base = MassForm(self.n_components).element_matrices(geometry)
        return base * np.asarray(self.mask, dtype=float)[:, None, None]


def sample_field(field: FieldValue, geometry: ElementGeometry, n_components: int) -> FloatArray:
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
class DiffusionForm(BilinearForm[ScalarFieldSolution]):
    '''The diffusion form ∫ κ ∇u·∇v: the Laplacian at κ ≡ 1, the operator of Poisson,
    heat, and (with κ = c²) the wave equation.

    `coefficient` is κ, a constant or a callable of position. A constant scales the
    element's own rule (one point on P1); a callable is sampled at the quadrature
    points of a rule of `rule_degree` (2, three points on a triangle, resolves a
    smoothly varying κ against a P1 field), so a spatially varying conductivity is one
    multiply per point in the sum.
    '''
    coefficient: FieldValue = 1.0
    rule_degree: int = 2

    @property
    def is_sampled(self) -> bool:
        return callable(self.coefficient)

    def quadrature_degree(self, shape_degree: int) -> int:
        return self.rule_degree if self.is_sampled else 0

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        grad_phi = geometry.grad_phi
        if not self.is_sampled:
            kappa = float(np.asarray(self.coefficient, dtype=float).reshape(-1)[0])
            return kappa * np.einsum('eqid,eqjd,eq->eij', grad_phi, grad_phi, geometry.weight_detJ)
        kappa = sample_field(self.coefficient, geometry, 1)[..., 0]   # (n_el, n_qp)
        return np.einsum(
            'eqid,eqjd,eq->eij', grad_phi, grad_phi, geometry.weight_detJ * kappa)

    def derived_field(self) -> DerivedField:
        return GradientField()

    def solution(self, space: 'FunctionSpace', u: FloatArray) -> 'ScalarFieldSolution':
        return ScalarFieldSolution.from_solve(space, u)


def rigid_body_modes(vertices: Vertices, n_components: int) -> FloatArray:
    '''The rigid-body modes of an elastic body: the AMG near-kernel for elasticity.

    Rigid translations and (infinitesimal) rotations produce no strain, so they lie
    in the kernel of the unconstrained stiffness: the low-energy modes a plain
    smoother cannot damp and the coarse levels must represent. Feeding them to AMG
    keeps CG's iteration count flat under mesh refinement for a lightly constrained
    body; the constant vector pyamg assumes by default does not.

    Returns `(n_dofs, n_modes)` in the interleaved DOF order (component `d` of node
    `v` at `n_components*v + d`): 3 modes in 2D (two translations, one rotation), 6
    in 3D (three of each). Restrict the rows to the free DOFs before use, so the
    block matches the one `IterativeBackend` is handed.
    '''
    n = len(vertices)
    if n_components == 2:
        x, y = vertices[:, 0], vertices[:, 1]
        B = np.zeros((2 * n, 3))
        B[0::2, 0] = 1.0                      # translate x
        B[1::2, 1] = 1.0                      # translate y
        B[0::2, 2], B[1::2, 2] = -y, x        # rotate in-plane: (-y, x)
        return B
    if n_components == 3:
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        B = np.zeros((3 * n, 6))
        B[0::3, 0] = B[1::3, 1] = B[2::3, 2] = 1.0   # three translations
        B[1::3, 3], B[2::3, 3] = -z, y               # rotate about x: (0, -z, y)
        B[0::3, 4], B[2::3, 4] = z, -x               # rotate about y: (z, 0, -x)
        B[0::3, 5], B[1::3, 5] = -y, x               # rotate about z: (-y, x, 0)
        return B
    raise ValueError(f'rigid-body modes are defined for 2D or 3D elasticity, not n_components={n_components}')


@dataclass(frozen=True)
class LinearElasticForm(BilinearForm[ElasticSolution]):
    '''Small-strain linear elasticity ∫ ε(u):D:ε(v), so G = B, C = D.'''
    material: LinearElasticMaterial

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        B = strain_displacement(geometry.grad_phi)   # (n_el, n_qp, n_strains, k)
        D = self.material.constitutive_matrices(
            geometry.reference_dim, geometry.n_elements
        )
        # B^T D B summed over quadrature points q and strain indices j, k, weighted
        # per point. D does not vary within an element, so it carries no q axis.
        # optimize=True matters: the default left-to-right order forms a large
        # intermediate and runs far slower.
        return np.einsum('eqji,ejk,eqkl,eq->eil', B, D, B, geometry.weight_detJ, optimize=True)

    def derived_field(self) -> DerivedField:
        return StressField(self)

    def solution(self, space: 'FunctionSpace', u: FloatArray) -> 'ElasticSolution':
        return ElasticSolution.from_solve(space, u, self)

    def near_null_space(self, space: 'FunctionSpace') -> FloatArray:
        # Built at the space's node coordinates: a P2 space has edge nodes of its own,
        # and AMG wants the modes at every DOF.
        return rigid_body_modes(space.node_coords, space.n_components)

    def sample(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticPointFields:
        '''Strain and stress at every point of `geometry`'s rule.

        Full `(n_elements, n_qp, 3, 3)` tensors, not the Voigt vectors assembly works
        in (see `voigt_to_tensor`); a 2D result is lifted to the 3D state its
        plane-strain assumption implies. Constant across the points for P1, linear
        within the element for a straight P2 triangle.
        '''
        B = strain_displacement(geometry.grad_phi)          # (n_el, n_qp, n_strains, k)
        D = self.material.constitutive_matrices(
            geometry.reference_dim, geometry.n_elements
        )
        u_flat = np.asarray(u_elements).reshape(geometry.n_elements, -1)
        strain_voigt = np.einsum('eqsk,ek->eqs', B, u_flat)
        stress_voigt = np.einsum('est,eqt->eqs', D, strain_voigt)   # (n_el, n_qp, n_strains)

        n_el, n_qp = stress_voigt.shape[:2]
        strain = voigt_to_tensor(strain_voigt.reshape(n_el * n_qp, -1), shear_factor=2.0)
        stress = voigt_to_tensor(stress_voigt.reshape(n_el * n_qp, -1), shear_factor=1.0)

        if strain.shape[-1] == 2:
            # Plane strain: eps_zz is zero by definition, and the material
            # develops sigma_zz holding it there. von Mises without it is computed
            # on the wrong state.
            sigma_zz = self.material.out_of_plane_stress(strain)
            strain = _with_out_of_plane(strain, np.zeros(len(strain)))
            stress = _with_out_of_plane(stress, sigma_zz)

        return ElasticPointFields(strain.reshape(n_el, n_qp, 3, 3),
                                  stress.reshape(n_el, n_qp, 3, 3))

    def derived_fields(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticFields:
        '''Element strain, stress, and compliance from nodal displacements.

        Strain and stress are the element mean of `sample` over the rule: the
        exact value for P1 and the centroid value for a straight P2 triangle.
        Compliance is `∫ sigma : eps` over the element.
        '''
        fields = self.sample(geometry, u_elements)
        # The full double contraction. eps_zz is zero under plane strain, so the
        # lift above leaves this equal to the in-plane Voigt dot product it replaces.
        compliance = np.einsum('eqij,eqij,eq->e', fields.stress, fields.strain,
                               geometry.weight_detJ)
        return ElasticFields(_element_mean(fields.strain, geometry.weight_detJ),
                             _element_mean(fields.stress, geometry.weight_detJ),
                             compliance)

@dataclass(frozen=True, eq=False)
class GeometricStiffnessForm(BilinearForm[FieldSolution]):
    '''Geometric (initial-stress) stiffness ∫ Gᵀ Σ₀ G, from a per-element prestress.

    The material `C` here is not a constitutive law but the stress the structure
    already carries: `G = gradient_displacement` (the full gradient, not the
    symmetric strain `B`) and `Σ₀ = I ⊗ σ₀` block-diagonalises the prestress, so the
    quadratic form is `Σ_c (∇u_c)ᵀ σ₀ (∇u_c)`. This term stiffens a structure in
    tension and softens it in compression, to the point of buckling, where `K + λ K_g`
    loses positive-definiteness.

    It is `term1` of the St-Venant–Kirchhoff consistent tangent
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
class ScaledForm(Form[S]):
    '''A form scaled by a constant: `factor * form`, such as c² times the Laplacian for
    the wave operator, or κ times a boundary mass for a Robin term.

    Every hook is the wrapped form's; the energy, residual, and tangent are scaled. A
    sum is never wrapped: `factor * (a + b)` distributes into a sum of scaled terms.
    '''
    factor: float
    form: Form[S]

    def __post_init__(self) -> None:
        if len(self.form.terms) > 1:
            raise TypeError('scale the terms of a sum, not the sum: write factor * form')

    @property
    def constant_tangent(self) -> bool:
        return self.form.constant_tangent

    @property
    def has_energy(self) -> bool:
        return self.form.has_energy

    @property
    def domain(self) -> Literal['volume', 'boundary']:
        return self.form.domain

    def quadrature_degree(self, shape_degree: int) -> int:
        return self.form.quadrature_degree(shape_degree)

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        return self.factor * self.form.element_matrices(geometry)

    def element_residuals(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        return self.factor * self.form.element_residuals(geometry, u_elements)

    def element_tangents(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        return self.factor * self.form.element_tangents(geometry, u_elements)

    def element_energies(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        return self.factor * self.form.element_energies(geometry, u_elements)

    def derived_field(self) -> DerivedField | None:
        return self.form.derived_field()

    def near_null_space(self, space: 'FunctionSpace') -> FloatArray | None:
        return self.form.near_null_space(space)

    def solution(self, space: 'FunctionSpace', u: FloatArray) -> S:
        return self.form.solution(space, u)


@dataclass(frozen=True)
class SumForm(Form[S]):
    '''The sum of forms, each integrating over its own domain: `a + b`.

    The tangent is constant when every term's is, and the energy exists when every
    term has one. The derived field, the near-null space, and the solution packaging
    come from the one term that answers (the physics term; a boundary mass answers
    none), and two answering terms is an error. A sum has no element blocks of its
    own: `FunctionSpace` assembles each of `terms` and adds the results.
    '''
    forms: tuple[Form[Any], ...]

    def __post_init__(self) -> None:
        if len(self.forms) < 2:
            raise ValueError('a SumForm needs at least two terms')
        if any(len(form.terms) > 1 for form in self.forms):
            raise ValueError('a SumForm is flat; build it with a + b')

    @property
    def constant_tangent(self) -> bool:
        return all(f.constant_tangent for f in self.forms)

    @property
    def has_energy(self) -> bool:
        return all(f.has_energy for f in self.forms)

    @property
    def terms(self) -> tuple[Form[Any], ...]:
        return self.forms

    def _physics_term(self) -> Form[Any] | None:
        '''The one term that names a derived field, or None.'''
        physics = [term for term in self.terms if term.derived_field() is not None]
        if len(physics) > 1:
            names = ', '.join(type(term).__name__ for term in physics)
            raise ValueError(f'more than one term of the sum names a derived field: {names}')
        return physics[0] if physics else None

    def derived_field(self) -> DerivedField | None:
        physics = self._physics_term()
        return None if physics is None else physics.derived_field()

    def near_null_space(self, space: 'FunctionSpace') -> FloatArray | None:
        modes = [m for m in (term.near_null_space(space) for term in self.terms) if m is not None]
        if len(modes) > 1:
            raise ValueError('more than one term of the sum names a near-null space')
        return modes[0] if modes else None

    def solution(self, space: 'FunctionSpace', u: FloatArray) -> S:
        physics = self._physics_term()
        return super().solution(space, u) if physics is None else physics.solution(space, u)

    def element_matrices(self, geometry: ElementGeometry) -> FloatArray:
        raise TypeError(_NO_BLOCKS)

    def element_residuals(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        raise TypeError(_NO_BLOCKS)

    def element_tangents(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        raise TypeError(_NO_BLOCKS)

    def element_energies(self, geometry: ElementGeometry, u_elements: FloatArray) -> FloatArray:
        raise TypeError(_NO_BLOCKS)


_NO_BLOCKS = 'a SumForm has no element blocks of its own; assemble its terms'


@dataclass(frozen=True, eq=False)
class PrecomputedForm(BilinearForm[FieldSolution]):
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
    '''The material law an `EnergyForm` integrates: `fem.physics.energies` implements it.'''

    energy_degree: int
    '''Polynomial degree of W in the displacement gradient, setting the quadrature rule.'''

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
class EnergyForm(Form[ElasticSolution]):
    '''A hyperelastic `Form`: the energy, residual, and tangent of a stored-energy density.

    Maps geometry and the current nodal displacement to three volume-weighted
    quantities, all batched over the mesh:

    - the stored energy (a scalar per element),
    - its gradient (the residual, one vector per element),
    - its Hessian (the tangent, one matrix per element).

    A quadratic energy gives a constant tangent independent of the state; the
    `LinearElasticForm` is that special case as a `BilinearForm`.

    The physics is delegated to an energy density (`fem.physics.energies`), which evaluates
    the full derivative chain once for the whole mesh and returns a
    `StrainEnergyDerivatives` bundle (derivatives of W, distinct from those of the
    total potential Pi this form builds). It contracts those against `dF_dx` (the
    shape-function contribution to the deformation gradient) to produce the
    assembly-ready element quantities.
    '''
    energy_density: EnergyDensity
    has_energy = True

    def quadrature_degree(self, shape_degree: int) -> int:
        '''The rule degree that integrates this energy on an element of `shape_degree`.

        The displacement gradient has degree `shape_degree - 1`, and the density's energy
        is `energy_degree`-degree in it, so the energy integrand is their product. This is
        higher than the linear stiffness's default on P2 (quartic St-VK reaches degree 4),
        so the energy path asks for its own rule.
        '''
        return self.energy_density.energy_degree * max(0, shape_degree - 1)

    def _dF_dx(self, geometry: ElementGeometry, q: int) -> FloatArray:
        '''(n_el, d, d, N, d): dF/dx at quadrature point q. `dF_dx[e,c,i,p,k]` is
        ∂F_ci/∂u_{p,k} = grad_phi[p,i] δ_ck, the shape-function contribution to the
        deformation gradient in the standard `F[c,i]` orientation.'''
        d = geometry.spatial_dim
        return np.einsum('epi,ck->ecipk', geometry.grad_phi[:, q, :, :d], np.eye(d))

    def element_energies(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements,) element energies at the given nodal displacements: the
        density at each quadrature point, weighted by `weight_detJ` and summed.'''
        grad_u = geometry.gradients(u_elements)   # (n_el, n_qp, d, d)
        total = np.zeros(geometry.n_elements)
        for q in range(geometry.n_qp):
            t = self.energy_density.evaluate(grad_u[:, q])
            total += t.W * geometry.weight_detJ[:, q]
        return total

    def element_residuals(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements, k) element residuals dPi/dx, node-major over the N nodes' d components.'''
        grad_u = geometry.gradients(u_elements)
        n_nodes, d = geometry.grad_phi.shape[2], geometry.spatial_dim
        residual = np.zeros((geometry.n_elements, n_nodes, d))
        for q in range(geometry.n_qp):
            t = self.energy_density.evaluate(grad_u[:, q])
            dF_dx = self._dF_dx(geometry, q)
            dW_dx = np.einsum('eci,ecipk->epk', t.P, dF_dx)
            residual += dW_dx * geometry.weight_detJ[:, q][:, None, None]
        return residual.reshape(geometry.n_elements, n_nodes * d)

    def element_tangents(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> FloatArray:
        '''(n_elements, k, k) element tangents d²Pi/dx², k = N * d in the residual's order.'''
        # d²Pi/dx² = A : dF_dx : dF_dx, where A = d²W/dF² is the density's material
        # tangent and dF_dx maps a nodal DOF to its contribution to F. Each "einsum"
        # is one double contraction (the shared F-index pairs "ci" and "kl"), summed
        # over the quadrature points and weighted by `weight_detJ`.
        grad_u = geometry.gradients(u_elements)
        n_nodes, d = geometry.grad_phi.shape[2], geometry.spatial_dim
        tangent = np.zeros((geometry.n_elements, n_nodes, d, n_nodes, d))
        for q in range(geometry.n_qp):
            t = self.energy_density.evaluate(grad_u[:, q])
            dF_dx = self._dF_dx(geometry, q)
            d2W_dx2 = np.einsum('ecikl,ecipq,eklrs->epqrs', t.A, dF_dx, dF_dx)
            tangent += d2W_dx2 * geometry.weight_detJ[:, q][:, None, None, None, None]
        k = n_nodes * d
        return tangent.reshape(geometry.n_elements, k, k)

    def _point_state(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        '''`(strain, cauchy, W)` at every point of the rule, the first two as
        `(n_elements, n_qp, 3, 3)` and the energy density as `(n_elements, n_qp)`.

        Stress is **Cauchy**, `sigma = J^-1 P F^T`, not the first Piola-Kirchhoff
        `P = dW/dF` the energy derivative gives: P is measured per unit undeformed
        area, so it is not comparable with the small-strain path's stress. The two
        agree to O(||grad u||). Strain is the density's own measure.

        The one convention still to reconcile is the plane-strain reduction a 2D
        solve makes; `F`, `P`, and the strain arrive in the standard orientation.
        '''
        grad_qp = geometry.gradients(u_elements)          # (n_el, n_qp, d, d)
        n_el, n_qp, d = grad_qp.shape[:3]
        grad_u = grad_qp.reshape(n_el * n_qp, d, d)
        t = self.energy_density.evaluate(grad_u)

        F = np.eye(d) + grad_u
        P = t.P

        J = np.linalg.det(F)
        cauchy = np.einsum('e,eci,eki->eck', 1.0 / J, P, F)

        # The density's own measure: Green-Lagrange for St-VK, eps for its linearisation.
        strain = self.energy_density.strain(grad_u)

        if d == 2:
            # Restore the stress in the restrained direction, which the 2D Voigt
            # vector omits. Without it the nonlinear path would report a different
            # von Mises than the linear one for the same material.
            sigma_zz = self.energy_density.out_of_plane_stress(strain) / J
            strain = _with_out_of_plane(strain, np.zeros(len(strain)))
            cauchy = _with_out_of_plane(cauchy, sigma_zz)

        return (strain.reshape(n_el, n_qp, 3, 3), cauchy.reshape(n_el, n_qp, 3, 3),
                np.asarray(t.W).reshape(n_el, n_qp))

    def derived_field(self) -> DerivedField:
        return StressField(self)

    def solution(self, space: 'FunctionSpace', u: FloatArray) -> 'ElasticSolution':
        return ElasticSolution.from_solve(space, u, self)

    def sample(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticPointFields:
        '''Strain and Cauchy stress at every point of `geometry`'s rule; see `_point_state`.'''
        strain, cauchy, _ = self._point_state(geometry, u_elements)
        return ElasticPointFields(strain, cauchy)

    def derived_fields(
        self, geometry: ElementGeometry, u_elements: FloatArray,
    ) -> ElasticFields:
        '''Element strain, Cauchy stress, and compliance at a solved displacement.

        Strain and stress are the element mean of `sample` over the rule,
        matching the linear path. Compliance is twice the stored energy integrated
        over the element, which for a quadratic W is `∫ S : E`, the work-conjugate
        pair. Contracting the reported Cauchy stress with E instead mixes measures
        and runs ~30% wrong at finite strain.
        '''
        strain, cauchy, W = self._point_state(geometry, u_elements)
        compliance = 2.0 * np.einsum('eq,eq->e', W, geometry.weight_detJ)
        return ElasticFields(_element_mean(strain, geometry.weight_detJ),
                             _element_mean(cauchy, geometry.weight_detJ), compliance)
