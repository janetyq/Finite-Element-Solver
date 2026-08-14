"""The buckling facade: mesh + equation + reference load -> critical loads and modes.

Linearised (eigenvalue) buckling, the finite-element analogue of Euler's column
formula. Where `Solver` answers "what shape does this load hold the structure in?",
`BucklingSolver` answers "how far can this load be scaled before the structure snaps
sideways into a different shape?" -- the load factor λ and the shape (mode) it buckles
into.

The method is three steps, and only the middle one is new to the package:

1. **Reference solve.** Apply a reference load and solve the ordinary linear-elastic
   problem (through `Solver`), recovering the membrane prestress σ₀ it puts every
   element under.
2. **Geometric stiffness.** Assemble `K_g(σ₀)` (`GeometricStiffnessForm`), the
   initial-stress matrix that softens the structure under compression.
3. **Eigenproblem.** Solve `K φ = -λ K_g φ` for the lowest few λ. `λ_1` is the
   critical load factor: the reference load times `λ_1` is the buckling load, and `φ_1`
   is the shape it buckles into.

This is the package's first solve that is not `A x = b`: `DiscreteSystem` factors a
matrix and back-substitutes right-hand sides, which an eigenproblem has none of, so the
free-DOF reduction is done here and handed to `scipy.sparse.linalg.eigsh`. Everything
else -- the reference solve, the assembly, the stress recovery -- is the existing
linear machinery reused unchanged.
"""
import logging

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import LinearElastic, StrainMeasure
from fem.forms import GeometricStiffnessForm, LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.solution import BucklingSolution, ElasticSolution
from fem.solver import Solver
from fem.typing import DofIndices, FloatArray, Operator

logger = logging.getLogger(__name__)


class BucklingSolver:
    '''Linearised buckling: the load factors and mode shapes of a compressed structure.

    Holds the same three things a steady solve does -- a mesh, a `LinearElastic`
    equation, and a boundary-condition spec -- where the conditions also encode the
    *reference load* whose buckling multiplier is sought (a compressive traction on an
    end, say). The load factors it reports are dimensionless multipliers of that load,
    so the caller multiplies by the reference load's magnitude to get the buckling load
    in physical units.

    Small-strain only: linearised buckling is built on the constant elastic stiffness
    and a prestress from one linear solve, so a Green-Lagrange equation (whose stiffness
    is not constant) is rejected -- that is the nonlinear post-buckling path, which needs
    an arc-length Newton solve the package does not have yet.
    '''

    def __init__(
        self,
        mesh: Mesh,
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions | None = None,
        n_modes: int = 4,
        element_type: type[Element] | None = None,
    ) -> None:
        if not isinstance(equation, LinearElastic):
            raise ValueError(
                f'BucklingSolver analyses elastic buckling; got {type(equation).__name__}.'
            )
        if equation.kinematics is not StrainMeasure.SMALL:
            raise NotImplementedError(
                'linearised buckling uses the small-strain stiffness and a prestress from '
                'one linear solve; a Green-Lagrange equation has no constant stiffness. '
                'Nonlinear post-buckling (path-following past the bifurcation) is a '
                'separate solve the package does not provide.'
            )
        if n_modes < 1:
            raise ValueError(f'n_modes must be at least 1, got {n_modes}')

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = (
            boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        )
        self.n_modes = n_modes
        # The element order, `None` meaning the linear element for the mesh. Bending
        # dominates a buckling mode, and the linear (constant-strain) triangle locks in
        # bending -- it converges to Euler only on a mesh refined through the thickness.
        # `QuadraticTriangleElement` (P2) reaches the same accuracy on a far coarser mesh.
        self.element_type = element_type
        # The reference (pre-buckling) solve, kept like `Solver.solution` so a caller
        # can read the prestress state the mode shapes were computed about.
        self.reference: ElasticSolution | None = None

    def solve(self) -> BucklingSolution:
        '''Solve the buckling eigenproblem and return its factors and modes.'''
        logger.info('Buckling: reference solve for the prestress state...')
        ref_solver = Solver(self.mesh, self.equation, self.boundary_conditions,
                            element_type=self.element_type)
        reference = ref_solver.solve()
        if not isinstance(reference, ElasticSolution):
            raise TypeError('buckling needs the recovered stress; got a bare FieldSolution')
        self.reference = reference

        space = ref_solver.space
        d = space.spatial_dim

        material = LinearElasticMaterial(self.equation.E, self.equation.nu)
        K = space.assemble(LinearElasticForm(material))
        # The in-plane prestress drives the geometric stiffness; σ_zz (the plane-strain
        # out-of-plane component) has no in-plane displacement gradient to couple to.
        prestress = np.ascontiguousarray(reference.stress[:, :d, :d])

        # Buckling needs compression somewhere: if every principal stress of the prestress
        # is non-negative, K_g is positive-semidefinite and K + λ K_g stays SPD for all
        # λ > 0, so nothing buckles. A rigorous, cheap guard for the clean cases -- an
        # unstressed structure or one in pure tension -- that also keeps a degenerate
        # pencil away from `eigsh`. It does not catch a member in *overall* tension whose
        # clamped end develops local corner compression: that discretely does have a
        # (huge, spurious) mode, and there is no threshold-free way to rule it out here.
        principal = np.linalg.eigvalsh(prestress)     # (n_el, d), ascending per element
        scale = float(np.abs(prestress).max())
        if scale == 0.0 or float(principal.min()) > -1e-9 * scale:
            raise ValueError(
                'no compressive prestress: the reference load leaves the structure in '
                'tension everywhere (or unstressed), which stiffens rather than buckles, '
                'so there is no buckling mode. Reverse the load.'
            )

        K_g = space.assemble(GeometricStiffnessForm(prestress))

        resolved = self.boundary_conditions.resolve(space.nodes, space.n_components)
        logger.info('Buckling: solving the eigenproblem K phi = -lambda K_g phi...')
        factors, mode_free = self._eigensolve(K, K_g, resolved.free_idxs)

        # The buckling modes satisfy the homogeneous essential conditions -- a clamped
        # node cannot move in the mode either -- so the fixed DOFs stay zero and only
        # the free block is filled in.
        modes = np.zeros((len(factors), space.n_dofs))
        modes[:, resolved.free_idxs] = mode_free.T
        return BucklingSolution(self.mesh, space.n_components, factors, modes)

    def _eigensolve(
        self, K: Operator, K_g: Operator, free: DofIndices,
    ) -> tuple[FloatArray, FloatArray]:
        '''The lowest buckling factors and their free-DOF mode vectors.

        Reduces both operators to the free-free block (the fixed DOFs are zero in every
        mode) and solves the symmetric-definite pencil `-K_g φ = μ K φ` with `K` as the
        positive-definite "mass" matrix. `μ = 1/λ`, so the algebraically largest μ are
        the smallest load factors -- and `which='LA'` finds those directly, without the
        shift-invert a smallest-λ request would need. Only positive μ are buckling
        modes: a negative μ is a direction the reference load stiffens (tension) rather
        than softens, and cannot buckle.
        '''
        Kff = K[np.ix_(free, free)]
        Kgff = K_g[np.ix_(free, free)]
        n_free = Kff.shape[0]

        # eigsh finds interior-of-spectrum pairs by Lanczos, which needs a subspace
        # comfortably larger than the number of modes requested; cap the request so a
        # small structure (or a smoke-test mesh) asks for fewer rather than failing.
        k = min(self.n_modes, n_free - 2)
        if k < 1:
            raise ValueError(
                f'too few free DOFs ({n_free}) to extract a buckling mode; '
                'the structure is over-constrained or the mesh is trivially small'
            )

        # Symmetrise against round-off: assembly is symmetric by construction, but eigsh
        # assumes exact symmetry and a stray 1e-16 asymmetry perturbs 'LA'.
        A = -0.5 * (Kgff + Kgff.T)
        M = 0.5 * (Kff + Kff.T)
        try:
            mu, vecs = eigsh(A.tocsc(), k=k, M=M.tocsc(), which='LA')
        except ArpackNoConvergence as failure:
            # Keep the modes that did converge: a coarse mesh resolves the first few cleanly
            # while the higher ones, which it represents poorly, can stall. Reporting the
            # lower modes is what a caller wants -- they are the ones that buckle first --
            # so only a total failure (nothing converged) is unrecoverable.
            mu, vecs = failure.eigenvalues, failure.eigenvectors
            if mu.size == 0:
                raise ValueError(
                    'the buckling eigensolver did not converge to any mode; the mesh may '
                    'be too coarse for the modes requested, or the structure ill-posed'
                ) from failure

        tol = 1e-8 * float(np.max(np.abs(mu)))
        positive = mu > tol
        if not positive.any():
            raise ValueError(
                'no positive buckling factor found: the reference load puts the structure '
                'in tension, which stiffens rather than buckles it. Reverse the load.'
            )

        factors = 1.0 / mu[positive]
        vecs = vecs[:, positive]
        order = np.argsort(factors)
        return factors[order], vecs[:, order]
