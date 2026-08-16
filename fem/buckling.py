"""The buckling facade: mesh + equation + reference load -> critical loads and modes.

Linearised (eigenvalue) buckling, the finite-element analogue of Euler's column
formula. Where `Solver` answers "what shape does this load hold the structure in?",
`BucklingSolver` answers "how far can this load be scaled before the structure snaps
sideways into a different shape?" It reports the load factor λ and the shape (mode)
it buckles into.

The method is three steps, and only the middle one is new to the package:

1. **Reference solve.** Apply a reference load and solve the ordinary linear-elastic
   problem (through `Solver`), recovering the membrane prestress σ₀ it puts every
   element under.
2. **Geometric stiffness.** Assemble `K_g(σ₀)` (`GeometricStiffnessForm`), the
   initial-stress matrix that softens the structure under compression.
3. **Eigenproblem.** Solve `K φ = -λ K_g φ` for the lowest few λ. `λ_1` is the
   critical load factor: the reference load times `λ_1` is the buckling load, and `φ_1`
   is the shape it buckles into.

Every other solver here answers `A x = b`: one matrix, one right-hand side, solved for `x`
by `DiscreteSystem` (which removes the fixed Dirichlet DOFs and factors the matrix once).
Buckling asks a different question (`K φ = -λ K_g φ`, which load factors λ and shapes φ
satisfy it), and an eigenproblem has no right-hand side. So removing the fixed DOFs and
calling `scipy.sparse.linalg.eigsh` is `EigenSolve`'s job (the eigenproblem's counterpart to
`LinearSolve`). `BucklingSolver` just assembles `K` and `K_g`, passes them to `EigenSolve`,
and turns the returned eigenvalues into load factors; the reference solve, the assembly, and
the stress recovery are existing machinery reused unchanged.
"""
import logging

import numpy as np

from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import LinearElastic, StrainMeasure
from fem.forms import GeometricStiffnessForm, LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.solution import BucklingSolution, ElasticSolution
from fem.solve import EigenSolve
from fem.solver import Solver
from fem.typing import FloatArray

logger = logging.getLogger(__name__)


class BucklingSolver:
    '''Linearised buckling: the load factors and mode shapes of a compressed structure.

    Holds the same three things a steady solve does (a mesh, a `LinearElastic`
    equation, and a boundary-condition spec), where the conditions also encode the
    reference load whose buckling multiplier is sought (a compressive traction on an
    end, say). The load factors it reports are dimensionless multipliers of that load,
    so the caller multiplies by the reference load's magnitude to get the buckling load
    in physical units.

    Small-strain only: linearised buckling is built on the constant elastic stiffness
    and a prestress from one linear solve, so a Green-Lagrange equation (whose stiffness
    is not constant) is rejected. That is the nonlinear post-buckling path, which needs
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
        # bending; it converges to Euler only on a mesh refined through the thickness.
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
        # λ > 0, so nothing buckles. A rigorous, cheap guard for the clean cases (an
        # unstressed structure or one in pure tension) that also spares `eigsh` a
        # trivial eigenproblem (`K_g = 0` forces every μ to 0, so no finite buckling
        # factor). It does not catch a member in overall tension whose
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
        # -K_g φ = μ K φ with K the PD side; μ = 1/λ, so the largest μ ('LA') are the
        # smallest load factors, reached directly without shift-invert.
        eigensolve = EigenSolve(self.n_modes, which='LA')
        mu, modes = eigensolve.solve(-K_g, K, resolved.free_idxs, space.n_dofs)
        factors, modes = self._buckling_factors(mu, modes)
        return BucklingSolution(self.mesh, space.n_components, factors, modes)

    @staticmethod
    def _buckling_factors(
        mu: FloatArray, modes: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        '''Raw eigenvalues `μ = 1/λ` to ascending, positive-only load factors.

        Only positive μ buckle: a negative μ is a direction the load stiffens, not
        softens. Modes ride along with their factors to stay aligned.
        '''
        tol = 1e-8 * float(np.max(np.abs(mu)))
        positive = mu > tol
        if not positive.any():
            raise ValueError(
                'no positive buckling factor found: the reference load puts the structure '
                'in tension, which stiffens rather than buckles it. Reverse the load.'
            )

        factors = 1.0 / mu[positive]
        modes = modes[positive]
        order = np.argsort(factors)
        return factors[order], modes[order]
