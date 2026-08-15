"""The modal facade: mesh + equation -> natural frequencies and mode shapes.

Free (undamped) vibration -- the finite-element analogue of a beam's natural tones.
Where `BucklingSolver` asks how far a load can be scaled before the structure snaps
sideways, `ModalSolver` asks the load-free question: what shapes does the structure
oscillate in when displaced and released, and at what frequencies?

Undamped free vibration is `M u'' + K u = 0`; a standing wave `u(t) = phi cos(omega t)`
turns it into the generalized symmetric eigenproblem

    K phi = omega^2 M phi,

with `K` the elastic stiffness and `M` the consistent mass matrix. The eigenvalues are
the squared natural angular frequencies and the eigenvectors the mode shapes. No applied
load enters -- unlike buckling, whose `K_g` comes from a reference solve -- so the result
is a property of the structure alone (its stiffness, mass, and supports), the way a
bell's pitch is a property of the bell and not of how hard it is struck.

The lowest frequencies are the ones that matter (a forcing near them resonates), so the
eigensolve uses shift-invert about `sigma = 0` through `EigenSolve` -- the standard way
to pull the smallest eigenvalues of a large sparse pencil, factoring `K` on the free
block once. `MassForm` integrates the unit-density mass, so a scalar `density` scales it
to physical units; the frequencies then go as `sqrt(E / density)`, the material
dependence a modal analysis turns on (the mode *shapes*, by contrast, are set by geometry
and supports, unchanged by a uniform scaling of stiffness or mass).
"""
import logging

import numpy as np

from fem.boundary import BoundaryConditions
from fem.elements import Element
from fem.equations import LinearElastic, StrainMeasure
from fem.forms import LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.solution import ModalSolution
from fem.solve import EigenSolve
from fem.solver import Solver
from fem.typing import FloatArray

logger = logging.getLogger(__name__)


class ModalSolver:
    '''Free-vibration modal analysis: the natural frequencies and mode shapes.

    Holds a mesh, a `LinearElastic` equation, and a boundary-condition spec whose
    Dirichlet supports ground the structure (a cantilever's clamp). No load is needed or
    read -- the modes are load-free. `density` is the mass density scaling the unit-
    density consistent mass matrix, and so sets the physical frequency units.

    Small-strain only, like buckling: the modes are computed about the undeformed,
    unstressed configuration on the constant elastic stiffness, so a Green-Lagrange
    equation (whose stiffness is not constant) is rejected.

    The supports must remove every rigid-body mode: shift-invert about zero factors `K`
    on the free block, which is singular if the structure can translate or rotate
    freely. A fully unsupported structure therefore needs a different shift and is out of
    scope here; a clamped or otherwise grounded structure is the intended case.
    '''

    def __init__(
        self,
        mesh: Mesh,
        equation: LinearElastic,
        boundary_conditions: BoundaryConditions | None = None,
        n_modes: int = 6,
        element_type: type[Element] | None = None,
        density: float = 1.0,
    ) -> None:
        if not isinstance(equation, LinearElastic):
            raise ValueError(
                f'ModalSolver analyses elastic vibration; got {type(equation).__name__}.'
            )
        if equation.kinematics is not StrainMeasure.SMALL:
            raise NotImplementedError(
                'modal analysis linearises about the unstressed configuration on the '
                'constant small-strain stiffness; a Green-Lagrange equation has no '
                'constant stiffness.'
            )
        if n_modes < 1:
            raise ValueError(f'n_modes must be at least 1, got {n_modes}')
        if density <= 0:
            raise ValueError(f'density must be positive, got {density}')

        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = (
            boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        )
        self.n_modes = n_modes
        # As in buckling, bending dominates the low modes and the constant-strain triangle
        # locks in bending; QuadraticTriangleElement reaches the analytic frequencies on a
        # far coarser mesh. `None` means the linear element for the mesh's node count.
        self.element_type = element_type
        self.density = density

    def solve(self) -> ModalSolution:
        '''Solve the modal eigenproblem and return its frequencies and mode shapes.'''
        # Only the discretization is needed, not a solved state -- modal analysis reads no
        # prestress, unlike buckling -- so the Solver is built for its space and not run.
        space = Solver(self.mesh, self.equation, self.boundary_conditions,
                       element_type=self.element_type).space

        material = LinearElasticMaterial(self.equation.E, self.equation.nu)
        K = space.assemble(LinearElasticForm(material))
        M = self.density * space.mass_matrix

        resolved = self.boundary_conditions.resolve(space.nodes, space.n_components)
        logger.info('Modal: solving the eigenproblem K phi = omega^2 M phi...')
        # Shift-invert about zero returns the smallest omega^2 -- the lowest frequencies,
        # the ones a forcing resonates with -- factoring K on the free block once.
        eigensolve = EigenSolve(self.n_modes, sigma=0.0, which='LM')
        omega_squared, modes = eigensolve.solve(K, M, resolved.free_idxs, space.n_dofs)

        frequencies, modes = self._natural_frequencies(omega_squared, modes)
        return ModalSolution(self.mesh, space.n_components, frequencies, modes)

    @staticmethod
    def _natural_frequencies(
        omega_squared: FloatArray, modes: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        '''Sort by frequency and take omega = sqrt(omega^2), ascending.

        Round-off can push a well-constrained mode's eigenvalue a hair below zero; clamp
        at zero before the square root so it reads as omega = 0 rather than a NaN. The
        modes ride along with their frequencies so the returned pair stays aligned.
        '''
        order = np.argsort(omega_squared)
        omega = np.sqrt(np.maximum(omega_squared[order], 0.0))
        return omega, modes[order]
