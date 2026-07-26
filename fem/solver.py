import logging
from enum import Enum

import numpy as np

from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.fields import FieldShape, Scalar, Vector
from fem.solution import ElasticSolution, FieldSolution, Solution
from fem.space import FunctionSpace, dof_indices
from fem.forms import Form, LaplacianForm, LinearElasticForm, MassForm
from fem.materials import LinearElasticMaterial
from fem.linalg import IterativeBackend, LinearAlgebra, rigid_body_modes
from fem.problem import LinearProblem
from fem.solve import LinearSolve
from fem.typing import ElementField, FieldValue

logger = logging.getLogger(__name__)

class Equation:
    '''Base class for a PDE to solve.

    An Equation is typed data: it says *what* to solve and carries the physical
    parameters, while the Solver owns *how* to solve it (the same equation, e.g.
    LinearElastic, may be handled by several solvers). Transient problems are not
    equation types: heat and wave are a steady operator paired with a time
    integrator (see fem.problem.heat / .wave and fem.integrators).

    `field` says what kind of value the unknown takes; the DOFs per node follow
    from it and the mesh, so no subclass writes the count down. Not a ClassVar:
    a system of k equations would carry its count as constructor data.

    `source` is the PDE's right-hand side f (a body force for elasticity), given
    as a constant or a callable of position. It lives here rather than on
    BoundaryConditions because it is data of the equation, not of the boundary.
    '''
    field: FieldShape = Scalar()

    def __init__(self, source: FieldValue = None) -> None:
        self.source = source


class Projection(Equation):
    '''L2 projection of the source field onto the FE space (M u = b).'''


class Poisson(Equation):
    '''Poisson equation (K u = b).'''


class StrainMeasure(Enum):
    '''Which strain the elastic energy is built on -- the kinematics axis.

    The material `W` is one function; the two paths differ only in the strain fed
    to it (see `fem.energies`). SMALL is the infinitesimal `ε = ½(∇u + ∇uᵀ)`,
    solved directly by `Solver`; GREEN_LAGRANGE is the geometrically exact
    `S = ½(FᵀF − I)` (St-Venant–Kirchhoff), which only `EnergySolver` can solve
    because its energy is not quadratic.
    '''
    SMALL = 'small'
    GREEN_LAGRANGE = 'green_lagrange'


class LinearElastic(Equation):
    '''Elasticity with a selectable strain measure. `kinematics` is SMALL by
    default (infinitesimal strain, the linear `Solver` path); GREEN_LAGRANGE
    selects the St-Venant–Kirchhoff model, which needs `EnergySolver`. E may be a
    scalar or a per-element array (TopologyOptimizer sets a density-scaled modulus).'''
    field: FieldShape = Vector()

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue = None,
        kinematics: StrainMeasure = StrainMeasure.SMALL,
    ) -> None:
        super().__init__(source)
        self.E = E
        self.nu = nu
        self.kinematics = kinematics


def stiffness_form(equation: Equation) -> Form:
    '''The bilinear stiffness form for an equation.

    LinearElastic carries material data, so its form is built from a
    LinearElasticMaterial; the scalar diffusion family (Projection / Poisson, and
    the Laplacian behind the heat / wave problems) shares the material-free
    Laplacian. This is the one equation-specific choice the steady solve makes --
    selecting the operator -- named and lifted out of the solve so the solve itself
    stays PDE-agnostic.

    The bilinear form exists only for the small-strain measure: a Green-Lagrange
    energy is not quadratic, so it has no constant stiffness. A finite-strain
    LinearElastic is rejected here rather than silently linearised.
    '''
    if isinstance(equation, LinearElastic):
        if equation.kinematics is not StrainMeasure.SMALL:
            raise NotImplementedError(
                f'Solver is small-strain only; {equation.kinematics.name} kinematics '
                'has no constant stiffness. Use EnergySolver.'
            )
        return LinearElasticForm(LinearElasticMaterial(equation.E, equation.nu))
    return LaplacianForm()


class Solver:
    def __init__(
        self,
        mesh: Mesh,
        equation: Equation,
        boundary_conditions: BoundaryConditions | None = None,
        backend: LinearAlgebra | None = None,
    ) -> None:
        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        # The linear-algebra backend for the steady solve: direct by default, or an
        # IterativeBackend for a large SPD system. A steady LinearElastic / Poisson is
        # SPD; the facade forwards it to LinearSolve untouched.
        self.backend = backend
        # Derived, never passed: the component count follows from the equation's
        # field and the mesh, so a space that disagrees with the equation it is
        # solving is not constructible here.
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.space = FunctionSpace(mesh, n_components=self.n_components)
        # The most recent solve, so an adaptive-refinement estimator can read it.
        self.solution: Solution | None = None

        self._resolve_bc()

    def _resolve_bc(self) -> None:
        '''Bind the boundary-condition spec to the current mesh and component count.

        Called again whenever the mesh changes (adaptive refinement), which is
        the whole reason the spec is kept separate from its resolution.
        '''
        self.resolved_bc = self.boundary_conditions.resolve(self.mesh, self.n_components)

    def remesh(self, mesh: Mesh) -> None:
        '''Rebind the solver to a new mesh, rebuilding the space and re-resolving BCs.

        A refined mesh renumbers vertices, so every derived, index-keyed object is
        rebuilt from its specification rather than carried over: the space owns
        cached operators sized to the old mesh, and the resolved BC is keyed by it.
        This is what lets an outer driver (AdaptiveRefinement) advance the solver
        across meshes without reaching into its state.
        '''
        self.mesh = mesh
        self.space = FunctionSpace(mesh, n_components=self.n_components)
        self._resolve_bc()

    def solve(self) -> Solution:
        if isinstance(self.equation, (Projection, Poisson, LinearElastic)):
            self.solution = self._solve_steady()
            return self.solution
        raise ValueError(f"No solver for equation type: {type(self.equation).__name__}")

    def _steady_problem(self) -> LinearProblem:
        '''The composition for a steady equation: operator + source + constraints.

        The operator is the only equation-specific choice -- the mass matrix for an
        L2 projection, the stiffness otherwise. Built on the solver's own space so
        adaptive refinement (which rebuilds the space) is picked up on the next solve.
        '''
        operator: Form = (
            MassForm(self.n_components)
            if isinstance(self.equation, Projection)
            else stiffness_form(self.equation)
        )
        return LinearProblem(self.space, operator, self.equation.source, self.boundary_conditions)

    def _backend_for(self, problem: LinearProblem) -> LinearAlgebra | None:
        '''The solve backend, giving an elastic AMG solve its rigid-body near-kernel.

        A vector elasticity stiffness has the rigid-body modes as its low-energy
        near-kernel; AMG needs them to keep CG's iteration count flat under
        refinement. This is the one solve detail that depends on *which* equation is
        being solved, so the equation-aware facade supplies it -- restricted to the
        free DOFs, to match the block the backend factors -- rather than the
        physics-agnostic backend guessing. An explicit near-kernel the caller set is
        left untouched; the scalar Laplacian family needs none.
        '''
        if isinstance(self.equation, LinearElastic) and isinstance(self.backend, IterativeBackend) \
                and self.backend.near_null_space is None:
            free = problem.constraints[0]
            modes = rigid_body_modes(self.mesh.vertices, self.n_components)[free]
            return self.backend.with_near_null_space(modes)
        return self.backend

    def _solve_steady(self) -> Solution:
        '''Steady linear solve, through the composition core.

        A LinearProblem hands a matrix, a load, and the constraints to LinearSolve;
        an elastic problem additionally recovers stress fields from the same form
        that assembled its operator, returning an ElasticSolution rather than a bare
        FieldSolution.
        '''
        logger.info('Solving steady system...')
        problem = self._steady_problem()
        u = LinearSolve(self._backend_for(problem)).solve(problem)

        if isinstance(problem.operator, LinearElasticForm):
            u_elements = u[dof_indices(self.mesh.elements, self.n_components)]
            strain, stress, compliance = problem.operator.derived_fields(self.space.geometry, u_elements)
            return ElasticSolution(
                self.mesh, self.n_components, u,
                np.linalg.norm(strain, axis=-1),
                np.linalg.norm(stress, axis=-1),
                compliance,
            )
        return FieldSolution(self.mesh, self.n_components, u)

    # # residuals
    # def calculate_residuals(self):
    #     # apriori and aposteriori error estimation

    #     equation_residuals = {
    #         "projection": self._calculate_projection_residuals,
    #         "poisson": self._calculate_poisson_residuals,
    #         "heat": self._calculate_heat_residuals,
    #         "wave": self._calculate_wave_residuals,
    #         "linear_elastic": self._calculate_linear_elastic_residuals,
    #     }

    #     residual_method = equation_residuals.get(self.equation.name)
    #     if residual_method:
    #         residual_method()
    #     else:
    #         raise ValueError(f"Unknown equation name: {self.equation.name}")

    # def calculate_projection_residuals(self, apriori=True):
    #     # Apriori error ||e|| <= C * h^2 * ||f"||
    #     if apriori:
    #         # compute apriori residual
    #         residuals = np.zeros(len(self.mesh.elements))
    #         for e_idx, element in enumerate(self.mesh.elements):
    #             residuals[e_idx] = 0 # placeholder
    #         self.solution.set_values("apriori_residual", residuals)
    #     else:
    #         # compute aposteriori residual
    #         residuals = None
    #         self.solution.set_values("aposteriori_residual", residuals)
    #         pass
