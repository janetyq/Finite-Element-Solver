import logging
from collections.abc import Callable

import numpy as np

from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.fields import FieldShape, Scalar, Vector
from fem.solution import Solution
from fem.space import FunctionSpace, dof_indices
from fem.forms import Form, LaplacianForm, LinearElasticForm, MassForm
from fem.materials import LinearElasticMaterial
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


class LinearElastic(Equation):
    '''Small-strain linear elasticity. E may be a scalar or a per-element array
    (TopologyOptimizer sets a density-scaled modulus).'''
    field: FieldShape = Vector()

    def __init__(
        self,
        E: float | ElementField,
        nu: float,
        source: FieldValue = None,
    ) -> None:
        super().__init__(source)
        self.E = E
        self.nu = nu


def stiffness_form(equation: Equation) -> Form:
    '''The bilinear stiffness form for an equation.

    LinearElastic carries material data, so its form is built from a
    LinearElasticMaterial; the scalar diffusion family (Projection / Poisson /
    Heat / Wave) shares the material-free Laplacian. This is the one
    equation-specific choice the steady solve makes -- selecting the operator --
    named and lifted out of the solve so the solve itself stays PDE-agnostic.
    '''
    if isinstance(equation, LinearElastic):
        return LinearElasticForm(LinearElasticMaterial(equation.E, equation.nu))
    return LaplacianForm()


class Solver:
    def __init__(
        self,
        mesh: Mesh,
        equation: Equation,
        boundary_conditions: BoundaryConditions | None = None,
    ) -> None:
        self.mesh = mesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions()
        # Derived, never passed: the component count follows from the equation's
        # field and the mesh, so a space that disagrees with the equation it is
        # solving is not constructible here.
        self.n_components = self.equation.field.components_for(mesh.spatial_dim)
        self.space = FunctionSpace(mesh, n_components=self.n_components)
        self.solution = Solution(mesh, self.n_components)

        self._resolve_bc()

    def _resolve_bc(self) -> None:
        '''Bind the boundary-condition spec to the current mesh and component count.

        Called again whenever the mesh changes (adaptive refinement), which is
        the whole reason the spec is kept separate from its resolution.
        '''
        self.resolved_bc = self.boundary_conditions.resolve(self.mesh, self.n_components)

    def solve(self) -> Solution:
        self.solution.reset()
        if isinstance(self.equation, (Projection, Poisson, LinearElastic)):
            self._solve_steady()
        else:
            raise ValueError(f"No solver for equation type: {type(self.equation).__name__}")
        return self.solution

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

    def _solve_steady(self) -> None:
        '''Steady linear solve, through the composition core.

        A LinearProblem hands a matrix, a load, and the constraints to LinearSolve;
        an elastic problem additionally recovers stress fields from the same form
        that assembled its operator.
        '''
        logger.info('Solving steady system...')
        problem = self._steady_problem()
        u = LinearSolve().solve(problem)
        self.solution.set_values("u", u)

        if isinstance(problem.operator, LinearElasticForm):
            u_elements = u[dof_indices(self.mesh.elements, self.n_components)]
            strain, stress, compliance = problem.operator.derived_fields(self.space.geometry, u_elements)
            self.solution.set_values("strain", np.linalg.norm(strain, axis=-1))
            self.solution.set_values("stress", np.linalg.norm(stress, axis=-1))
            self.solution.set_values("compliance", compliance)

    def adaptive_refinement(
        self,
        estimator: Callable[['Solver'], ElementField],
        max_triangles: int = 1000,
        max_iters: int = 20,
        refine_fraction: float = 0.9,
    ) -> Solution:
        '''Refine where the error estimate is largest, re-solving on each new mesh.

        `estimator(solver) -> per-element error` is a parameter rather than a key
        read out of `self.solution` because the estimate has to be recomputed
        every round: once elements have been split, the previous array is both
        stale and the wrong length, so indexing it selects unrelated elements.
        That was the "bug somewhere" this method used to carry.

        Elements whose estimate is within `refine_fraction` of the largest are
        refined. Returns the solution on the final mesh.
        '''
        self.boundary_conditions.check_remeshable()

        refiner = RedGreenRefiner(self.mesh)
        for _ in range(max_iters):
            if len(self.mesh.elements) >= max_triangles:
                break

            residuals = np.asarray(estimator(self), dtype=float)
            if len(residuals) != len(self.mesh.elements):
                raise ValueError(
                    f'estimator returned {len(residuals)} values for '
                    f'{len(self.mesh.elements)} elements'
                )
            refine_idxs = np.flatnonzero(residuals >= refine_fraction * residuals.max())
            if len(refine_idxs) == 0:
                break

            self.mesh = refiner.refine([int(i) for i in refine_idxs])
            # The refined mesh renumbers vertices, so anything index-keyed has to
            # be rebuilt from its specification rather than carried over. The
            # space owns cached operators sized to the old mesh, so it is
            # replaced rather than refreshed.
            self.space = FunctionSpace(self.mesh, n_components=self.n_components)
            self._resolve_bc()
            self.solution = Solution(self.mesh, self.n_components)
            self.solve()

        return self.solution

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
