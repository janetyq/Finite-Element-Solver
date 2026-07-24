import logging
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from scipy.sparse import block_array

from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.fields import FieldShape, Scalar, Vector
from fem.regions import evaluate_field
from fem.solution import Solution
from fem.space import FunctionSpace, dof_indices
from fem.forms import Form, LaplacianForm, LinearElasticForm
from fem.materials import LinearElasticMaterial
from fem.system import DiscreteSystem
from fem.typing import (
    Constraints,
    DofVector,
    ElementField,
    FieldValue,
    Operator,
    VertexField,
)

EquationT = TypeVar('EquationT', bound='Equation')

logger = logging.getLogger(__name__)

class Equation:
    '''Base class for a PDE to solve.

    An Equation is typed data: it says *what* to solve and carries the physical
    parameters / initial conditions, while the Solver owns *how* to solve it
    (the same equation, e.g. LinearElastic, may be handled by several solvers).

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


class Heat(Equation):
    '''Transient heat equation, solved with backward Euler.'''

    def __init__(
        self,
        u_initial: VertexField,
        dt: float,
        iters: int,
        source: FieldValue = None,
    ) -> None:
        super().__init__(source)
        self.u_initial = u_initial
        self.dt = dt
        self.iters = iters


class Wave(Equation):
    '''Wave equation, solved with Crank-Nicolson.'''

    def __init__(
        self,
        u_initial: VertexField,
        dudt_initial: VertexField,
        c: float,
        dt: float,
        iters: int,
        source: FieldValue = None,
    ) -> None:
        super().__init__(source)
        self.u_initial = u_initial
        self.dudt_initial = dudt_initial
        self.c = c
        self.dt = dt
        self.iters = iters


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

    def _equation_as(self, kind: type[EquationT]) -> EquationT:
        '''The equation, narrowed to the type the calling solver expects.

        `solve` dispatches on the exact type, so this never fires in practice;
        it states that invariant somewhere both a reader and a type checker can
        see it, instead of each solve_* reaching for attributes that the
        declared type does not have.
        '''
        if not isinstance(self.equation, kind):
            raise TypeError(
                f'{kind.__name__} solver called with {type(self.equation).__name__}'
            )
        return self.equation

    def solve(self) -> Solution:
        self.assemble_everything() # TODO: don't call this every time
        self.solution.reset()
        
        equation_solvers = {
            Projection: self.solve_steady,
            Poisson: self.solve_steady,
            Heat: self.solve_heat,
            Wave: self.solve_wave,
            LinearElastic: self.solve_steady,
        }

        solver_fn = equation_solvers.get(type(self.equation))
        if solver_fn is None:
            raise ValueError(f"No solver for equation type: {type(self.equation).__name__}")

        solver_fn()

        return self.solution

    def assemble_everything(self) -> None:
        # The mass matrices are geometry only, so the space caches them. Stiffness
        # takes material data for the elastic case, so it is rebuilt here -- which
        # is also why it cannot be a property of a material-free space.
        self.M = self.space.mass_matrix
        self.M_b = self.space.boundary_mass_matrix
        self.stiffness = stiffness_form(self.equation)
        self.K = self.space.assemble(self.stiffness)

        # RHS: the linear form L(v) = int f.v over the volume plus int t.v over the
        # boundary. M @ f is the *exact* integral of f's P1 interpolant (M_ij =
        # int phi_i phi_j), so the load is already assembled through a Form -- the
        # mass form used as a load operator rather than a system matrix. A
        # first-class LinearForm earns its place only once the source varies within
        # an element, which needs quadrature to sample f at interior points; a
        # time-varying f(.,t) does not -- it just needs re-evaluating M @ f_t each
        # step. A Robin condition would add its matrix term to the LHS here, from a
        # boundary stiffness the space would assemble.
        source_load = evaluate_field(self.equation.source, self.mesh.vertices, self.n_components)
        self.b = (self.M @ source_load.flatten()).flatten()
        self.b += (self.M_b @ self.resolved_bc.neumann_load.flatten()).flatten()

    def wave_energy(self, u: VertexField, dudt: VertexField, c: float) -> float:
        '''Total wave energy 1/2 (c^2 u^T K u + dudt^T M dudt).

        The quantity Crank-Nicolson actually conserves, so it is a usable
        integrator diagnostic -- provided the potential term keeps its c^2 and
        the kinetic term uses the consistent mass matrix. Pairing a lumped
        kinetic term with an exact potential one makes the total swing by ~20%
        as energy sloshes between them, which is pure measurement artifact.
        '''
        return float(0.5 * (c**2 * (u @ self.K @ u) + dudt @ self.M @ dudt))

    def solve_linear_system(
        self,
        A: Operator,
        b: DofVector,
        constraints: Constraints | None = None,
    ) -> DofVector:
        '''Solve A x = b with the Dirichlet DOFs eliminated rather than penalised.

        `constraints` is a (free, fixed, fixed_values) triple and defaults to the
        solver's own boundary conditions. It is a parameter because the unknown
        is not always one value per node: the wave solver's unknown is the
        stacked block [u; du/dt], which needs its own DOF numbering.
        '''
        if constraints is None:
            bc = self.resolved_bc
            constraints = (bc.free_idxs, bc.fixed_idxs, bc.fixed_values)
        return DiscreteSystem(A, constraints).solve(b)

    def solve_steady(self) -> None:
        '''Steady linear solve A u = b.

        The three steady equations differ only in the operator: an L2 projection
        solves against the mass matrix, Poisson and elasticity against the
        stiffness. LinearElastic additionally recovers stress fields -- keyed off
        the stiffness form actually being elastic, which is the same condition.
        '''
        logger.info('Solving steady system...')
        A = self.M if isinstance(self.equation, Projection) else self.K
        u = self.solve_linear_system(A, self.b)
        self.solution.set_values("u", u)

        if isinstance(self.stiffness, LinearElasticForm):
            u_elements = u[dof_indices(self.mesh.elements, self.n_components)]
            strain, stress, compliance = self.stiffness.derived_fields(self.space.geometry, u_elements)
            self.solution.set_values("strain", np.linalg.norm(strain, axis=-1))
            self.solution.set_values("stress", np.linalg.norm(stress, axis=-1))
            self.solution.set_values("compliance", compliance)

    def solve_heat(self) -> None:
        logger.info('Solving heat equation...')  # M @ u' + K @ u = b
        #  (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        equation = self._equation_as(Heat)
        u = equation.u_initial
        dt, iters = equation.dt, equation.iters

        # The LHS (M + K*dt) is constant across steps, so factor it once and reuse
        # the factorization; only the right-hand side changes each step.
        bc = self.resolved_bc
        system = DiscreteSystem(self.M + self.K * dt, (bc.free_idxs, bc.fixed_idxs, bc.fixed_values))

        t_values: list[float] = [0.0]
        u_values = [u]
        for i in range(iters):
            u = system.solve(self.M @ u + self.b * dt)
            t_values.append(dt * (i+1))
            u_values.append(u.copy())
            logger.debug('t = %.3f, mean temp = %.3f', t_values[-1], self.space.mean_value(u_values[-1]))

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)

    def _wave_block_constraints(self, N: int) -> Constraints:
        '''Lift the nodal Dirichlet conditions onto the wave solver's [u; du/dt] block.

        Holding a node at a constant value g constrains two block DOFs, not one:
        u = g at index v, and du/dt = 0 at index N + v. Time-varying Dirichlet
        data would need a nonzero velocity there, hence the constant-in-time
        assumption baked in below.
        '''
        bc = self.resolved_bc
        fixed, free, fixed_values = bc.fixed_idxs, bc.free_idxs, bc.fixed_values

        # An initial state that disagrees with the constraints is a modelling
        # error: the solve would silently jump to satisfy them at the first step.
        equation = self._equation_as(Wave)
        u_initial = np.asarray(equation.u_initial, dtype=float)
        dudt_initial = np.asarray(equation.dudt_initial, dtype=float)
        if not np.allclose(u_initial[fixed], fixed_values):
            raise ValueError('u_initial disagrees with the Dirichlet values at fixed nodes')
        if not np.allclose(dudt_initial[fixed], 0):
            raise ValueError('dudt_initial must be zero at Dirichlet-fixed nodes')

        block_free = np.concatenate([free, N + free])
        block_fixed = np.concatenate([fixed, N + fixed])
        block_values = np.concatenate([fixed_values, np.zeros(len(fixed))])
        return block_free, block_fixed, block_values

    def solve_wave(self) -> None:
        logger.info('Solving wave equation...')  # M @ u" + K @ u = b
        equation = self._equation_as(Wave)
        u, dudt = equation.u_initial, equation.dudt_initial
        c = equation.c
        dt, iters = equation.dt, equation.iters
        N = len(self.mesh.vertices)

        # Crank-Nicolson method - average of forward and backward Euler. M and K
        # are sparse, so the 2N block system is built with block_array, not np.block.
        A_left = block_array([[self.M, -dt/2 * self.M],
                              [c**2 * dt/2 * self.K, self.M]])
        A_right = block_array([[self.M, dt/2 * self.M],
                               [-c**2 * dt/2 * self.K, self.M]])
        # The load row of Crank-Nicolson is dt/2 * (b_n + b_{n+1}). Nothing in the
        # solver makes the load time-dependent, so b_n == b_{n+1} and the average
        # collapses to dt * b. Reinstate the two-term form if loads ever vary in t.
        b_right = np.block([np.zeros_like(self.b), dt * self.b])

        # A_left is constant across steps -- factor it once behind the constraints.
        constraints = self._wave_block_constraints(N)
        system = DiscreteSystem(A_left, constraints)
        x = np.block([u, dudt])
        t_values: list[float] = [0.0]
        u_values = [x[:N]]
        dudt_values = [x[N:]]
        total_energy = self.wave_energy(u_values[-1], dudt_values[-1], c)
        logger.debug('t = %.3f, total energy = %.3f', t_values[-1], total_energy)

        for i in range(iters):
            x = system.solve(A_right @ x + b_right)
            t_values.append(dt * (i+1))
            u_values.append(x[:N])
            dudt_values.append(x[N:])
            total_energy = self.wave_energy(u_values[-1], dudt_values[-1], c)
            logger.debug('t = %.3f, total energy = %.3f', t_values[-1], total_energy)

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)
        self.solution.set_values("dudt_values", dudt_values)

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
