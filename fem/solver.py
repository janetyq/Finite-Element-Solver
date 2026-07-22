import logging
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.mesh import Mesh
from fem.boundary import BoundaryConditions
from fem.fields import FieldShape, Scalar, Vector
from fem.regions import evaluate_field
from fem.solution import Solution
from fem.space import FunctionSpace, dof_indices
from fem.forms import LaplacianForm, LinearElasticForm, strain_displacement
from fem.materials import LinearElasticMaterial
from fem.typing import (
    DofIndices,
    DofVector,
    ElementField,
    FieldValue,
    FloatArray,
    Matrix,
    VertexField,
)

# (free_idxs, fixed_idxs, fixed_values) -- the DOF partition a solve works in.
# Passed explicitly where the unknown is not one value per node, as in the wave
# solver's stacked [u; du/dt] block.
Constraints = tuple[DofIndices, DofIndices, FloatArray]

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

    def copy(self: EquationT) -> EquationT:
        # shallow copy that works regardless of subclass __init__ signature
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


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
            Projection: self.solve_projection,
            Poisson: self.solve_poisson,
            Heat: self.solve_heat,
            Wave: self.solve_wave,
            LinearElastic: self.solve_linear_elastic,
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
        if isinstance(self.equation, LinearElastic):
            self.material = LinearElasticMaterial(self.equation.E, self.equation.nu)
            self.K = self.space.assemble(LinearElasticForm(self.material))
        else:
            self.K = self.space.assemble(LaplacianForm())

        # RHS: the equation's source term over the volume, plus the boundary
        # traction over the boundary. A Robin condition would add its matrix
        # term to the LHS here, from a boundary stiffness the space would assemble.
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
        A: Matrix,
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
        free, fixed, fixed_values = constraints
        free = np.asarray(free, dtype=int)
        fixed = np.asarray(fixed, dtype=int)

        x = np.zeros_like(b)
        x[fixed] = fixed_values
        A_mod = A[np.ix_(free, free)]
        b_mod = b[free] - A[np.ix_(free, fixed)] @ x[fixed]
        x[free] = np.linalg.solve(A_mod, b_mod)
        return x

    def solve_nonlinear_system(
        self,
        A: Callable[[DofVector], Matrix],
        b: Callable[[DofVector], DofVector],
        x0: DofVector,
        tol: float = 1e-6,
        max_iters: int = 100,
    ) -> DofVector:
        # newton solver
        x = x0.copy()
        for iter in range(max_iters):
            logger.debug('newton iter %d', iter)
            dx = self.solve_linear_system(A(x), A(x) @ x - b(x))
            if np.linalg.norm(dx) < tol:
                break
            x -= dx
        return x

    def solve_projection(self) -> None:
        logger.info('Solving L2 projection...')  # M @ u = b
        u = self.solve_linear_system(self.M, self.b)
        self.solution.set_values("u", u)
    
    def solve_poisson(self) -> None:
        logger.info('Solving Poisson equation...')  # K @ u = b
        u = self.solve_linear_system(self.K, self.b)
        self.solution.set_values("u", u)

    def solve_heat(self) -> None:
        logger.info('Solving heat equation...')  # M @ u' + K @ u = b
        #  (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        equation = self._equation_as(Heat)
        u = equation.u_initial
        dt, iters = equation.dt, equation.iters

        t_values: list[float] = [0.0]
        u_values = [u]
        for i in range(iters):
            u = self.solve_linear_system(self.M + self.K * dt, self.M @ u + self.b * dt)
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

        # Crank-Nicolson method - average of forward and backward Euler
        A_left = np.block([[self.M, -dt/2 * self.M],
                           [c**2 * dt/2 * self.K, self.M]])
        A_right = np.block([[self.M, dt/2 * self.M],
                            [-c**2 * dt/2 * self.K, self.M]])
        # The load row of Crank-Nicolson is dt/2 * (b_n + b_{n+1}). Nothing in the
        # solver makes the load time-dependent, so b_n == b_{n+1} and the average
        # collapses to dt * b. Reinstate the two-term form if loads ever vary in t.
        b_right = np.block([np.zeros_like(self.b), dt * self.b])

        constraints = self._wave_block_constraints(N)
        x = np.block([u, dudt])
        t_values: list[float] = [0.0]
        u_values = [x[:N]]
        dudt_values = [x[N:]]
        total_energy = self.wave_energy(u_values[-1], dudt_values[-1], c)
        logger.debug('t = %.3f, total energy = %.3f', t_values[-1], total_energy)

        for i in range(iters):
            x = self.solve_linear_system(A_left, A_right @ x + b_right, constraints=constraints)
            t_values.append(dt * (i+1))
            u_values.append(x[:N])
            dudt_values.append(x[N:])
            total_energy = self.wave_energy(u_values[-1], dudt_values[-1], c)
            logger.debug('t = %.3f, total energy = %.3f', t_values[-1], total_energy)

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)
        self.solution.set_values("dudt_values", dudt_values)

    def solve_linear_elastic(self) -> None:
        u = self.solve_linear_system(self.K, self.b)

        eps_elements = []
        sigma_elements = []
        compliance_elements = []

        for e_idx, element in enumerate(self.mesh.elements):
            element_obj = self.space.element_objs[e_idx]
            B = strain_displacement(element_obj.grad_phi)
            D = self.material.constitutive_matrix(element_obj.reference_dim, e_idx)
            u_element = u[dof_indices(element, self.n_components)]
            eps = B @ u_element
            sigma = D @ eps
            compliance = sigma @ eps * element_obj.volume
            eps_elements.append(eps)
            sigma_elements.append(sigma)
            compliance_elements.append(compliance)

        self.solution.set_values("u", u)
        self.solution.set_values("strain", np.linalg.norm(eps_elements, axis=-1))
        self.solution.set_values("stress", np.linalg.norm(sigma_elements, axis=-1))
        self.solution.set_values("compliance", np.array(compliance_elements))

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
