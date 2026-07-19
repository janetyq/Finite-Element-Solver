import logging

import numpy as np

from fem.mesh.refinement import RefinementMesh
from fem.mesh.femesh import dof_indices
from fem.boundary import BoundaryConditions
from fem.solution import Solution
from fem.materials import Enu_to_Lame

logger = logging.getLogger(__name__)

class Equation:
    '''Base class for a PDE to solve.

    An Equation is typed data: it says *what* to solve and carries the physical
    parameters / initial conditions, while the Solver owns *how* to solve it
    (the same equation, e.g. LinearElastic, may be handled by several solvers).
    `dim` is the number of DOFs per node: 1 for scalar PDEs, 2 for 2D elasticity.
    '''
    dim = 1

    def copy(self):
        # shallow copy that works regardless of subclass __init__ signature
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new


class Projection(Equation):
    '''L2 projection of a source field onto the FE space (M u = b).'''
    dim = 1


class Poisson(Equation):
    '''Poisson equation (K u = b).'''
    dim = 1


class Heat(Equation):
    '''Transient heat equation, solved with backward Euler.'''
    dim = 1

    def __init__(self, u_initial, dt, iters):
        self.u_initial = u_initial
        self.dt = dt
        self.iters = iters


class Wave(Equation):
    '''Wave equation, solved with Crank-Nicolson.'''
    dim = 1

    def __init__(self, u_initial, dudt_initial, c, dt, iters):
        self.u_initial = u_initial
        self.dudt_initial = dudt_initial
        self.c = c
        self.dt = dt
        self.iters = iters


class LinearElastic(Equation):
    '''Small-strain linear elasticity. E may be a scalar or a per-element array
    (TopologyOptimizer sets a density-scaled modulus).'''
    dim = 2

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

class Solver:
    def __init__(self, femesh, equation, boundary_conditions=None):
        self.femesh = femesh
        self.equation = equation
        self.boundary_conditions = boundary_conditions if boundary_conditions is not None else BoundaryConditions(femesh)
        self.dim = self.equation.dim
        self.solution = Solution(femesh, self.dim)

        self.boundary_conditions.do(self.femesh.vertices.shape[0], dim=self.dim)

    def solve(self):
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

    def assemble_everything(self):
        if isinstance(self.equation, LinearElastic):
            E = np.full(len(self.femesh.elements), self.equation.E)
            nu = np.full(len(self.femesh.elements), self.equation.nu)
            mu, lamb = Enu_to_Lame(E, nu) 
            self.mu, self.lamb = mu, lamb
            self.femesh.prepare_matrices(dim=self.dim, mu=mu, lamb=lamb)
        else:
            self.femesh.prepare_matrices(dim=self.dim)

        self.b = (self.femesh.M @ self.boundary_conditions.force_load.flatten()).flatten()
        self.b += (self.femesh.M_b @ self.boundary_conditions.neumann_load.flatten()).flatten()

    def solve_linear_system(self, A, b, use_bc=True):
        # solves Au=b, taking fixed vars and loads into account
        x = np.zeros_like(b)
        if use_bc:
            free = self.boundary_conditions.free_idxs
            fixed = self.boundary_conditions.fixed_idxs
            x[fixed] = self.boundary_conditions.fixed_values
            A_mod = A[np.ix_(free, free)]
            b_mod = b[free] - A[np.ix_(free, fixed)] @ x[fixed]
            x[free] = np.linalg.solve(A_mod, b_mod)
            return x
        else:
            return np.linalg.solve(A, b)

    def solve_nonlinear_system(self, A, b, x0, tol=1e-6, max_iters=100):
        # newton solver
        x = x0.copy()
        for iter in range(max_iters):
            logger.debug('newton iter %d', iter)
            dx = self.solve_linear_system(A(x), A(x) @ x - b(x))
            if np.linalg.norm(dx) < tol:
                break
            x -= dx
        return x

    def solve_projection(self):
        logger.info('Solving L2 projection...')  # M @ u = b
        u = self.solve_linear_system(self.femesh.M, self.b)
        self.solution.set_values("u", u)
    
    def solve_poisson(self):
        logger.info('Solving Poisson equation...')  # K @ u = b
        u = self.solve_linear_system(self.femesh.K, self.b)
        self.solution.set_values("u", u)

    def solve_heat(self):
        logger.info('Solving heat equation...')  # M @ u' + K @ u = b
        #  (M + K*dt) @ u_{n+1} = M @ u_n + b*dt, backwards Euler
        u = self.equation.u_initial
        dt, iters = self.equation.dt, self.equation.iters

        t_values = [0]
        u_values = [u]
        for i in range(iters):
            u = self.solve_linear_system(self.femesh.M + self.femesh.K * dt, self.femesh.M @ u + self.b * dt)
            t_values.append(dt * (i+1))
            u_values.append(u.copy())
            logger.debug('t = %.3f, mean temp = %.3f', t_values[-1], self.femesh.calculate_mean_value(u_values[-1]))

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)

    def solve_wave(self):
        logger.info('Solving wave equation...')  # M @ u" + K @ u = b
        # The time-stepping below runs with use_bc=False, so Dirichlet
        # constraints would be silently ignored. Fail loudly instead of
        # returning a solution that doesn't satisfy them.
        if self.boundary_conditions.dirichlet:
            raise NotImplementedError('solve_wave does not honor Dirichlet boundary conditions yet')
        u, dudt = self.equation.u_initial, self.equation.dudt_initial
        c = self.equation.c
        dt, iters = self.equation.dt, self.equation.iters
        
        # Crank-Nicolson method - average of forward and backward Euler
        A_left = np.block([[self.femesh.M, -dt/2 * self.femesh.M],
                           [c**2 * dt/2 * self.femesh.K, self.femesh.M]])
        A_right = np.block([[self.femesh.M, dt/2 * self.femesh.M],
                            [-c**2 * dt/2 * self.femesh.K, self.femesh.M]])
        # NOTE: np.roll(self.b, -1) rolls a *spatial* load vector as if it were a
        # time series -- harmless while self.b is zero, but a latent bug the moment
        # a real source term is added. See BACKLOG.md section 1.
        b_right = np.block([np.zeros_like(self.b), dt/2 * (self.b + np.roll(self.b, -1))])

        N = len(self.femesh.vertices)
        x = np.block([u, dudt])
        t_values = [0]
        u_values = [x[:N]]
        dudt_values = [x[N:]]
        total_energy = self.femesh.calculate_energy(u_values[-1], dudt_values[-1])
        logger.debug('t = %.3f, total energy = %.3f', t_values[-1], total_energy)

        for i in range(iters):
            x = self.solve_linear_system(A_left, A_right @ x + b_right, use_bc=False) # TODO: bc not supported for wave
            t_values.append(dt * (i+1))
            u_values.append(x[:N])
            dudt_values.append(x[N:])
            total_energy = self.femesh.calculate_energy(u_values[-1], dudt_values[-1])
            logger.debug('t = %.3f, total energy = %.3f', t_values[-1], total_energy)

        self.solution.set_values("t_values", t_values)
        self.solution.set_values("u_values", u_values)
        self.solution.set_values("dudt_values", dudt_values)

    def solve_linear_elastic(self):
        u = self.solve_linear_system(self.femesh.K, self.b)

        eps_elements = []
        sigma_elements = []
        compliance_elements = []

        for e_idx, element in enumerate(self.femesh.elements):
            element_obj = self.femesh.element_objs[e_idx]
            B = element_obj.calculate_B()
            D = element_obj.calculate_D(self.mu[e_idx], self.lamb[e_idx])
            u_element = u[dof_indices(element, self.dim)]
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

    def adaptive_refinement(self, max_triangles=1000, max_iters=20):
        # TODO: there's a bug somewhere
        if 'element_residuals' not in self.solution.values:
            raise ValueError('No element residuals found in solution')
        refinement_mesh = RefinementMesh(self.femesh)
        while len(self.femesh.elements) < max_triangles or max_iters == 0:
            element_residuals = self.solution.values['element_residuals']
            max_residual = max(element_residuals)
            refine_idxs = []
            for e_idx, residual in enumerate(element_residuals):
                if residual > 0.9 * max_residual:
                    refine_idxs.append(e_idx)

            refinement_mesh.refine_triangles(refine_idxs)
            self.femesh = refinement_mesh.get_mesh()
            max_iters -= 1

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
    #         residuals = np.zeros(len(self.femesh.elements))
    #         for e_idx, element in enumerate(self.femesh.elements):
    #             residuals[e_idx] = 0 # placeholder
    #         self.solution.set_values("apriori_residual", residuals)
    #     else:
    #         # compute aposteriori residual
    #         residuals = None
    #         self.solution.set_values("aposteriori_residual", residuals)
    #         pass
