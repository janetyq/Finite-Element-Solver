"""Time integrators: a scheme applied to a semi-discrete `Problem`.

The domain has two ODE orders -- heat is first (M u' + K u = b), wave is second
(M u'' + K u = b) -- so there is one integrator family per order rather than one
first-order interface for both. Each forms a *constant* effective operator from the
problem's mass and stiffness, factors it once through `DiscreteSystem`, and steps
by updating only the right-hand side. `dt` and the step count live here, not on the
equation; initial conditions come in through `run`.

The wave path uses Newmark, not a 2N first-order block: its effective operator
`M + β dt² c²K` is SPD and N-sized, so it stays inside the CG/preconditioning story
and needs no lifting of Dirichlet indices into a block DOF space.
"""
import numpy as np

from fem.problem import Problem
from fem.solution import Solution
from fem.system import DiscreteSystem
from fem.typing import DofVector


def _history(problem: Problem, t_values: list[float], u_values: list[DofVector],
             dudt_values: list[DofVector] | None = None) -> Solution:
    '''Package a time series into a Solution (the transient result container).'''
    solution = Solution(problem.space.mesh, problem.space.n_components)
    solution.set_values("t_values", t_values)
    solution.set_values("u_values", u_values)
    if dudt_values is not None:
        solution.set_values("dudt_values", dudt_values)
    return solution


class ThetaMethod:
    '''First-order integrator for M u' + K u = b.

    θ = ½ is Crank–Nicolson (second-order accurate, the default); θ = 1 is backward
    Euler. The step is (M + θ dt K) u_{n+1} = (M − (1−θ) dt K) u_n + dt b, whose LHS
    is constant, so it is factored once and reused.
    '''

    def __init__(self, dt: float, steps: int, theta: float = 0.5) -> None:
        self.dt = dt
        self.steps = steps
        self.theta = theta

    def run(self, problem: Problem, u0: DofVector) -> Solution:
        M = problem.space.mass_matrix
        K = problem.tangent(None)
        b = problem.load
        dt, theta = self.dt, self.theta

        system = DiscreteSystem(M + theta * dt * K, problem.constraints)
        rhs_operator = M - (1 - theta) * dt * K

        u = np.asarray(u0, dtype=float)
        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        for i in range(self.steps):
            u = system.solve(rhs_operator @ u + dt * b)
            t_values.append(dt * (i + 1))
            u_values.append(u.copy())
        return _history(problem, t_values, u_values)


class Newmark:
    '''Second-order integrator for M u'' + K u = b.

    β = ¼, γ = ½ is the average-acceleration scheme: unconditionally stable and, for
    a linear undamped system, energy-conserving. It solves for the acceleration
    against the SPD operator M + β dt² K, an N-sized system factored once. Constant
    Dirichlet displacement means zero velocity and acceleration at the fixed nodes,
    so those DOFs are pinned to zero in the acceleration solve -- the ordinary
    constraint, no lifting into a 2N block.
    '''

    def __init__(self, dt: float, steps: int, beta: float = 0.25, gamma: float = 0.5) -> None:
        self.dt = dt
        self.steps = steps
        self.beta = beta
        self.gamma = gamma

    def run(self, problem: Problem, u0: DofVector, v0: DofVector) -> Solution:
        M = problem.space.mass_matrix
        K = problem.tangent(None)  # already c²K for the wave factory
        b = problem.load
        free, fixed, fixed_values = problem.constraints

        u = np.asarray(u0, dtype=float)
        v = np.asarray(v0, dtype=float)
        # An initial state that disagrees with the constraints is a modelling error:
        # the solve would otherwise jump to satisfy them at the first step.
        if not np.allclose(u[fixed], fixed_values):
            raise ValueError('u0 disagrees with the Dirichlet values at fixed nodes')
        if not np.allclose(v[fixed], 0):
            raise ValueError('v0 must be zero at Dirichlet-fixed nodes')

        dt, beta, gamma = self.dt, self.beta, self.gamma
        accel_constraints = (free, fixed, np.zeros(len(fixed)))

        # Initial acceleration from M a0 = b − K u0, pinned to zero at fixed DOFs.
        a = DiscreteSystem(M, accel_constraints).solve(b - K @ u)
        effective = DiscreteSystem(M + beta * dt**2 * K, accel_constraints)

        t_values: list[float] = [0.0]
        u_values: list[DofVector] = [u.copy()]
        dudt_values: list[DofVector] = [v.copy()]
        for i in range(self.steps):
            u_pred = u + dt * v + dt**2 / 2 * (1 - 2 * beta) * a
            v_pred = v + dt * (1 - gamma) * a
            a = effective.solve(b - K @ u_pred)
            u = u_pred + beta * dt**2 * a
            v = v_pred + gamma * dt * a
            t_values.append(dt * (i + 1))
            u_values.append(u.copy())
            dudt_values.append(v.copy())
        return _history(problem, t_values, u_values, dudt_values)


def wave_energy(problem: Problem, u: DofVector, v: DofVector) -> float:
    '''Total wave energy ½(uᵀ K u + vᵀ M v), with K the c²-scaled stiffness.

    The quantity average-acceleration Newmark conserves for a linear system, so it
    is a usable integrator diagnostic. The consistent mass matrix is load-bearing:
    pairing a lumped kinetic term with the exact potential one makes the total swing
    as energy sloshes between them, a pure measurement artifact.
    '''
    M = problem.space.mass_matrix
    K = problem.tangent(None)
    return float(0.5 * (u @ K @ u + v @ M @ v))
