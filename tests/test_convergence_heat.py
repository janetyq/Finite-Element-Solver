"""Temporal convergence of the heat time integrators.

The spatial FEM discretization gives the semi-discrete system  M u' = -K u, whose
exact solution is  u(t) = expm(-t M^{-1} K) u0  on the free (interior) DOFs. The
integrator is compared against *that* -- not the continuous PDE -- so the error is
purely temporal: no spatial discretization error contaminates the observed order,
and the mesh can stay coarse (the exp is dense).

theta = 1 (backward Euler) is first order; theta = 1/2 (Crank-Nicolson, the
default) is second. Both are pinned here -- a scheme's temporal order is a fact the
net asserts rather than adjusts to fit.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from fem.boundary import BoundaryConditions, BCType
from fem.integrators import ThetaMethod
from fem.mesh.generation import create_rect_mesh
from fem.problem import heat
from fem.regions import everywhere


def _semidiscrete_exact(problem, u0, T):
    """Exact solution of the semi-discrete system M u' = -K u at time T."""
    free = problem.constraints[0]
    M = problem.space.mass_matrix.toarray()
    K = problem.tangent(None).toarray()
    M_ff, K_ff = M[np.ix_(free, free)], K[np.ix_(free, free)]
    propagator = expm(-T * np.linalg.solve(M_ff, K_ff))
    u_exact = np.zeros_like(u0)
    u_exact[free] = propagator @ u0[free]
    return u_exact


def _temporal_error(mesh, bc, u0, T, n_steps, theta):
    """M-weighted L2 error of a theta-method against the semi-discrete exact at T."""
    problem = heat(mesh, bc=bc)
    integrator = ThetaMethod(dt=T / n_steps, steps=n_steps, theta=theta)
    u_h = integrator.run(problem, u0.copy()).get_values("u_values")[-1]
    error = u_h - _semidiscrete_exact(problem, u0, T)
    return float(np.sqrt(error @ problem.space.mass_matrix @ error))


def _orders(mesh, bc, u0, T, theta, n_steps):
    dts = [T / k for k in n_steps]
    errors = [_temporal_error(mesh, bc, u0, T, k, theta) for k in n_steps]
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f"error grew under dt refinement: {errors}"
    return [
        np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(dts) - 1)
    ]


@pytest.fixture(scope="module")
def setup():
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(11, 11))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    x, y = mesh.vertices[:, 0], mesh.vertices[:, 1]
    u0 = np.sin(np.pi * x) * np.sin(np.pi * y)  # an eigenmode; zero on the boundary
    return mesh, bc, u0


def test_backward_euler_is_first_order(setup):
    mesh, bc, u0 = setup
    orders = _orders(mesh, bc, u0, T=0.02, theta=1.0, n_steps=[2, 4, 8, 16])
    for p in orders:
        assert 0.8 < p < 1.3, f"expected ~1st order in dt, got {orders}"


def test_crank_nicolson_is_second_order(setup):
    # Steps chosen to sit in the asymptotic band: coarser steps are pre-asymptotic
    # (lambda*dt not small), finer ones approach the roundoff floor of the expm.
    mesh, bc, u0 = setup
    orders = _orders(mesh, bc, u0, T=0.02, theta=0.5, n_steps=[16, 32, 64])
    for p in orders:
        assert 1.8 < p < 2.3, f"expected ~2nd order in dt, got {orders}"
