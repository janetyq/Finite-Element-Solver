"""Temporal convergence of the transient heat solver.

The spatial FEM discretization gives the semi-discrete system  M u' = -K u, whose
exact solution is  u(t) = expm(-t M^{-1} K) u0  on the free (interior) DOFs. The
time-stepper is compared against *that* -- not the continuous PDE -- so the error
is purely temporal: no spatial discretization error contaminates the observed
order, and the mesh can stay coarse (the exp is dense).

Backward Euler is first order in dt, so halving dt halves the error. This is the
safety net for the time-integrator refactor: the scheme changes, but a scheme's
temporal order is a fact the net pins down rather than adjusts to fit.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from fem.boundary import BoundaryConditions, BCType
from fem.mesh.generation import create_rect_mesh
from fem.regions import everywhere
from fem.solver import Heat, Solver


def _semidiscrete_exact(solver, u0, T):
    """Exact solution of the semi-discrete system M u' = -K u at time T.

    Solved on the free DOFs (the fixed ones are held at 0 by the homogeneous
    Dirichlet data), so it is the target a time-stepper approximates with no
    spatial error of its own.
    """
    free = solver.resolved_bc.free_idxs
    M = solver.space.mass_matrix.toarray()
    K = solver.K.toarray()
    M_ff, K_ff = M[np.ix_(free, free)], K[np.ix_(free, free)]
    propagator = expm(-T * np.linalg.solve(M_ff, K_ff))
    u_exact = np.zeros_like(u0)
    u_exact[free] = propagator @ u0[free]
    return u_exact


def _heat_temporal_error(mesh, bc, u0, T, n_steps):
    """M-weighted L2 error of backward Euler against the semi-discrete exact at T."""
    eq = Heat(u_initial=u0.copy(), iters=n_steps, dt=T / n_steps)
    solver = Solver(mesh, eq, bc)
    u_h = solver.solve().get_values("u_values")[-1]
    error = u_h - _semidiscrete_exact(solver, u0, T)
    return float(np.sqrt(error @ solver.space.mass_matrix @ error))


@pytest.fixture(scope="module")
def heat_temporal_data():
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(11, 11))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    x, y = mesh.vertices[:, 0], mesh.vertices[:, 1]
    u0 = np.sin(np.pi * x) * np.sin(np.pi * y)  # an eigenmode; zero on the boundary
    T = 0.02  # decays to ~67% of u0, so the dynamics are non-trivial

    n_steps = [2, 4, 8, 16]
    dts = [T / k for k in n_steps]
    errors = [_heat_temporal_error(mesh, bc, u0, T, k) for k in n_steps]
    return dts, errors


def test_heat_temporal_error_decreases(heat_temporal_data):
    _, errors = heat_temporal_data
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse, f"error grew under dt refinement: {errors}"


def test_heat_backward_euler_is_first_order(heat_temporal_data):
    dts, errors = heat_temporal_data
    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(dts) - 1)
    ]
    # Backward Euler is O(dt); allow a band for the coarse-step regime.
    for p in orders:
        assert 0.8 < p < 1.3, f"expected ~1st order in dt, got orders {orders}"
