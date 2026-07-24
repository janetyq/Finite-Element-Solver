"""Integration smoke tests: run the solver paths that have no dedicated
correctness test, so a call-time error (e.g. a bad import after refactoring)
surfaces here rather than only when a human next runs the feature.

These assert "runs and produces finite output", not detailed correctness —
that is the job of the other test modules.
"""
import numpy as np

from fem.numerics import bump_function
from fem.boundary import BoundaryConditions, BCType
from fem.regions import on_plane
from fem.solver import LinearElastic
from fem.problem import wave
from fem.integrators import Newmark
from fem.energy_solver import EnergySolver
from fem.topology import TopologyOptimizer
from fem.mesh.refinement import RedGreenRefiner


def test_wave_solver_runs(make_unit_square):
    mesh = make_unit_square(15)
    u0 = bump_function(mesh.vertices, mesh.vertices.mean(axis=0), mag=1, size=0.3)
    problem = wave(mesh, c=1)
    solution = Newmark(dt=0.02, steps=5).run(problem, u0, np.zeros(len(mesh.vertices)))
    u_values = solution.get_values("u_values")
    assert len(u_values) == 6
    assert np.all(np.isfinite(u_values[-1]))


def test_energy_solver_runs(make_unit_square):
    mesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [0.1, 0])

    eq = LinearElastic(E=200, nu=0.4)
    solution = EnergySolver(mesh, eq, bc, verbose=False).solve()
    assert np.all(np.isfinite(solution.get_values("u")))


def test_topology_optimizer_runs(make_unit_square):
    mesh = make_unit_square(12)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    eq = LinearElastic(E=200, nu=0.4, source=[0, -0.5])
    topopt = TopologyOptimizer(mesh, eq, bc, iters=2, volume_frac=0.5)
    topopt.solve()
    assert np.all(np.isfinite(topopt.rho))
    assert topopt.rho.min() >= 0.0


def test_bc_plotting_runs(make_unit_square):
    """plot_bc reads the spec through entries(), which resolves regions without
    needing a component count; exercise it so a break surfaces here, not on a human's screen."""
    import matplotlib.pyplot as plt

    from fem.plot.helpers import plot_bc

    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])

    _fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    plt.close(_fig)


def test_refinement_increases_element_count(make_unit_square):
    mesh = make_unit_square(6)
    n_before = len(mesh.elements)
    refiner = RedGreenRefiner(mesh)
    refined = refiner.refine([0, 1, 2])
    assert len(refined.elements) > n_before
