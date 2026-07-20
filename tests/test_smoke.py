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
from fem.solver import Solver, Wave, LinearElastic
from fem.energy_solver import EnergySolver
from fem.topology import TopologyOptimizer
from fem.mesh.refinement import RefinementMesh


def test_wave_solver_runs(make_unit_square):
    femesh = make_unit_square(15)
    u0 = bump_function(femesh.vertices, femesh.vertices.mean(axis=0), mag=1, size=0.3)
    eq = Wave(
        u_initial=u0,
        dudt_initial=np.zeros(len(femesh.vertices)),
        c=1, dt=0.02, iters=5,
    )
    solution = Solver(femesh, eq).solve()
    u_values = solution.get_values("u_values")
    assert len(u_values) == 6
    assert np.all(np.isfinite(u_values[-1]))


def test_energy_solver_runs(make_unit_square):
    femesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [0.1, 0])

    eq = LinearElastic(E=200, nu=0.4)
    solution = EnergySolver(femesh, eq, bc, verbose=False).solve()
    assert np.all(np.isfinite(solution.get_values("u")))


def test_topology_optimizer_runs(make_unit_square):
    femesh = make_unit_square(12)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    eq = LinearElastic(E=200, nu=0.4, source=[0, -0.5])
    topopt = TopologyOptimizer(femesh, eq, bc, iters=2, volume_frac=0.5)
    topopt.solve()
    assert np.all(np.isfinite(topopt.rho))
    assert topopt.rho.min() >= 0.0


def test_bc_plotting_runs(make_unit_square):
    """plot_bc reads the spec through entries(), which resolves regions without
    needing a dim; exercise it so a break surfaces here, not on a human's screen."""
    import matplotlib.pyplot as plt

    from fem.plot.helpers import plot_bc

    femesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])

    _fig, ax = plt.subplots()
    plot_bc(ax, femesh, bc)
    plt.close(_fig)


def test_refinement_increases_element_count(make_unit_square):
    femesh = make_unit_square(6)
    n_before = len(femesh.elements)
    refiner = RefinementMesh(femesh)
    refiner.refine_triangles([0, 1, 2])
    refined = refiner.get_mesh()
    assert len(refined.elements) > n_before
