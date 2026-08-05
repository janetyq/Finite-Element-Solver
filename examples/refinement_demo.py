"""Demo: adaptive mesh refinement driven by an a posteriori error estimator.

Run via the shared CLI:

    uv run python examples/cli.py run refinement
"""
from math import e

import numpy as np

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.equations import Poisson
from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.ruppert import create_rect_mesh
from fem.plot.plotter import Plotter
from fem.regions import everywhere
from fem.solver import Solver

from demo_registry import Demo, DemoResult, Figure
from domains import square


def demo_refinement(_mesh):
    """Adaptive refinement on a peaked Poisson source: the error estimator tells
    the refiner which elements need splitting, concentrating the mesh where the
    solution is hardest to approximate."""
    # Start with a moderate mesh so there is room to refine
    mesh = create_rect_mesh(corners=[[0.0, 0.0], [1.0, 1.0]], resolution=(20, 20))
    w, h = 1.0, 1.0

    def peaked_source(point):
        a = 50
        x, y = point - np.array([w/2, h/2])
        r2 = x**2 + y**2
        return 4*a*a*(1-a*r2)*e**(-a*r2)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0)
    equation = Poisson(source=peaked_source)

    # Solve on the initial coarse mesh and compute error estimate
    coarse_solver = Solver(mesh.copy(), equation, bc)
    coarse_solution = coarse_solver.solve()
    coarse_mesh = coarse_solver.mesh
    coarse_error = equation.error_estimate(coarse_solver)
    n_coarse = len(coarse_mesh.elements)

    # Run adaptive refinement driven by the residual error estimator
    refined_solver = Solver(mesh.copy(), equation, bc)
    refined_solution = AdaptiveRefinement(
        refined_solver,
        equation.error_estimate,
        max_triangles=3000,
        max_iters=20,
    ).run()
    refined_mesh = refined_solver.mesh
    refined_error = equation.error_estimate(refined_solver)
    n_refined = len(refined_mesh.elements)

    # Use a shared log scale for both error plots so they're comparable
    min_error = min(coarse_error.min(), refined_error.min())
    max_error = max(coarse_error.max(), refined_error.max())
    error_clim = (max(min_error, 1e-10), max_error)

    # Compute error statistics for display
    coarse_max = coarse_error.max()
    coarse_norm = float(np.sqrt(np.sum(coarse_error**2)))
    refined_max = refined_error.max()
    refined_norm = float(np.sqrt(np.sum(refined_error**2)))
    max_reduction = 100 * (1 - refined_max / coarse_max)
    norm_reduction = 100 * (1 - refined_norm / coarse_norm)

    before = Plotter(1, 3, title=f'Before: uniform mesh ({n_coarse} elements)')
    before.plot(coarse_mesh, coarse_solution.u, mode='surface', title='Solution', idx=(0, 0))
    before.plot(coarse_mesh, mode='mesh', title='Mesh', idx=(0, 1))
    before.plot(coarse_mesh, coarse_error, mode='colored',
                title=f'Error η (max: {coarse_max:.3f}, ‖η‖: {coarse_norm:.2f})',
                idx=(0, 2), clim=error_clim, cmap='YlOrRd', log_scale=True)

    after = Plotter(1, 3, title=f'After: adaptive refinement ({n_refined} elements)')
    after.plot(refined_mesh, refined_solution.u, mode='surface', title='Solution', idx=(0, 0))
    after.plot(refined_mesh, mode='mesh', title='Mesh', idx=(0, 1))
    after.plot(refined_mesh, refined_error, mode='colored',
               title=f'Error η (max: {refined_max:.3f}, ‖η‖: {refined_norm:.2f})',
               idx=(0, 2), clim=error_clim, cmap='YlOrRd', log_scale=True)

    # Show what red-green splitting does to an element — the mechanism beneath
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    elements = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    small = Mesh(vertices, elements, boundary)
    original = small.copy()

    refiner = RedGreenRefiner(small)
    small = refiner.refine([0, 2])
    small = refiner.refine([1])

    machinery = Plotter(1, 2, title='Red-green refinement', axis_labels=False)
    machinery.plot(original, mode='mesh', idx=(0, 0), title='Original')
    machinery.plot(small, values=refiner.leaf_classifications(), mode='refinement',
                   idx=(0, 1), title='Refined (red / green)')

    return DemoResult([
        Figure(before,
               'Uniform mesh with a posteriori error estimate η. The estimator bounds '
               'the local discretization error; high values (red) indicate where the '
               'mesh under-resolves the solution.',
               'before'),
        Figure(after,
               f'After adaptive refinement driven by η: max error dropped {max_reduction:.0f}% '
               f'({coarse_max:.3f} → {refined_max:.3f}), ‖η‖ dropped {norm_reduction:.0f}% '
               f'({coarse_norm:.2f} → {refined_norm:.2f}).',
               'after'),
        Figure(machinery,
               'Red splits an element into four; green bisects a neighbour so '
               'the mesh stays conforming.',
               'red-green'),
    ])


DEMOS = [
    Demo('refinement', demo_refinement, section='Accuracy & performance', domain=square),
]
