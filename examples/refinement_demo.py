"""Demo: adaptive mesh refinement driven by an a posteriori error estimator.

Run via the shared CLI:

    uv run python examples/cli.py run refinement
"""
from math import e

import numpy as np

from fem.adaptivity import AdaptiveRefinement
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import l2_norm
from fem.equations import Poisson
from fem.estimators import residual_estimator
from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.structured import create_rect_mesh
from fem.plot.plotter import Plotter
from fem.regions import everywhere
from fem.solver import Solver

from demo_registry import Demo, DemoResult, Figure
from domains import square


def demo_refinement(_mesh, uniform_resolutions=(10, 20, 40, 80, 160), adaptive_rounds=30):
    """Adaptive refinement driven by an error estimator on a peaked Poisson source,
    against uniform refinement at the same cost."""
    mesh = create_rect_mesh(corners=[[0.0, 0.0], [1.0, 1.0]], resolution=(20, 20))
    w, h = 1.0, 1.0
    a = 300      # the peak's sharpness: its width is about 1/sqrt(2a)

    # The source is -laplacian of a * exp(-a r^2), which is within 2e-4 of zero on the
    # boundary, so that peak is the exact solution to the precision this chart needs.
    def peaked_source(point):
        x, y = point - np.array([w/2, h/2])
        r2 = x**2 + y**2
        return 4*a*a*(1-a*r2)*e**(-a*r2)

    def exact(points):
        r2 = np.sum((points - np.array([w/2, h/2]))**2, axis=1)
        return a * np.exp(-a * r2)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0)
    equation = Poisson(source=peaked_source)

    coarse_solver = Solver(mesh.copy(), equation, bc)
    coarse_solution = coarse_solver.solve()
    coarse_mesh = coarse_solver.mesh
    estimator = residual_estimator(equation)
    coarse_error = estimator.estimate(coarse_solver)
    n_coarse = len(coarse_mesh.elements)

    refined_solver = Solver(mesh.copy(), equation, bc)
    refined_solution = AdaptiveRefinement(
        refined_solver,
        estimator,
        max_triangles=3000,
        max_iters=20,
        refine_fraction=0.5,
    ).run()
    refined_mesh = refined_solver.mesh
    refined_error = estimator.estimate(refined_solver)
    n_refined = len(refined_mesh.elements)

    # A shared log scale for both error plots.
    min_error = min(coarse_error.min(), refined_error.min())
    max_error = max(coarse_error.max(), refined_error.max())
    error_clim = (max(min_error, 1e-10), max_error)

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

    # The payoff: error against cost, refining uniformly and adaptively. Each adaptive
    # round refines the worst elements once and re-solves; each point is one round.
    def error_of(solver, solution):
        return l2_norm(solver.space, solution.u - exact(solver.space.node_coords))

    uniform_dofs, uniform_errors = [], []
    for n in uniform_resolutions:
        solver = Solver(create_rect_mesh(corners=[[0.0, 0.0], [1.0, 1.0]],
                                         resolution=(n, n)), equation, bc)
        uniform_dofs.append(solver.space.n_dofs)
        uniform_errors.append(error_of(solver, solver.solve()))

    adaptive_solver = Solver(mesh.copy(), equation, bc)
    adaptive_dofs = [adaptive_solver.space.n_dofs]
    adaptive_errors = [error_of(adaptive_solver, adaptive_solver.solve())]
    for _ in range(adaptive_rounds):
        solution = AdaptiveRefinement(adaptive_solver, estimator, max_triangles=10**9,
                                      max_iters=1, refine_fraction=0.5).run()
        if adaptive_solver.space.n_dofs > max(uniform_dofs) // 2:
            break
        adaptive_dofs.append(adaptive_solver.space.n_dofs)
        adaptive_errors.append(error_of(adaptive_solver, solution))

    payoff = Plotter(title='Error against cost')
    chart = payoff.chart_ax(xlabel='degrees of freedom', ylabel='L2 error')
    chart.loglog(uniform_dofs, uniform_errors, 'o-', color='tab:blue', label='uniform')
    chart.loglog(adaptive_dofs, adaptive_errors, '.-', color='tab:red', label='adaptive')
    chart.set_title('Uniform against adaptive refinement')
    chart.grid(True, which='both', alpha=0.3)
    chart.legend()

    # What red-green splitting does to an element.
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
        Figure(payoff,
               'The error against the number of unknowns, refining the whole mesh (blue) '
               'and refining where the estimator points (red). The peak is small next to '
               'the domain, so uniform refinement spends most of its unknowns where the '
               'solution is already flat; the adaptive mesh reaches the same error with '
               'about a third as many.',
               'payoff'),
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
    Demo('refinement', demo_refinement, section='Accuracy & performance', domain=square,
         smoke_kwargs={'uniform_resolutions': (10, 20), 'adaptive_rounds': 2}),
]
