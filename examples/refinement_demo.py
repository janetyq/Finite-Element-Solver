"""Demo: why you would refine a mesh, and the machinery that does it.

Run via the shared CLI:

    uv run python examples/cli.py run refinement
"""
import random
from math import e

import numpy as np

from fem.boundary import BoundaryConditions, BCType
from fem.equations import Poisson
from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.plot.plotter import Plotter
from fem.regions import everywhere
from fem.solver import Solver

from demo_registry import Demo, DemoResult, Figure
from domains import square


def demo_refinement(mesh):
    """Show the problem adaptive refinement exists for, and the red-green splitting that
    would answer it -- the two halves of a loop that is not yet closed."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])

    def peaked_source(point):
        a = 50
        x, y = point - np.array([w/2, h/2])
        r2 = x**2 + y**2
        return [4*a*a*(1-a*r2)*e**(-a*r2)] # TODO: list thing is awkward

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0)
    solver = Solver(mesh, Poisson(source=peaked_source), bc)
    solution = solver.solve()

    problem = Plotter(1, 2, title='The case for refining: one sharp interior peak')
    problem.plot(mesh, solution.u, mode='surface', title='Poisson solution', idx=(0, 0))
    problem.plot(mesh, solver.space.gradient(solution.u), mode='arrows',
                 title='Gradient', idx=(0, 1))

    # Red-green on its own small mesh rather than on the one above: the point here is
    # what the refiner does to a triangle and to its neighbours, which is legible at
    # four elements and not at sixteen hundred.
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    elements = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    small = Mesh(vertices, elements, boundary)
    original = small.copy()

    refiner = RedGreenRefiner(small)
    for _ in range(8):
        refine_list = set(random.randint(0, len(small.elements) - 1) for _ in range(5))
        small = refiner.refine(refine_list)

    machinery = Plotter(1, 2, title='Red-green refinement', axis_labels=False)
    machinery.plot(original, mode='mesh', idx=(0, 0), title='Original')
    machinery.plot(small, values=refiner.leaf_classifications(), mode='refinement',
                   idx=(0, 1), title='Refined (red / green)')

    # `AdaptiveRefinement(solver, estimator).run()` drives the loop between these two
    # figures correctly, and is covered by tests/test_refinement.py. What is missing is
    # the estimator: without one there is nothing to tell the refiner *which* elements
    # sit under the peak, so the demo shows the two ends and not the join. See BACKLOG.md.
    return DemoResult([
        Figure(problem,
               'A Poisson source with one sharp interior peak. A uniform mesh spends most '
               'of its elements far from the peak, resolving nothing, and too few on it.',
               'problem'),
        Figure(machinery,
               'Eight rounds of refinement on randomly chosen elements. Red splits an '
               'element into four; green bisects a neighbour so the mesh stays conforming. '
               'Driving this from the error above needs an estimator, which is the piece '
               'still open.',
               'machinery'),
    ])


DEMOS = [
    Demo('refinement', demo_refinement, section='Accuracy & performance', domain=square),
]
