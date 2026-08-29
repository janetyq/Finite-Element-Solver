"""Adaptive mesh refinement driven by an a posteriori error estimator, on a Poisson
problem whose solution is a sharp peak in the middle of a unit square.

`adapt`, `uniform_sweep`, and `adaptive_sweep` each pose and solve one study; `run`
calls them and returns a `RefinementStudy` of plain results. Nothing here draws:
`figures.py` does that from the `RefinementStudy`, and this file is what the gallery
shows.
"""
from dataclasses import dataclass
from math import e

import numpy as np

from fem.analysis.adaptivity import AdaptiveRefinement
from fem.boundary import Dirichlet
from fem.conditions import Conditions
from mms import l2_norm
from fem.physics.equations import Poisson
from fem.analysis.estimators import ResidualEstimator
from fem.mesh.mesh import Mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.structured import box_mesh
from fem.regions import everywhere
from fem.post.solution import DiffusionSolution
from fem.loads import Source

W, H = 1.0, 1.0
A = 300      # the peak's sharpness: its width is about 1/sqrt(2a)
CENTRE = np.array([W/2, H/2])


# The source is -laplacian of a * exp(-a r^2), which is within 2e-4 of zero on the
# boundary, so that peak is the exact solution to the precision this chart needs.
def peaked_source(point):
    x, y = point - CENTRE
    r2 = x**2 + y**2
    return 4*A*A*(1-A*r2)*e**(-A*r2)


def exact(points):
    r2 = np.sum((points - CENTRE)**2, axis=1)
    return A * np.exp(-A * r2)


bc = Conditions(Dirichlet(everywhere(), 0), Source(peaked_source))
equation = Poisson()
estimator = ResidualEstimator()


def square_mesh(n) -> Mesh:
    return box_mesh(corners=[[0.0, 0.0], [W, H]], resolution=(n, n))


def problem_for(m):
    return equation.problem(m, bc)


def solve(m):
    problem = problem_for(m)
    return problem, problem.solve()


def error_of(solution) -> float:
    """The L2 error of `solution` against the exact peak."""
    space = solution.space
    return l2_norm(space, solution.u - exact(space.node_coords))


def adapt(mesh, max_triangles=3000, max_iters=20) -> tuple[Mesh, DiffusionSolution, np.ndarray]:
    """Refine `mesh` where the estimator points, half the worst elements each round,
    until the mesh reaches `max_triangles` or `max_iters` rounds. Returns the refined
    mesh, its solution, and the estimated error per element."""
    refinement = AdaptiveRefinement(
        mesh,
        problem_for,
        estimator,
        max_triangles=max_triangles,
        max_iters=max_iters,
        refine_fraction=0.5,
    )
    solution = refinement.run()
    assert refinement.problem is not None
    return refinement.mesh, solution, estimator.estimate(refinement.problem, solution)


def uniform_sweep(resolutions) -> tuple[list[int], list[float]]:
    """The L2 error against unknowns, refining the whole square uniformly."""
    dofs, errors = [], []
    for n in resolutions:
        _, solution = solve(square_mesh(n))
        dofs.append(solution.space.n_dofs)
        errors.append(error_of(solution))
    return dofs, errors


def adaptive_sweep(mesh, rounds, max_dofs) -> tuple[list[int], list[float]]:
    """The L2 error against unknowns, refining adaptively from `mesh`. Each round
    refines the worst elements once and re-solves; each point is one round. Stops
    once a round passes `max_dofs`."""
    _, solution = solve(mesh)
    dofs = [solution.space.n_dofs]
    errors = [error_of(solution)]
    for _ in range(rounds):
        step = AdaptiveRefinement(mesh, problem_for, estimator, max_triangles=10**9,
                                  max_iters=1, refine_fraction=0.5)
        solution = step.run()
        mesh = step.mesh
        if solution.space.n_dofs > max_dofs:
            break
        dofs.append(solution.space.n_dofs)
        errors.append(error_of(solution))
    return dofs, errors


def red_green_example() -> tuple[Mesh, Mesh, list[str]]:
    """What red-green splitting does to an element: a four-triangle square, two of them
    refined red and then a third, with the leaves classified red or green."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    elements = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    boundary = [[0, 1], [1, 2], [2, 3], [3, 0]]
    original = Mesh(vertices, elements, boundary)
    small = original

    refiner = RedGreenRefiner(small)
    small = refiner.refine([0, 2])
    small = refiner.refine([1])
    return original, small, refiner.leaf_classifications()


@dataclass
class RefinementStudy:
    """Everything `run` computed, for the figures to read."""
    coarse_mesh: Mesh
    coarse_solution: DiffusionSolution
    coarse_error: np.ndarray        # the estimator's eta per element
    refined_mesh: Mesh
    refined_solution: DiffusionSolution
    refined_error: np.ndarray
    uniform_dofs: list[int]
    uniform_errors: list[float]
    adaptive_dofs: list[int]
    adaptive_errors: list[float]
    red_green_original: Mesh
    red_green_refined: Mesh
    red_green_classes: list[str]

    @property
    def n_coarse(self) -> int:
        return len(self.coarse_mesh.elements)

    @property
    def n_refined(self) -> int:
        return len(self.refined_mesh.elements)

    @property
    def coarse_max(self) -> float:
        return float(self.coarse_error.max())

    @property
    def coarse_norm(self) -> float:
        return float(np.sqrt(np.sum(self.coarse_error**2)))

    @property
    def refined_max(self) -> float:
        return float(self.refined_error.max())

    @property
    def refined_norm(self) -> float:
        return float(np.sqrt(np.sum(self.refined_error**2)))

    @property
    def max_reduction(self) -> float:
        """Percent drop in the largest element error from coarse to refined."""
        return 100 * (1 - self.refined_max / self.coarse_max)

    @property
    def norm_reduction(self) -> float:
        """Percent drop in the estimator's norm from coarse to refined."""
        return 100 * (1 - self.refined_norm / self.coarse_norm)


def run(_mesh, uniform_resolutions=(10, 20, 40, 80, 160), adaptive_rounds=30,
        coarse_resolution=20) -> RefinementStudy:
    """Solve on a coarse square, refine it adaptively, and compare the error against
    cost of uniform and adaptive refinement. The registry's mesh is ignored: the study
    builds its own squares at the resolutions it needs."""
    mesh = square_mesh(coarse_resolution)

    coarse_mesh = mesh
    coarse_problem, coarse_solution = solve(coarse_mesh)
    coarse_error = estimator.estimate(coarse_problem, coarse_solution)

    refined_mesh, refined_solution, refined_error = adapt(mesh)

    uniform_dofs, uniform_errors = uniform_sweep(uniform_resolutions)
    adaptive_dofs, adaptive_errors = adaptive_sweep(mesh, adaptive_rounds,
                                                    max(uniform_dofs) // 2)

    original, refined, classes = red_green_example()
    return RefinementStudy(coarse_mesh, coarse_solution, coarse_error,
                           refined_mesh, refined_solution, refined_error,
                           uniform_dofs, uniform_errors, adaptive_dofs, adaptive_errors,
                           original, refined, classes)
