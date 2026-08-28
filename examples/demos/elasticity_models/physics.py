"""One clamped block stretched four ways: a linear solve, energy minimisation, and two
finite-strain material laws.

`stretch_bc` states the condition and `stretch_four_ways` solves it; `run` calls them
and returns a `StretchStudy` of plain results. Nothing here draws: `figures.py` does
that from the `StretchStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet
from fem.energies import NeohookeanEnergyDensity
from fem.equations import FiniteStrainElastic, LinearElastic
from fem.forms import EnergyForm
from fem.mesh.mesh import Mesh
from fem.problem import Problem
from fem.regions import on_plane
from fem.solution import ElasticSolution
from fem.solve import BacktrackingLineSearch, NewtonSolve
from fem.solver import Solver

E, NU = 200.0, 0.4


def stretch_bc(width, stretch) -> BoundaryConditions:
    """Both ends Dirichlet: the left held at zero, the right displaced to `stretch` of
    the width. Nothing is loaded."""
    return BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, width), [stretch*width, 0]),
    )


def stretch_four_ways(mesh: Mesh, bc: BoundaryConditions):
    """Solve the stretch as a linear system, as an energy minimisation, and under the
    St-Venant-Kirchhoff and Neo-Hookean finite-strain laws.

    The first two are the same physics reached two ways: solving K u = f, and Newton
    on the elastic energy that system is the stationary point of. Their displacements
    agree to machine precision. Their stress does not, since the two recover different
    measures: sigma = D:eps against the true Cauchy stress J^-1 P F^T at the deformed
    configuration, which agree only to O(||grad u||).

    The last two change the physics: two finite-strain laws that share the same
    small-strain linearisation but differ once the stretch is large. Green-Lagrange
    St-Venant-Kirchhoff has a polynomial energy; Neo-Hookean is written in the
    invariants of C = F^T F and carries the log J terms that keep it stable in
    compression. Both reach the same energy at small strain and part company here.

    The finite-strain solves are seeded with the linear solution. Seeded from zero,
    the elements beside the displaced edge start stretched by the whole prescribed
    displacement over one element width, and Newton's early steps from there teeter on
    inverting them (a J <= 0 element has infinite Neo-Hookean energy); the small-strain
    answer is a few Newton steps from either finite-strain one.

    Returns the named solutions, and the energy problem with its minimiser.
    """
    linear = LinearElastic(E=E, nu=NU)
    stvk = FiniteStrainElastic(E=E, nu=NU)
    neohookean = FiniteStrainElastic(E=E, nu=NU, law=NeohookeanEnergyDensity)
    # The second solve states small strain as an energy and minimises it: the same
    # density the linear stiffness is the Hessian of, under Newton.
    energy_problem = Problem(linear.space(mesh), EnergyForm(linear.energy_density()), bc=bc)
    energy_u = NewtonSolve(line_search=BacktrackingLineSearch()).solve(energy_problem)
    linear_solution = Solver(mesh, linear, bc).solve()
    newton = NewtonSolve(line_search=BacktrackingLineSearch())
    solutions = [
        ('Linear solve\n(small strain)', linear_solution),
        ('Energy minimisation\n(small strain)', energy_problem.solution(energy_u)),
        ('Green-Lagrange\n(St-Venant-Kirchhoff)',
         Solver(mesh, stvk, bc).problem().solve(strategy=newton, u0=linear_solution.u)),
        ('Neo-Hookean\n(invariants of C)',
         Solver(mesh, neohookean, bc).problem().solve(strategy=newton, u0=linear_solution.u)),
    ]
    return solutions, energy_problem, energy_u


@dataclass
class StretchStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    mesh: Mesh
    stretch: float
    bc: BoundaryConditions
    solutions: list[tuple[str, ElasticSolution]]   # (panel name, solution)
    energy_problem: Problem
    energy_u: np.ndarray

    @property
    def von_mises(self) -> list[np.ndarray]:
        return [solution.von_mises for _, solution in self.solutions]

    @property
    def drift(self) -> float:
        """Relative difference between the linear and energy-minimised displacements."""
        linear_u = self.solutions[0][1].u
        return float(np.linalg.norm(self.energy_u - linear_u) / np.linalg.norm(linear_u))

    @property
    def minimised_energy(self) -> float:
        return float(self.energy_problem.energy(self.energy_u))


def run(mesh: Mesh, stretch=0.5) -> StretchStudy:
    """Stretch the block on `mesh` by `stretch` of its width, four ways."""
    bc = stretch_bc(np.max(mesh.vertices[:, 0]), stretch)
    solutions, energy_problem, energy_u = stretch_four_ways(mesh, bc)
    return StretchStudy(mesh, stretch, bc, solutions, energy_problem, energy_u)
