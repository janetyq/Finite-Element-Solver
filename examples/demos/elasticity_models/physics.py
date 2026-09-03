"""One clamped block stretched four ways: a linear solve, energy minimisation, and two
finite-strain material laws — then the same stretch walked up from zero, so the
force-stretch curve separates the laws quantitatively.

`stretch_bc` states the condition and `stretch_four_ways` solves it;
`load_deflection` ramps the stretch through `QuasiStaticStepping` and reads the
reaction force at every level. `run` calls them and returns a `StretchStudy` of plain
results. Nothing here draws: `figures.py` does that from the `StretchStudy`, and this
file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import Dirichlet
from fem.conditions import Conditions, Initial
from fem.physics.energies import NeohookeanEnergyDensity
from fem.physics.equations import FiniteStrainElastic, LinearElastic
from fem.physics.forms import EnergyForm
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.problem import Problem
from fem.regions import on_plane
from fem.post.solution import ElasticSolution
from fem.algebra.solve import BacktrackingLineSearch, NewtonSolve
from fem.algebra.stepping import QuasiStaticStepping

E, NU = 200.0, 0.4


def stretch_bc(width, stretch) -> Conditions:
    """Both ends Dirichlet: the left held at zero, the right displaced to `stretch` of
    the width. Nothing is loaded."""
    return Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, width), [stretch*width, 0]),
    )


def stretch_four_ways(mesh: Mesh, bc: Conditions):
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
    energy_problem = Problem(linear.space(mesh), EnergyForm(linear.energy_density()), bc)
    energy_u = NewtonSolve(line_search=BacktrackingLineSearch()).solve(energy_problem)
    linear_solution = linear.problem(mesh, bc).solve()
    newton = NewtonSolve(line_search=BacktrackingLineSearch())
    solutions = [
        ('Linear solve\n(small strain)', linear_solution),
        ('Energy minimisation\n(small strain)', energy_problem.solution(energy_u)),
        ('Green-Lagrange\n(St-Venant-Kirchhoff)',
         stvk.problem(mesh, bc).solve(strategy=newton, initial=Initial(linear_solution))),
        ('Neo-Hookean\n(invariants of C)',
         neohookean.problem(mesh, bc).solve(strategy=newton, initial=Initial(linear_solution))),
    ]
    return solutions, energy_problem, energy_u


def load_deflection(width, stretch, steps=10, resolution=24):
    """Walk the stretch up from zero and read the force it takes, per material law.

    The pulled edge is displacement-controlled, so the load is the reaction: the
    internal force at that edge's x DOFs, summed. `QuasiStaticStepping` scales the
    prescribed displacement to each level and warm-starts Newton from the last one,
    which is also how a finite-strain solve is walked past a seed a single solve
    would diverge from. A coarse mesh of its own: the curve is a global scalar per
    level, converged long before the stress field of the main figure is.

    Returns `[(name, stretches, forces)]`, one triple per law.
    """
    mesh = box_mesh([[0.0, 0.0], [width, width]], (resolution, resolution))
    bc = stretch_bc(width, stretch)
    models = [
        ('Small strain', LinearElastic(E=E, nu=NU)),
        ('St-Venant-Kirchhoff', FiniteStrainElastic(E=E, nu=NU)),
        ('Neo-Hookean', FiniteStrainElastic(E=E, nu=NU, law=NeohookeanEnergyDensity)),
    ]
    stepping = QuasiStaticStepping(steps=steps)
    curves = []
    for name, equation in models:
        problem = equation.problem(mesh, bc)
        history = stepping.solve(problem)
        pulled = np.isclose(problem.space.node_coords[:, 0], width)
        x_dofs = 2 * np.flatnonzero(pulled)
        forces = np.array([float(problem.internal_residual(u)[x_dofs].sum())
                           for u in history.dofs])
        curves.append((name, history.t * stretch, forces))
    return curves


@dataclass
class StretchStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    mesh: Mesh
    stretch: float
    bc: Conditions
    solutions: list[tuple[str, ElasticSolution]]   # (panel name, solution)
    energy_problem: Problem
    energy_u: np.ndarray
    curves: list[tuple[str, np.ndarray, np.ndarray]]   # (law, stretches, reaction forces)

    @property
    def von_mises(self) -> list[np.ndarray]:
        return [solution.von_mises for _, solution in self.solutions]

    @property
    def drift(self) -> float:
        """Relative difference between the linear and energy-minimised displacements."""
        linear_u = self.solutions[0][1].dofs
        return float(np.linalg.norm(self.energy_u - linear_u) / np.linalg.norm(linear_u))

    @property
    def minimised_energy(self) -> float:
        return float(self.energy_problem.energy(self.energy_u))


def run(mesh: Mesh, stretch=0.5, curve_steps=10, curve_resolution=24) -> StretchStudy:
    """Stretch the block on `mesh` by `stretch` of its width, four ways; then walk the
    same stretch up from zero for the force-stretch curve of each law."""
    width = np.max(mesh.vertices[:, 0])
    bc = stretch_bc(width, stretch)
    solutions, energy_problem, energy_u = stretch_four_ways(mesh, bc)
    curves = load_deflection(width, stretch, curve_steps, curve_resolution)
    return StretchStudy(mesh, stretch, bc, solutions, energy_problem, energy_u, curves)
