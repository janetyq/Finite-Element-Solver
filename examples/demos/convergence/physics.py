"""Convergence rates in space and time against manufactured solutions.

The one demo that shows not what the solver computed but how wrong it was:

  in space  P1 elements are O(h^2) (halve h, quarter the error) for a scalar
            unknown and for a coupled vector one alike; P2 is O(h^3);
  in time   the theta method's order is theta's to choose: 1 at backward Euler,
            2 at Crank-Nicolson, the default.

The same studies run as assertions in tests/test_convergence.py and tests/test_convergence_heat.py.
`run` gathers them into a `RatesStudy` of plain results. Nothing here draws:
`figures.py` does that from the study, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np
from mms import (
    ConvergenceStudy,
    elastic_convergence,
    load_comparison_convergence,
    mixed_bc_convergence,
    poisson_convergence,
    poisson_p2_convergence,
    theta_convergence,
)

from fem.elements import QuadraticTriangleElement
from fem.space import FunctionSpace


def space_studies(resolutions, elastic_resolutions):
    """P1 and P2 Poisson and P1 elasticity against h, with the DOF counts of the two
    Poisson sequences for the accuracy-per-cost view: P2 spends more unknowns per
    element, and the question is whether its faster rate pays that back."""
    solves = poisson_convergence(resolutions)
    p2_solves = poisson_p2_convergence(resolutions)
    p1_dofs = np.array([FunctionSpace(s.mesh).n_dofs for s in solves])
    p2_dofs = np.array([FunctionSpace(s.mesh, QuadraticTriangleElement).n_dofs
                        for s in p2_solves])
    return (ConvergenceStudy.from_solves(solves), ConvergenceStudy.from_solves(p2_solves),
            ConvergenceStudy.from_solves(elastic_convergence(elastic_resolutions)),
            p1_dofs, p2_dofs)


def time_studies(step_counts):
    """Crank-Nicolson and backward Euler against dt.

    Step counts chosen to sit in the asymptotic band: over coarser steps Crank-Nicolson
    reads an order near 3, because lambda*dt is not yet small and the leading error term
    is not yet the one that dominates.
    """
    return theta_convergence(0.5, step_counts), theta_convergence(1.0, step_counts)


def load_studies(resolutions):
    """The same P1 solve with the source read only at the vertices (its linear
    interpolant) against one sampled at the quadrature points: the rate is the same,
    the constant is not."""
    loads = load_comparison_convergence(resolutions)
    steps = np.array([lc.h for lc in loads])
    nodal = ConvergenceStudy(steps, np.array([lc.nodal_error for lc in loads]))
    sampled = ConvergenceStudy(steps, np.array([lc.sampled_error for lc in loads]))
    return nodal, sampled


@dataclass
class RatesStudy:
    """Everything `run` computed, for the figure and the table to read."""
    poisson: ConvergenceStudy
    p2: ConvergenceStudy
    elastic: ConvergenceStudy
    p1_dofs: np.ndarray
    p2_dofs: np.ndarray
    crank_nicolson: ConvergenceStudy
    backward_euler: ConvergenceStudy
    nodal: ConvergenceStudy
    sampled: ConvergenceStudy
    poisson_3d: ConvergenceStudy
    mixed_bc: ConvergenceStudy

    @property
    def table(self) -> list[tuple[str, ConvergenceStudy, int]]:
        """Each study with its name and the order theory expects of it."""
        return [('Poisson P1 (h)', self.poisson, 2),
                ('Poisson P2 (h)', self.p2, 3),
                ('Poisson 3D (h)', self.poisson_3d, 2),
                ('Elasticity (h)', self.elastic, 2),
                ('Neumann/Robin (h)', self.mixed_bc, 2),
                ('Crank-Nicolson (dt)', self.crank_nicolson, 2),
                ('Backward Euler (dt)', self.backward_euler, 1),
                ('Nodal load (h)', self.nodal, 2),
                ('Sampled load (h)', self.sampled, 2)]


def run(resolutions=(11, 21, 41, 81), elastic_resolutions=(9, 17, 33),
        step_counts=(16, 32, 64, 128), poisson_3d_resolutions=(5, 9, 13)) -> RatesStudy:
    """Run every study and collect the measured rates."""
    poisson, p2, elastic, p1_dofs, p2_dofs = space_studies(resolutions, elastic_resolutions)
    crank_nicolson, backward_euler = time_studies(step_counts)
    nodal, sampled = load_studies(resolutions)
    poisson_3d = ConvergenceStudy.from_solves(poisson_convergence(poisson_3d_resolutions, dim=3))
    mixed_bc = ConvergenceStudy.from_solves(mixed_bc_convergence(resolutions))
    return RatesStudy(poisson, p2, elastic, p1_dofs, p2_dofs, crank_nicolson, backward_euler,
                      nodal, sampled, poisson_3d, mixed_bc)
