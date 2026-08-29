"""Poisson's equation as potential flow over a NACA airfoil, on P2 elements.

`run` meshes the airfoil in its channel and solves Laplace's equation for the velocity
potential, returning a `FlowStudy` of plain results. Nothing here draws: `figures.py`
does that from the `FlowStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet
from fem.elements import QuadraticTriangleElement
from fem.physics.equations import Poisson
from fem.mesh.mesh import Mesh
from fem.regions import on_plane
from fem.post.solution import ScalarFieldSolution

from domains import airfoil_channel_outline


@dataclass
class FlowStudy:
    """Everything `run` computed, for the figures to read."""
    angle_of_attack: float
    mesh: Mesh
    bc: BoundaryConditions
    solution: ScalarFieldSolution      # the velocity potential phi, on P2

    @property
    def speed(self) -> np.ndarray:
        """|v| = |grad(phi)|, read at the nodes so the P2 tessellation draws it smoothly."""
        return np.linalg.norm(self.solution.nodal_flux(), axis=1)   # (n_nodes,)

    @property
    def speed_cap(self) -> float:
        """Ideal flow with no Kutta condition predicts a near-singular velocity at the
        sharp edges; clip it to a high percentile so the flow over the wing stays legible."""
        return float(np.percentile(self.speed, 96))


def run(length=7.0, height=4.0, chord=3.0, angle_of_attack=12.0, n_points=80, min_angle=20,
        max_area_fraction=0.0015) -> FlowStudy:
    """Mesh the airfoil in its channel and solve for the velocity potential."""
    # An ideal (incompressible, irrotational) flow has a velocity potential phi with
    # v = grad(phi) and div(v) = 0, so phi solves Laplace's equation, Poisson's with no
    # source. The wing carries no
    # flow through it, the natural (zero-flux) condition of the weak form: say nothing
    # on its surface and it becomes a streamline the flow parts around.
    outline = airfoil_channel_outline(length, height, chord, angle_of_attack, n_points=n_points)
    mesh = outline.mesh(min_angle=min_angle, max_area_fraction=max_area_fraction)

    equation = Poisson(source=0)   # Laplace: no sources in the flow
    # phi rises from inlet to outlet, so v = grad(phi) runs left to right. The wing and
    # the walls take no condition, so they are no-flux streamlines.
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), 0.0),
        Dirichlet(on_plane(0, length), 1.0),
    )

    problem = equation.problem(mesh, bc, element_type=QuadraticTriangleElement)
    solution = problem.solve()
    return FlowStudy(angle_of_attack, mesh, bc, solution)
