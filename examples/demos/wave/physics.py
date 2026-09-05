"""A wave front meeting a harbor breakwater, diffracting through its gap.

`run` meshes the basin, sets a front travelling toward the wall, and steps the wave
equation by Newmark, returning a `HarborStudy` of plain results. Nothing here draws:
`figures.py` does that from the `HarborStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.algebra.integrators import NewmarkMethod
from fem.conditions import Conditions, Initial
from fem.field import NodalField
from fem.mesh.mesh import Mesh
from fem.mesh.outline import Outline
from fem.physics.equations import Wave
from fem.post.solution import TransientSolution


def harbor_outline(length: float = 6.0, width: float = 4.0, wall_x: float = 2.5,
                   wall_thickness: float = 0.15, gap: float = 1.2) -> Outline:
    """A rectangular basin crossed by a breakwater with one gap, as a single loop.

    Open water lies left of the wall at `wall_x`, the sheltered harbor to its right. The
    two wall arms grow inward from the top and bottom edges, leaving `gap` open at
    mid-width.
    """
    x0, x1 = wall_x, wall_x + wall_thickness
    y0, y1 = (width - gap) / 2, (width + gap) / 2
    outline = np.array([
        [0.0, 0.0], [x0, 0.0], [x0, y0], [x1, y0], [x1, 0.0], [length, 0.0],
        [length, width], [x1, width], [x1, y1], [x0, y1], [x0, width], [0.0, width],
    ])
    return Outline.from_polygons([outline])

WALL_X, WALL_THICKNESS = 2.5, 0.15


@dataclass
class HarborStudy:
    """Everything `run` computed, for the figures to read."""
    mesh: Mesh
    u_initial: NodalField
    dudt_initial: NodalField
    solution: TransientSolution

    @property
    def u_values(self) -> np.ndarray:
        return self.solution.dofs

    @property
    def t_values(self) -> np.ndarray:
        return self.solution.t

    @property
    def harbor(self) -> np.ndarray:
        """Mask of the vertices on the sheltered side of the breakwater."""
        return self.mesh.vertices[:, 0] > WALL_X + WALL_THICKNESS


def run(c=1.0, front_x=1.0, front_width=0.25, dt=0.02, steps=400, min_angle=28, max_area=0.04,
        uniform_rounds=2) -> HarborStudy:
    """Mesh the basin, launch a front at the breakwater, and step it by Newmark."""
    pslg = harbor_outline(wall_x=WALL_X, wall_thickness=WALL_THICKNESS)
    # Ruppert's meshes the outline coarsely; uniform red refinement then supplies the
    # resolution the front needs, keeping the angle bound at a fraction of the cost.
    mesh = pslg.mesh(min_angle=min_angle, max_area=max_area)
    for _ in range(uniform_rounds):
        mesh = mesh.refined()

    # A straight front on the open side, travelling toward the wall. Given d'Alembert's
    # pairing u = g(x - ct), du/dt = -c g'(x), so it moves one way instead of splitting.
    def profile(p):
        return np.exp(-((p[:, 0] - front_x) / front_width) ** 2)

    # No boundary conditions, so every edge is a wall: the natural du/dn = 0 reflects
    # a wave the same way up.
    bc = Conditions(Initial(profile, v0=lambda p: 2 * c * (p[:, 0] - front_x) / front_width**2 * profile(p)))
    wave = Wave(stiffness=c**2).problem(mesh, bc)
    solution = NewmarkMethod(dt=dt, steps=steps).solve(wave)
    return HarborStudy(mesh, wave.u0, wave.v0, solution)
