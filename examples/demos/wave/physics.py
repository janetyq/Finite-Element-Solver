"""A wave front meeting a harbor breakwater, diffracting through its gap.

`run` meshes the basin, sets a front travelling toward the wall, and steps the wave
equation by Newmark, returning a `HarborStudy` of plain results. Nothing here draws:
`figures.py` does that from the `HarborStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions
from fem.equations import Wave
from fem.integrators import NewmarkMethod
from fem.mesh.mesh import Mesh
from fem.solution import TransientSolution

from domains import harbor_pslg

WALL_X, WALL_THICKNESS = 2.5, 0.15


@dataclass
class HarborStudy:
    """Everything `run` computed, for the figures to read."""
    mesh: Mesh
    u_initial: np.ndarray
    dudt_initial: np.ndarray
    solution: TransientSolution

    @property
    def u_values(self) -> list[np.ndarray]:
        return self.solution.u

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
    pslg = harbor_pslg(wall_x=WALL_X, wall_thickness=WALL_THICKNESS)
    # Ruppert's meshes the outline coarsely; uniform red refinement then supplies the
    # resolution the front needs, keeping the angle bound at a fraction of the cost.
    mesh = pslg.mesh(min_angle=min_angle, max_area=max_area)
    for _ in range(uniform_rounds):
        mesh = mesh.refined()

    # No conditions, so every edge is a wall: the natural du/dn = 0 reflects a wave
    # the same way up.
    bc = BoundaryConditions()
    wave = Wave(stiffness=c**2).problem(mesh, bc)

    # A straight front on the open side, travelling toward the wall. Given d'Alembert's
    # pairing u = g(x - ct), du/dt = -c g'(x), so it moves one way instead of splitting.
    def profile(p):
        return np.exp(-((p[0] - front_x) / front_width) ** 2)

    u_initial = wave.space.interpolate(profile)
    dudt_initial = wave.space.interpolate(
        lambda p: 2 * c * (p[0] - front_x) / front_width**2 * profile(p))
    solution = NewmarkMethod(dt=dt, steps=steps).solve(wave, u_initial, dudt_initial)
    return HarborStudy(mesh, u_initial, dudt_initial, solution)
