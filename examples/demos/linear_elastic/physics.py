"""A cantilever under a tip load, in 2D and 3D.

`bend_2d` and `bend_3d` each state and solve one problem; `run` calls them and returns
a `CantileverStudy` of plain results. Nothing here draws: `figures.py` does that from
the `CantileverStudy`, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.algebra.backends import IterativeBackend
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.physics.equations import LinearElastic
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.regions import in_box, intersect, on_plane
from fem.post.solution import ElasticSolution

E, NU = 200.0, 0.4


def clamp_and_tip_load(width) -> Conditions:
    """Clamped on the left, pulled down over the middle of the right edge."""
    return Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        # Transverse, so the beam bends. Sized for a tip deflection near 9% of the span,
        # inside the small-strain regime.
        Neumann(intersect(on_plane(0, width), in_box([None, 0.2], [None, 0.8])), [0, -0.5]),
    )


def bend_2d(mesh: Mesh, bc: Conditions) -> ElasticSolution:
    """The 2D cantilever solve."""
    return LinearElastic(E, NU).problem(mesh, bc).solve()


def bend_3d(n_3d) -> tuple[Mesh, ElasticSolution]:
    """The same clamp-and-load, one dimension up.

    The same assembly, the equation reading the tetrahedron off the connectivity. AMG-CG
    rather than a direct factorization, whose fill-in hurts in 3D.
    """
    box = box_mesh(corners=[[0, 0, 0], [4, 1, 1]],
                          resolution=(4 * n_3d // 2, n_3d // 2, n_3d // 2))
    bc_3d = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0, 0]),
        Neumann(on_plane(0, 4.0), [0, 0, -0.5]),
    )
    solution = LinearElastic(E, NU).problem(box, bc_3d).solve(backend=IterativeBackend())
    return box, solution


@dataclass
class CantileverStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    mesh: Mesh
    bc: Conditions
    solution: ElasticSolution
    box: Mesh
    solution_3d: ElasticSolution

    @property
    def tip_3d(self) -> float:
        """The 3D solve's largest vertical deflection."""
        return float(np.abs(self.solution_3d.u.reshape(-1, 3)[:, 2]).max())

    @property
    def invariants(self) -> list[tuple[str, np.ndarray]]:
        """Rotation-invariant reductions of the 2D stress tensor: von Mises, mean normal
        stress, the Tresca measure, and the largest tensile principal value."""
        s = self.solution
        return [
            ('Von Mises', s.von_mises),
            ('Pressure', s.pressure),
            ('Max shear', s.max_shear),
            ('Max principal', s.principal_stress[:, -1]),
        ]


def run(mesh: Mesh, n_3d=14) -> CantileverStudy:
    """Solve the cantilever in 2D on `mesh` and in 3D on a box `n_3d` deep."""
    bc = clamp_and_tip_load(np.max(mesh.vertices[:, 0]))
    solution = bend_2d(mesh, bc)
    box, solution_3d = bend_3d(n_3d)
    return CantileverStudy(mesh, bc, solution, box, solution_3d)
