"""SIMP topology optimization of a simply supported beam to half its material.

`mbb_conditions`, `solve_solid`, and `optimize` each state and solve one problem; `run`
calls them and returns a `TopologyStudy` of plain results. Nothing here draws:
`figures.py` does that from the study, and this file is what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.design import DesignHistory, DesignOptimizer, SIMPModel, calculate_smoothing_matrix
from fem.equations import LinearElastic
from fem.mesh.mesh import Mesh
from fem.regions import in_box, intersect, on_plane
from fem.solution import ElasticSolution
from fem.solver import Solver

E, NU = 200.0, 0.4
equation = LinearElastic(E, NU)


def mbb_conditions(mesh) -> BoundaryConditions:
    """A simply supported (MBB) beam, the classic topology-optimization test: pinned at
    one bottom corner, a vertical roller at the other, a downward load at the top
    centre."""
    w = np.max(mesh.vertices[:, 0])
    h = np.max(mesh.vertices[:, 1])
    bottom, top = on_plane(1, 0.0), on_plane(1, h)
    return BoundaryConditions(
        Dirichlet(intersect(bottom, in_box([None, None], [0.04 * w, None])), [0, 0]),
        Dirichlet(intersect(bottom, in_box([0.96 * w, None], [None, None])), [None, 0]),
        # A load over the central fifth of the top rather than a point, so it lands on a
        # boundary edge on any mesh, including the tiny smoke-test one.
        Neumann(intersect(top, in_box([0.4 * w, None], [0.6 * w, None])), [0, -0.5]),
    )


def solve_solid(mesh, bc) -> ElasticSolution:
    """The solid block: 100% material, the baseline the optimized one is measured
    against."""
    return Solver(mesh, equation, bc).solve()


def optimize(mesh, bc, iters, smoothing_radius=0.05) -> tuple[DesignOptimizer, DesignHistory]:
    """Where to put half the material. Compliance is u.f, the work the load does, so a
    lower value is a stiffer structure; SIMP minimizes it under the volume constraint.
    The smoothing radius is a physical length, so a finer mesh resolves the same
    structure rather than growing thinner members."""
    model = SIMPModel(equation.problem(mesh, bc),
                      sensitivity_filter=calculate_smoothing_matrix(mesh, smoothing_radius))
    design = DesignOptimizer(model, volume_frac=0.5, iters=iters, move=0.1)
    return design, design.run()


@dataclass
class TopologyStudy:
    """Everything `run` computed, for the figures and the summary to read."""
    mesh: Mesh
    bc: BoundaryConditions
    solid: ElasticSolution
    optimized: ElasticSolution
    history: DesignHistory

    @property
    def aspect(self) -> float:
        return float(np.max(self.mesh.vertices[:, 0]) / np.max(self.mesh.vertices[:, 1]))

    @property
    def compliance_solid(self) -> float:
        return float(self.solid.compliance.sum())

    @property
    def compliance_opt(self) -> float:
        return float(self.history.objective[-1])

    @property
    def ratio(self) -> float:
        """The optimized beam's compliance as a fraction of the solid one's."""
        return self.compliance_opt / self.compliance_solid


def run(mesh, iters=60) -> TopologyStudy:
    """Solve the solid beam, then optimize half its material away."""
    bc = mbb_conditions(mesh)
    solid = solve_solid(mesh, bc)
    design, history = optimize(mesh, bc, iters)
    assert design.solution is not None
    return TopologyStudy(mesh, bc, solid, design.solution, history)
