"""An oscillatory function projected onto one coarse mesh's P1 and P2 spaces.

`project` states and solves one projection; `run` returns a `ProjectionStudy` of plain
results. Nothing here draws: `figures.py` does that from the study, and this file is
what the gallery shows.
"""
from dataclasses import dataclass

import numpy as np

from fem.elements import QuadraticTriangleElement
from fem.physics.equations import Projection
from fem.mesh.mesh import Mesh
from fem.post.solution import FieldSolution
from fem.loads import Source
from fem.conditions import Conditions


def target(point):
    """sin(40 r^2) about the unit square's centre: rings that tighten with radius."""
    x, y = point - np.array([0.5, 0.5])
    return [np.sin(40 * (x**2 + y**2))]


def project(mesh, element_type=None) -> FieldSolution:
    """The L2 projection of `target` onto the mesh's space of the given element."""
    return Projection().problem(mesh, Conditions(Source(target)), element_type=element_type).solve()


@dataclass
class ProjectionStudy:
    """Everything `run` computed, for the figure to read."""
    mesh: Mesh
    p1: FieldSolution
    p2: FieldSolution


def run(mesh) -> ProjectionStudy:
    """Project the target onto the mesh's P1 and P2 spaces."""
    return ProjectionStudy(mesh, project(mesh), project(mesh, QuadraticTriangleElement))
