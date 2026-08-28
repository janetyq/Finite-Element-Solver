"""Four outlines, traced and generated, through one pipeline: simplify the traced
ones with Douglas-Peucker, triangulate with Ruppert's algorithm, and solve a Poisson
problem on each.

`zoo_shapes` gathers the outlines, `mesh_outline` and `dome` handle one each, and `run`
returns an `OutlineStudy` of plain results. Nothing here draws: `figures.py` does that
from the `OutlineStudy`, and this file is what the gallery shows.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet
from fem.equations import Poisson
from fem.geometry import calculate_triangle_min_angle
from fem.mesh.mesh import Mesh
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.svg import (
    PSLG, read_svg_to_list_of_path_points, read_svg_to_pslg, douglas_peucker)
from fem.regions import everywhere
from fem.solver import Solver

from domains import gear_pslg, star_pslg

# Resolved against the repo, so a demo does not depend on where it was launched from.
DEFAULT_SVG_FILE = str(Path(__file__).resolve().parents[3] / 'files' / 'california.svg')
CLOUD_SVG_FILE = str(Path(__file__).resolve().parents[3] / 'files' / 'cloud.svg')

# Douglas-Peucker: drop points that deviate less than this fraction of the curve's
# bounding-box extent. Ruppert's cost grows steeply in point count.
DEFAULT_SIMPLIFICATION_TOLERANCE = 0.005

# The Poisson "dome": a unit source pinned to zero on every boundary, so the field is a
# picture of the domain itself, tallest where it is widest.
dome_bc = BoundaryConditions(Dirichlet(everywhere(), 0))
dome_equation = Poisson(source=1.0)


def get_curve_from_svg(svg_file):
    output = read_svg_to_list_of_path_points(svg_file)
    curve = max(output, key=lambda x: len(x)) # get the longest path
    return np.array(curve)


def close_ring(points):
    """`points` with its first vertex repeated at the end, for plotting.

    A closed SVG path comes back as a ring whose closing edge is implied, as
    `PSLG.from_loops` assumes; `ax.plot` needs it spelled out.
    """
    return np.vstack([points, points[:1]])


def curve_extent(curve) -> float:
    """The longer side of `curve`'s bounding box, which the tolerance is a fraction of."""
    return float(max(np.max(curve, axis=0) - np.min(curve, axis=0)))


def simplify_curve(curve, tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE):
    """Simplify `curve` with Douglas-Peucker, `tolerance` a fraction of its extent."""
    return douglas_peucker(curve, tolerance * curve_extent(curve))


def save_curve(curve, save_file='douglas_peucker_output.json'):
    """Write a simplified curve out as JSON, to be read back as an outline later."""
    with open(save_file, 'w') as f:
        json.dump(np.asarray(curve).tolist(), f)


def zoo_shapes(svg_tolerance=DEFAULT_SIMPLIFICATION_TOLERANCE) -> list[tuple[str, PSLG]]:
    """The outlines the zoo meshes, as (name, PSLG) pairs.

    California and the cloud are traced from `files/*.svg` and simplified on the way in;
    the star and gear are generated (`domains.py`). Each puts a different demand on the
    mesher: disconnected islands, a curved boundary, sharp reentrant corners, and
    repeated teeth around a circular bore.
    """
    return [
        ('California', read_svg_to_pslg(DEFAULT_SVG_FILE, tolerance=svg_tolerance)),
        ('Cloud', read_svg_to_pslg(CLOUD_SVG_FILE, tolerance=svg_tolerance)),
        ('Gear', gear_pslg()),
        ('Star', star_pslg()),
    ]


def mesh_outline(pslg: PSLG, min_angle, max_area_fraction) -> Mesh:
    """Triangulate `pslg` by Ruppert's algorithm to a minimum-angle bound and a maximum
    triangle area given as a fraction of the outline's own area."""
    pslg.validate()
    return RuppertsAlgorithm(pslg, min_angle=min_angle,
                             max_area=max_area_fraction * pslg.area()).refine()


def dome(mesh: Mesh) -> np.ndarray:
    """Solve -div(grad u) = 1 with u = 0 on every boundary of `mesh`."""
    return Solver(mesh, dome_equation, dome_bc).solve().u


@dataclass
class MeshedOutline:
    """One outline through the pipeline: its name, point count, mesh, and dome."""
    name: str
    n_points: int
    mesh: Mesh
    u: np.ndarray

    @property
    def n_triangles(self) -> int:
        return len(self.mesh.elements)

    @property
    def worst_angle(self) -> float:
        """The smallest angle in the mesh (degrees), against the bound Ruppert's was set."""
        return float(calculate_triangle_min_angle(
            np.asarray(self.mesh.vertices)[np.asarray(self.mesh.elements)]).min())


@dataclass
class OutlineStudy:
    """Everything `run` computed, for the figure and the table to read."""
    shapes: list[MeshedOutline]
    min_angle: float


def run(min_angle=28, max_area_fraction=0.0008, svg_tolerance=0.001) -> OutlineStudy:
    """Mesh and solve every outline in the zoo.

    `svg_tolerance` is finer than the default so the coastline keeps its detail (the
    raw trace has ~1700 points).
    """
    shapes = []
    for name, pslg in zoo_shapes(svg_tolerance):
        mesh = mesh_outline(pslg, min_angle, max_area_fraction)
        shapes.append(MeshedOutline(name, len(pslg.vertices), mesh, dome(mesh)))
    return OutlineStudy(shapes, min_angle)
