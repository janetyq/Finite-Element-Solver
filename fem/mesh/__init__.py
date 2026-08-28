"""Mesh data structures and generation: `Mesh`, the structured builders, Ruppert's
Delaunay refinement of a `PSLG`, and red-green refinement."""
from fem.mesh.curves import Arc, Circle, CubicBezier, Curve
from fem.mesh.mesh import Mesh, boundary_facets
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.structured import annulus_mesh, box_mesh
from fem.mesh.svg import PSLG, read_svg_to_pslg

__all__ = [
    'Arc', 'Circle', 'CubicBezier', 'Curve', 'Mesh', 'boundary_facets', 'RedGreenRefiner',
    'RuppertsAlgorithm', 'annulus_mesh', 'box_mesh', 'PSLG', 'read_svg_to_pslg',
]
