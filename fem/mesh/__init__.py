"""Mesh data structures and generation: `Mesh`, the structured builders, Ruppert's
Delaunay refinement of a `PSLG`, and red-green refinement."""
from fem.mesh.curves import Arc, Circle, CubicBezier, Curve
from fem.mesh.mesh import Mesh, boundary_facets, triangle_angles, triangle_min_angle
from fem.mesh.pslg import PSLG, point_in_polygon, polygon_area
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.structured import annulus_mesh, box_mesh
from fem.mesh.svg import read_svg_to_pslg

__all__ = [
    'Arc', 'Circle', 'CubicBezier', 'Curve', 'Mesh', 'PSLG', 'RedGreenRefiner',
    'RuppertsAlgorithm', 'annulus_mesh', 'boundary_facets', 'box_mesh', 'point_in_polygon',
    'polygon_area', 'read_svg_to_pslg', 'triangle_angles', 'triangle_min_angle',
]
