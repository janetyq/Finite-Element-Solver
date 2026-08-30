"""Mesh data structures and generation: `Mesh`, the structured builders, an `Outline`
of pieces meshed by Ruppert's Delaunay refinement of its sampled `PSLG`, and red-green
refinement."""
from fem.mesh.curves import Arc, Circle, CubicBezier, Curve, Line, Piece
from fem.mesh.mesh import Mesh, boundary_facets, triangle_angles, triangle_min_angle
from fem.mesh.outline import Outline, douglas_peucker
from fem.mesh.pslg import PSLG, point_in_polygon, polygon_area
from fem.mesh.refinement import RedGreenRefiner
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.structured import box_mesh

__all__ = [
    'Arc', 'Circle', 'CubicBezier', 'Curve', 'Line', 'Mesh', 'Outline', 'PSLG', 'Piece',
    'RedGreenRefiner', 'RuppertsAlgorithm', 'boundary_facets', 'box_mesh',
    'douglas_peucker', 'point_in_polygon', 'polygon_area', 'triangle_angles',
    'triangle_min_angle',
]
