"""Geometric primitives: areas, volumes, point-in-polygon, circumcenters,
triangle angles, and boundary extraction from a triangulation.
"""
import itertools
from collections import Counter

import numpy as np


def calculate_polygon_area(polygon):
    if polygon.shape[1] == 2:
        x, y = polygon.T
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if polygon.shape[1] == 3 and len(polygon) == 3:
        # Half the cross-product magnitude. Needed for the triangular boundary
        # facets of a tet mesh, so this gates every 3D path -- not only the
        # surface-mesh case.
        a, b = polygon[1] - polygon[0], polygon[2] - polygon[0]
        return 0.5 * float(np.linalg.norm(np.cross(a, b)))
    # A general planar polygon in 3D needs Newell's method to recover the normal;
    # nothing asks for one yet, so refuse rather than return a wrong number.
    raise NotImplementedError(
        f'polygon area is defined for 2D polygons and 3D triangles, '
        f'got {len(polygon)} points in {polygon.shape[1]}D'
    )


def calculate_tetrahedron_volume(tetrahedron): # TODO: similar for triangle?
    a, b, c = tetrahedron[1:] - tetrahedron[0]
    return np.abs(np.dot(a, np.cross(b, c)) / 6)


def point_in_polygon(point, polygon):
    x, y = point
    x_coords, y_coords = polygon.T
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[(i+1) % n], y_coords[(i+1) % n]
        if y1 < y <= y2 or y2 < y <= y1:
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                inside = not inside
    return inside


def calculate_circumcenter(triangle_points):
    '''Centre of the circle through a triangle's three vertices.

    Takes a single `(3, 2)` triangle or a stacked `(..., 3, 2)` array. Solved
    against the first vertex as origin, which keeps the accuracy of a very
    flat triangle -- the shape mesh refinement asks about most -- rather than
    intersecting bisectors by slope, where a near-horizontal edge loses most of
    the available precision.
    '''
    points = np.asarray(triangle_points, dtype=float)
    origin = points[..., 0, :]
    b = points[..., 1, :] - origin
    c = points[..., 2, :] - origin

    twice_area = 2 * (b[..., 0]*c[..., 1] - b[..., 1]*c[..., 0])
    if np.any(twice_area == 0):
        raise ValueError('a degenerate triangle has no circumcenter')

    b_sq = np.sum(b**2, axis=-1)
    c_sq = np.sum(c**2, axis=-1)
    return origin + np.stack([
        (c[..., 1]*b_sq - b[..., 1]*c_sq) / twice_area,
        (b[..., 0]*c_sq - c[..., 0]*b_sq) / twice_area,
    ], axis=-1)


def calculate_minimum_segment_angle(vertices, segments):
    '''The smallest angle, in degrees, between two segments sharing a vertex.

    Delaunay refinement is only guaranteed to terminate for inputs whose
    segments meet at 60 degrees or more; below that it refines around the
    corner without converging, and cost climbs steeply well before it stops
    converging at all. Returns 180 when no two segments meet.
    '''
    vertices = np.asarray(vertices, dtype=float)
    incident = {}
    for start, end in np.asarray(segments):
        incident.setdefault(int(start), []).append(int(end))
        incident.setdefault(int(end), []).append(int(start))

    smallest = 180.0
    for vertex, neighbours in incident.items():
        if len(neighbours) < 2:
            continue
        directions = vertices[neighbours] - vertices[vertex]
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(lengths == 0, 1, lengths)
        cosines = np.clip(directions @ directions.T, -1.0, 1.0)
        # The diagonal is each direction against itself; only distinct pairs count.
        pairs = np.triu_indices(len(directions), k=1)
        if len(pairs[0]):
            smallest = min(smallest, float(np.degrees(np.arccos(cosines[pairs])).min()))
    return smallest


def calculate_triangle_min_angle(triangle):
    '''The smallest interior angle, in degrees.

    Takes a single `(3, d)` triangle and returns a scalar, or a stacked
    `(..., 3, d)` array of triangles and returns an angle per triangle. Mesh
    refinement tests every element against an angle bound on every pass, so the
    batched form is the one that keeps that loop off Python.
    '''
    points = np.asarray(triangle, dtype=float)
    # Side i is opposite vertex i.
    sides = np.linalg.norm(np.roll(points, -1, axis=-2) - np.roll(points, 1, axis=-2), axis=-1)
    a, b, c = sides[..., 0], sides[..., 1], sides[..., 2]
    # Law of cosines. Clipped because a degenerate triangle can put the ratio a
    # hair outside [-1, 1], and a NaN angle would silently compare false against
    # any bound -- exactly the sliver a refinement loop must not accept as good.
    cosines = np.stack([
        (b**2 + c**2 - a**2) / (2 * b * c),
        (c**2 + a**2 - b**2) / (2 * c * a),
        (a**2 + b**2 - c**2) / (2 * a * b),
    ], axis=-1)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).min(axis=-1)


def get_boundary_from_vertices_elements(elements):
    '''Boundary facets of a linear simplex mesh, as sorted vertex-index lists.

    A facet is the codimension-1 face of an element -- an edge of a triangle, a
    face of a tet -- and it lies on the boundary exactly when it belongs to one
    element instead of two. Counting occurrences in a single pass is O(elements);
    the facets are unoriented, which is all the boundary mass matrix needs.
    '''
    facet_counts = Counter(
        facet
        for element in elements
        for facet in itertools.combinations(sorted(element), len(element) - 1)
    )
    return [list(facet) for facet, count in facet_counts.items() if count == 1]
