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
        # facets of a tet mesh, so this gates FEMesh construction in 3D at all --
        # not only the surface-mesh case.
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
    edge_vectors = [triangle_points[(i+1)%3] - triangle_points[i] for i in range(3)]
    edge_midpoints = [0.5 * (triangle_points[i] + triangle_points[(i+1)%3]) for i in range(3)]
    edge_perps = [[vec[1], -vec[0]] for vec in edge_vectors]

    # remove bisectors with 0 slope
    for i in range(3):
        if edge_perps[i][0] == 0:
            edge_perps.pop(i)
            edge_midpoints.pop(i)
            break

    # calculate center using intersection of bisectors
    s1, s2 = [perp[1] / perp[0] for perp in edge_perps[:2]]
    m1, m2 = edge_midpoints[:2]
    x = (m2[1] - m1[1] + m1[0]*s1 - m2[0]*s2) / (s1 - s2)
    y = m1[1] + s1*(x - m1[0])
    center = [x, y]

    return center


def calculate_triangle_min_angle(triangle):
    # returns the smallest angle (degrees) in the triangle
    lengths = np.linalg.norm([triangle[i] - triangle[(i+1)%3] for i in range(3)], axis=1)
    angles = np.arccos([
        (lengths[1]**2 + lengths[2]**2 - lengths[0]**2) / (2 * lengths[1] * lengths[2]),
        (lengths[2]**2 + lengths[0]**2 - lengths[1]**2) / (2 * lengths[2] * lengths[0]),
        (lengths[0]**2 + lengths[1]**2 - lengths[2]**2) / (2 * lengths[0] * lengths[1])
    ])
    return np.min(angles) * 180 / np.pi


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
