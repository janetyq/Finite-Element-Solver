"""Geometric primitives: areas, volumes, point-in-polygon, circumcenters,
triangle angles, and boundary extraction from a triangulation.
"""
import numpy as np


def calculate_polygon_area(polygon):
    if polygon.shape[1] == 2:
        x, y = polygon.T
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    else:
        raise NotImplementedError('Polygon area not supported for 3D')


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


def get_boundary_from_vertices_elements(vertices, elements):
    edges = set()
    boundary_edges = set()

    # Step 1: Convert elements to edges
    for element in elements:
        for i in range(3):  # Each element is a triangle (3 vertices)
            edge = tuple(sorted([element[i], element[(i + 1) % 3]]))  # Edges are represented by sorted vertex indices
            if edge in edges:
                # If edge is already in set, it's an interior edge, remove it from edges
                edges.remove(edge)
            else:
                # If edge is not in set, it's a new edge, add it to edges
                edges.add(edge)

    # Step 2: Identify boundary edges
    for edge in edges:
        count = 0
        for element in elements:
            if edge[0] in element and edge[1] in element:
                count += 1
        if count == 1:
            boundary_edges.add(edge)

    boundary_edges = [list(edge) for edge in boundary_edges]

    return boundary_edges
