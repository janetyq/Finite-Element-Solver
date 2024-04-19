import numpy as np
from math import sin, cos, pi

# 2d vector operations - faster than numpy
def calc_dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]

def calc_norm(vec):
    return (vec[0]**2 + vec[1]**2)**0.5

def calc_cross(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]

def bump_function(points, center, mag=100, size=0.5):
    return np.array([mag*cos(pi/2*np.linalg.norm(point - center)/size) if np.linalg.norm(point - center) < size else 0 for point in points])

def calculate_triangle_area(points):
    p1, p2, p3 = points
    return 0.5 * abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])))

def calculate_polygon_area(polygon):
    x, y = polygon.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

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

def get_boundary_from_points_faces(points, faces):
    edges = set()
    boundary_edges = set()

    # Step 1: Convert faces to edges
    for face in faces:
        for i in range(3):  # Each face is a triangle (3 vertices)
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))  # Edges are represented by sorted vertex indices
            if edge in edges:
                # If edge is already in set, it's an interior edge, remove it from edges
                edges.remove(edge)
            else:
                # If edge is not in set, it's a new edge, add it to edges
                edges.add(edge)

    # Step 2: Identify boundary edges
    for edge in edges:
        count = 0
        for face in faces:
            if edge[0] in face and edge[1] in face:
                count += 1
        if count == 1:
            boundary_edges.add(edge)

    boundary_edges = [list(edge) for edge in boundary_edges]

    return boundary_edges

