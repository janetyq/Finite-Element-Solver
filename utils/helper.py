import numpy as np
from math import sin, cos, pi

# for testing convenience
def print_min_max(array):
    print(f'Min: {np.min(array)}')
    print(f'Max: {np.max(array)}')

def move_to_origin(points):
    return points - np.min(points, axis=0)

def normalize_points(points):
    points_range = np.max(points, axis=0) - np.min(points, axis=0)
    max_range = np.max(points_range)
    range = np.array([max_range, max_range])
    return (points - np.min(points, axis=0)) / range

def bump_function(points, center, mag=100, size=0.5):
    return np.array([mag*cos(pi/2*np.linalg.norm(point - center)/size) if np.linalg.norm(point - center) < size else 0 for point in points])

def calculate_triangle_area(points):
    p1, p2, p3 = points
    return 0.5 * abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])))

