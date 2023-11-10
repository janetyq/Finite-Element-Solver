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


class ConvenientIndices:
    def __init__(self, points, boundary):
        w, h = np.max(points[:, 0]), np.max(points[:, 1])
        self.boundary_idxs = list(set(boundary.ravel()))
        self.inner_idxs = list(set(range(len(points))) - set(self.boundary_idxs))
        self.top_idxs = np.array([idx for idx in self.boundary_idxs if points[idx][1] > h-1e-6])
        self.bottom_idxs = np.array([idx for idx in self.boundary_idxs if points[idx][1] < 1e-6])
        self.left_idxs = np.array([idx for idx in self.boundary_idxs if points[idx][0] < 1e-6])
        self.right_idxs = np.array([idx for idx in self.boundary_idxs if points[idx][0] > w-1e-6])
        self.top_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, self.top_idxs))])
        self.bottom_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, self.bottom_idxs))])
        self.left_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, self.left_idxs))])
        self.right_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, self.right_idxs))])
