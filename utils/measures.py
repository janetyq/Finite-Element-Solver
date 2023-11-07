import numpy as np

## MEASURES ##
# TODO
# mean value, dirichlet energy
# verify gradient works

def calculate_total_value(points, faces, boundary, u):
    N = len(points)
    value = 0
    for face in faces:
        element = points[face]
        P = np.hstack([np.ones((3, 1)), element])
        K = np.abs(np.linalg.det(P) / 2)
        u_value = 1/3 * np.mean(u[face])
        value += K * u_value
    return value 

def calculate_gradient(points, faces, u):
    N = len(points)
    grad = np.zeros((N, 2))
    for face in faces:
        element = points[face]
        P = np.hstack([np.ones((3, 1)), element])
        K = np.abs(np.linalg.det(P) / 2)
        u_value = 1/3 * np.mean(u[face])
        grad[face] += K * u_value
    return grad