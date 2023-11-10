import numpy as np
from utils.quadrature import *
from utils.helper import *

# Choose quadrature method
calc_quadrature = simpsons_quadrature

## ELEMENTS ##
# Face elements take integral of 3 node functions on triangle face
# Boundary elements only take integral of 2 node functions on boundary
# Mass matrix = integral of phi_i * phi_j
# Stiffness matrix = integral of grad(phi_i) * grad(phi_j)
# Load vector = integral of f * phi_i

def calculate_element_mass_matrix(element, func):
    P = np.hstack([np.ones((3, 1)), element])
    area = calculate_triangle_area(element)
    f = calc_quadrature(func, element) # evaluate func on element
    return 1/12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * area * f

def calculate_element_stiffness_matrix(element, func):
    P = np.hstack([np.ones((3, 1)), element])
    area = calculate_triangle_area(element)
    phis = np.linalg.solve(P, np.eye(3))[1:].T
    f = calc_quadrature(func, element)
    return phis @ phis.T * area * f

def calculate_element_load_vector(element, func):
    area = calculate_triangle_area(element)
    f = calc_quadrature(func, element)
    return f * area

def calculate_element_boundary_mass_matrix(element, func):
    E = np.linalg.norm(element[0] - element[1])
    f = calc_quadrature(func, element)
    return 1/6 * np.array([[2, 1], [1, 2]]) * E * f

def calculate_element_boundary_load_vector(element, func):
    E = np.linalg.norm(element[0] - element[1])
    res = calc_quadrature(func, element)
    return 1/2 * res * E


## ASSEMBLY ##
# Assemble global matrices/vectors from element (triangle face) contributions 

def assemble_matrix(points, elements, calculate_element_matrix, func=None):
    if func is None:
        func = lambda x: 1
    N = len(points)
    A = np.zeros((N, N))
    for e in elements:
        element = points[e]
        element_matrix = calculate_element_matrix(element, func)
        A[np.ix_(e, e)] += element_matrix
    return A

def assemble_vector(points, elements, calculate_element_vector, func=None):
    if func is None:
        func = lambda x: 1
    N = len(points)
    b = np.zeros(N)
    for e in elements:
        element = points[e]
        b[np.ix_(e)] += calculate_element_vector(element, func)
    return b

# TODO: NEW STUFF
def calculate_hat_gradients(element):
    area = calculate_triangle_area(element)
    a, b, c = [], [], []
    for i in range(3):
        x_j, x_k = element[(i+1)%3], element[(i+2)%3]
        a.append(x_j[0]*x_k[1] - x_k[0]*x_j[1])
        b.append(x_j[1] - x_k[1])
        c.append(x_k[0] - x_j[0])
    a, b, c = np.array([a, b, c]) / (2 * area)
        
    return area, a, b, c


def calculate_element_elastic_stiffness_matrix(element, mu, lamb):
    # outputs 6x6 element stiffness matrix = a(u, v) = int (sigma(u) : epsilon(v)) over element
    D = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + \
        lamb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    area, a, b, c = calculate_hat_gradients(element)
    B = np.array([[b[0], 0, b[1], 0, b[2], 0],
                  [0, c[0], 0, c[1], 0, c[2]],
                  [c[0], b[0], c[1], b[1], c[2], b[2]]])
    return B.T @ D @ B * area

def calculate_element_elastic_mass_matrix(element):
    area = calculate_triangle_area(element)
    M = np.array([[2, 0, 1, 0, 1, 0],
                  [0, 2, 0, 1, 0, 1],
                  [1, 0, 2, 0, 1, 0],
                  [0, 1, 0, 2, 0, 1],
                  [1, 0, 1, 0, 2, 0],
                  [0, 1, 0, 1, 0, 2]]) * area / 12
    return M

def calculate_something(element, mu, lamb):
    # TODO: in development
    D = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + \
        lamb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    area, a, b, c = calculate_hat_gradients(element)
    B = np.array([[b[0], 0, b[1], 0, b[2], 0],
                  [0, c[0], 0, c[1], 0, c[2]],
                  [c[0], b[0], c[1], b[1], c[2], b[2]]])
    return B, D
