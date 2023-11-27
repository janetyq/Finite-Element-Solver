import numpy as np
from utils.quadrature import *
from utils.helper import *

# Choose quadrature method
calc_quadrature = simpsons_quadrature

def Enu_to_Lame(E, nu):
    # mu - shear modulus, lambda - Lame constant
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lamb

def Lame_to_Enu(mu, lamb):
    # E - Young's modulus, nu - Poisson's ratio
    E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
    nu = lamb / (2 * (lamb + mu))
    return E, nu

# TODO:
# confict between continous func -> quadrature and discrete func -> np apply

## ELEMENTS ##
# Face elements take integral of 3 node functions on triangle face
# Boundary elements only take integral of 2 node functions on boundary
# Mass matrix = integral of phi_i * phi_j
# Stiffness matrix = integral of grad(phi_i) * grad(phi_j)
# Load vector = integral of f * phi_i

def calculate_element_mass_matrix(element, func, dim=1):
    area = calculate_triangle_area(element)
    f = calc_quadrature(func, element)
    e = len(element)
    M = np.zeros((dim*e, dim*e))
    M[::dim, ::dim] = 1
    M += np.eye(dim*e)
    return 1/12 * area * f * M

def calculate_element_stiffness_matrix(element, func, dim=1):
    if dim == 1: # TODO collapse
        P = np.hstack([np.ones((3, 1)), element])
        area = calculate_triangle_area(element)
        phis = np.linalg.solve(P, np.eye(3))[1:].T
        f = calc_quadrature(func, element) if func is not None else 1
        return phis @ phis.T * area * f
    else: # dim == 2
        # outputs 6x6 element stiffness matrix = a(u, v) = int (sigma(u) : epsilon(v)) over element
        mu, lamb = Enu_to_Lame(func[0], func[1]) # TODO: make space varying?
        D = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + \
            lamb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        area, a, b, c = calculate_hat_gradients(element)
        B = np.array([[b[0], 0, b[1], 0, b[2], 0],
                    [0, c[0], 0, c[1], 0, c[2]],
                    [c[0], b[0], c[1], b[1], c[2], b[2]]])
        return B.T @ D @ B * area

def calculate_element_load_vector(element, func, dim=1):
    mass_matrix = calculate_element_mass_matrix(element, func=lambda x: 1, dim=dim)
    load = np.apply_along_axis(func, axis=1, arr=element).flatten()
    return mass_matrix @ load

def calculate_element_boundary_mass_matrix(element, func, dim=1):
    E = np.linalg.norm(element[0] - element[1])
    f = calc_quadrature(func, element)
    return 1/6 * np.array([[2, 1], [1, 2]]) * E * f

def calculate_element_boundary_load_vector(element, func, dim=1):
    E = np.linalg.norm(element[0] - element[1])
    res = np.array([func(point) for point in element]).flatten()
    return 1/2 * res * E


## ASSEMBLY ##
# Assemble global matrices/vectors from element (triangle face) contributions 

def assemble_matrix(points, elements, calculate_element_matrix, func=None, dim=1):
    if func is None:
        func = lambda x: 1
    N = len(points)
    A = np.zeros((dim * N, dim * N))
    for e in elements:
        element = points[e]
        e_idxs = np.array([dim*e + i for i in range(dim)]).T.flatten()
        element_matrix = calculate_element_matrix(element, func, dim=dim)
        A[np.ix_(e_idxs, e_idxs)] += element_matrix
    return A

def assemble_matrix2(points, elements, calculate_element_matrix, func=None, dim=1): #TODO: remove
    if func is None:
        func = lambda x: 1
    N = len(points)
    A = np.zeros((dim * N, dim * N))
    for e_idx, e in enumerate(elements):
        element = points[e]
        e_idxs = np.array([dim*e + i for i in range(dim)]).T.flatten()
        E, nu = func
        E_elt = E[e_idx]
        element_matrix = calculate_element_matrix(element, [E_elt, nu], dim=dim)
        A[np.ix_(e_idxs, e_idxs)] += element_matrix
    return A

def assemble_vector(points, elements, calculate_element_vector, func, dim=1):
    N = len(points)
    b = np.zeros(dim*N)
    for e in elements:
        element = points[e]
        e_idxs = np.array([dim*e + i for i in range(dim)]).T.flatten()
        b[np.ix_(e_idxs)] += calculate_element_vector(element, func, dim=dim)
    return b

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

# TODO: in development
def calculate_B(element, func, dim=2):
    area, a, b, c = calculate_hat_gradients(element)
    B = np.array([[b[0], 0, b[1], 0, b[2], 0],
                  [0, c[0], 0, c[1], 0, c[2]],
                  [c[0], b[0], c[1], b[1], c[2], b[2]]])
    return B

def calculate_D(element, func, dim=2):
    mu, lamb = func
    D = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]) + \
        lamb * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    return D
