import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

# 2d vector operations - faster than numpy
def calculate_dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]

def calculate_norm(vec):
    return (vec[0]**2 + vec[1]**2)**0.5

def calculate_cross(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]

def bump_function(vertices, center, mag=100, size=0.5):
    return np.array([mag*cos(pi/2*np.linalg.norm(point - center)/size) if np.linalg.norm(point - center) < size else 0 for point in vertices])

def calculate_triangle_area(vertices):
    p1, p2, p3 = vertices
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

# Material properties
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

# Printing
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def check_gradient(function, gradient, input_shape):
    u = np.random.random(input_shape)
    computed_gradient = gradient(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_gradient = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = function(u + eps * direction)
            eval_m = function(u - eps * direction)
            numerical_gradient.append((eval_p - eval_m) / (2 * eps))
        numerical_gradient = np.array(numerical_gradient).reshape(computed_gradient.shape)
        # print(f'numerical_gradient: {numerical_gradient} \ncomputed_gradient: {computed_gradient}')
        errors_list.append(np.linalg.norm(numerical_gradient - computed_gradient))
    
    plt.title('Gradient check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()

def check_hessian(gradient, hessian, input_shape):
    u = np.random.random(input_shape)
    computed_hessian = hessian(u)
    eps_list = np.logspace(-10, 0, 20)
    errors_list = []
    for eps in eps_list:
        numerical_hessian = []
        for idx in np.ndindex(input_shape):
            direction = np.zeros(input_shape)
            direction[idx] = 1
            eval_p = gradient(u + eps * direction)
            eval_m = gradient(u - eps * direction)
            numerical_hessian.append((eval_p - eval_m) / (2 * eps))
        numerical_hessian = np.array(numerical_hessian).reshape(computed_hessian.shape)
        errors_list.append(np.linalg.norm(numerical_hessian - computed_hessian))
    
    plt.title('Hessian check')
    plt.plot(eps_list, errors_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eps')
    plt.ylabel('error')
    plt.show()