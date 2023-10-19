import numpy as np
from math import sin, cos, pi
from file_io import *
from plotting import *
from quadrature import *

# TODO:
# gradient plot
# add measures
# add adaptive meshing

## ELEMENTS ##
# Face elements take integral of 3 node functions on triangle face
# Boundary elements only take integral of 2 node functions on boundary
# Mass matrix = integral of phi_i * phi_j
# Stiffness matrix = integral of grad(phi_i) * grad(phi_j)
# Load vector = integral of f * phi_i

def calculate_element_mass_matrix(element, func):
    P = np.hstack([np.ones((3, 1)), element])
    K = np.abs(np.linalg.det(P) / 2) # area of triangle element
    f = calc_quadrature(func, element) # evaluate func at centroid
    return 1/12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * K * f

def calculate_element_stiffness_matrix(element, func):
    P = np.hstack([np.ones((3, 1)), element])
    K = np.abs(np.linalg.det(P) / 2)
    phis = np.linalg.solve(P, np.eye(3))[1:].T
    f = calc_quadrature(func, element)
    return phis @ phis.T * K * f

def calculate_element_load_vector(element, func):
    K = np.abs(np.linalg.det(np.hstack([np.ones((3, 1)), element])) / 2)
    f = calc_quadrature(func, element)
    return f * K

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

def assemble_matrix(points, faces, calculate_element_matrix, func=None):
    if func is None:
        func = lambda x: 1
    N = len(points)
    A = np.zeros((N, N))
    for face in faces:
        element = points[face]
        element_matrix = calculate_element_matrix(element, func)
        A[np.ix_(face, face)] += element_matrix
    return A

def assemble_vector(points, faces, calculate_element_vector, func=None):
    if func is None:
        func = lambda x: 1
    N = len(points)
    b = np.zeros(N)
    for face in faces:
        element = points[face]
        b[np.ix_(face)] += calculate_element_vector(element, func)
    return b


## SOLVING EQUATIONS ##

def solve_robin(A, M, b, points, boundary):
    R = assemble_matrix(points, boundary, calculate_element_boundary_mass_matrix, K)
    r = assemble_vector(points, boundary, calculate_element_boundary_load_vector, 
                        lambda x: K(x) * g_D(x) + g_N(x))

    u = np.linalg.solve(A + R, b + r)
    return u

def solve_dirichlet(A, M, b, points, boundary):
    N = len(points)
    boundary_idxs = list(set(boundary.ravel()))
    inner_idxs = list(set(range(N)) - set(boundary_idxs))

    u = np.zeros(N)
    boundary_value = np.apply_along_axis(g_D, axis=1, arr=points[boundary_idxs])
    b_temp = b[inner_idxs] - A[np.ix_(inner_idxs, boundary_idxs)] @ boundary_value
    A_temp = A[np.ix_(inner_idxs, inner_idxs)]
    u[inner_idxs] = np.linalg.solve(A_temp, b_temp)
    u[boundary_idxs] = boundary_value
    return u

def solve_neumann(A, M, b, points, boundary):
    N = len(points)
    C = assemble_vector(points, faces, calculate_element_load_vector, lambda x: 1)
    A_temp = np.zeros((N+1, N+1))
    A_temp[:N, :N] = A
    A_temp[:N, N] = C
    A_temp[N, :N] = C
    A_temp[N, N] = 0

    boundary_value = assemble_vector(points, boundary, calculate_element_boundary_load_vector, g_N)
    b_temp = np.append(b + boundary_value, 0)
    u = np.zeros(N)
    soln = np.linalg.solve(A_temp, b_temp)
    u[:N] = soln[:N]
    force = soln[N]

    return u, force

def solve_heat_equation(A, M, b, points, boundary, u_initial, dt=0.1, num_iterations=10, plotting=True):
    u = u_initial.copy()
    colorscale = (min(u), max(u))
    if plotting:
        title = 'Heat Equation Simulation, t = 0'
        plot_colored_mesh(points, faces, u, title=title, contour=False, colorscale=colorscale)
    for i in range(num_iterations):
        # backwards Euler
        u = np.linalg.solve(M + A * dt, M @ u + b * dt)
        print(f'Total heat: {calculate_total_value(points, boundary, u)}')
        colorscale = (colorscale[0], (colorscale[1] + max(u)) / 2) # gradually decrease colorscale
        if plotting:
            title = f'Heat Equation Simulation, t = {dt * (i+1)}'
            plot_colored_mesh(points, faces, u, title=title, contour=False, colorscale=colorscale)
    return u

def solve_projection(A, M, b, points, boundary):
    return np.linalg.solve(M, b)


## MEASURES ##
# TODO: mean value, dirichlet energy

def calculate_total_value(points, boundary, u):
    N = len(points)
    value = 0
    for face in faces:
        element = points[face]
        P = np.hstack([np.ones((3, 1)), element])
        K = np.abs(np.linalg.det(P) / 2)
        u_value = 1/3 * np.mean(u[face])
        value += K * u_value
    return value 

if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    # Choose quadrature rule
    calc_quadrature = simpsons_quadrature

    # MESH
    MESH_FILE = 'meshes/hole_mesh.pkl'
    points, faces, boundary = load_mesh(MESH_FILE, plot=False)

    # POINTS (alternative) - mesh is generated from triangulation of points
    # POINTS_FILE = 'meshes/points.pkl'
    # points, faces, boundary = load_points(POINTS_FILE, plot=False)
    
    # BOUNDARY CONDITIONS
    def K(point): # robin boundary condition weight
        if point[0] < 0.001:
            return 10**6
        return 0

    def g_D(point): # dirichlet boundary condition
        if point[0] < 0.001:
            return 5
        return 0

    def g_N(point): # neumann boundary condition
        return -1
        if point[0] < 0.001:
            return 8
        return 0

    # LOAD FUNCTION
    # f = lambda x: -np.sqrt(x[0]**2 + x[1]**2)
    f = lambda x: 0 # no load

    # ASSEMBLY
    A = assemble_matrix(points, faces, calculate_element_stiffness_matrix)
    M = assemble_matrix(points, faces, calculate_element_mass_matrix)
    b = assemble_vector(points, faces, calculate_element_load_vector, f)

    # SOLVING - projection, different boundary conditions
    # u = solve_projection(A, M, b, points, boundary)
    # u = solve_dirichlet(A, M, b, points, boundary)
    u = solve_robin(A, M, b, points, boundary)
    # u, force = solve_neumann(A, M, b, points, boundary)
    plot_colored_mesh(points, faces, u, contour=True)
    # plot_surface_mesh(points, faces, u, contour=True)

    # SOLVING - heat equation
    heat_center = np.mean(points, axis=0) + np.array([0.4, -0.2])
    u_initial = np.array([100 if np.linalg.norm(points[i] - heat_center) < 0.3 else 0 for i in range(len(points))])
    u = solve_heat_equation(A, M, b, points, boundary, u_initial, dt=0.02, num_iterations=10, plotting=True)

    print('finite_element_2d.py done')