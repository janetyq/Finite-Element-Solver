import numpy as np
from math import sin, cos, pi
from file_io import *
from plotting import *
from quadrature import *

np.set_printoptions(suppress=True)


# TODO:
# ORGANIZE CODE - currently in preliminary development
# gradient plot
# add measures
# add adaptive meshing
# variable time stepping?
# check plane strain vs plane stress

## ELEMENTS ##
# Face elements take integral of 3 node functions on triangle face
# Boundary elements only take integral of 2 node functions on boundary
# Mass matrix = integral of phi_i * phi_j
# Stiffness matrix = integral of grad(phi_i) * grad(phi_j)
# Load vector = integral of f * phi_i

def calculate_element_mass_matrix(element, func):
    P = np.hstack([np.ones((3, 1)), element])
    K = np.abs(np.linalg.det(P) / 2) # area of triangle element
    f = calc_quadrature(func, element) # evaluate func on element
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
    # imposed boumdary values
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

def solve_heat_equation(A, M, b, points, boundary, u_initial, dt=0.1, num_iterations=10, plotting=True, plotting_freq=1):
    u = u_initial.copy()
    colorscale = (min(u), max(u))
    if plotting:
        title = 'Heat Equation Simulation, t = 0'
        plot_colored_mesh(points, faces, u, title=title, contour=False, colorscale=colorscale)
    for i in range(num_iterations):
        # backwards Euler
        u = np.linalg.solve(M + A * dt, M @ u + b * dt)
        print(f'Total heat: {calculate_total_value(points, boundary, u)}')
        # colorscale = (colorscale[0], (colorscale[1] + max(u)) / 2) # gradually decrease colorscale
        if i % plotting_freq == 0 and plotting:
            title = f'Heat Equation Simulation, t = {dt * (i+1)}'
            plot_colored_mesh(points, faces, u, title=title, contour=False, colorscale=None)
    return u

def solve_wave_equation(A, M, b, points, boundary, u_initial, dudt_initial, dt=0.1, num_iterations=10, plotting=True):
    c = 1 # wave speed
    N = len(points)
    x = np.block([u_initial, dudt_initial])
    A_left = np.block([[M, -dt/2 * M],
                       [c**2 * dt/2 * A, M]])
    A_right = np.block([[M, dt/2 * M],
                        [-c**2 * dt/2 * A, M]])

    b_right = np.block([np.zeros_like(b), dt/2 * (b + np.roll(b, -1))])

    if plotting:
        u = x[:N]
        dudt = x[N:]
        title = f'Wave Equation Simulation, t = 0'
        plot_colored_mesh(points, faces, u, title=title, contour=False)

    for i in range(num_iterations):
        x = np.linalg.solve(A_left, A_right @ x + b_right)
        if plotting:
            u = x[:N]
            dudt = x[N:]
            title = f'Wave Equation Simulation, t = {dt * (i+1)}'
            plot_colored_mesh(points, faces, u, title=title, contour=False)

    u = x[:N]
    dudt = x[N:]
    return u, dudt


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

def normalize_points(points):
    points_range = np.max(points, axis=0) - np.min(points, axis=0)
    max_range = np.max(points_range)
    range = np.array([max_range, max_range])
    return (points - np.min(points, axis=0)) / range

# NEW
def calculate_hat_gradients(element):
    area = np.abs(np.linalg.det(np.hstack([np.ones((3, 1)), element])) / 2)
    a = []
    b = []
    c = []
    for i in range(3):
        x_j, x_k = element[(i+1)%3], element[(i+2)%3]
        a.append(x_j[0]*x_k[1] - x_k[0]*x_j[1])
        b.append(x_j[1] - x_k[1])
        c.append(x_k[0] - x_j[0])
    a, b, c = np.array([a, b, c]) / (2 * area)

    # test for correctness
    for i in range(3):
        x1, x2 = element[i]
        result = a + b*x1 + c*x2
        assert(np.allclose(result[i], 1))
        
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
    area = np.abs(np.linalg.det(np.hstack([np.ones((3, 1)), element])) / 2)
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

def elastic_solver(points, faces, boundary, body_force=None, dirichlet_bc=None, neumann_bc=None, E=1, nu=0.2, alpha=0):
    N = len(points)
    # force = lambda x: np.array([0, 0]) if force is None else force
    mu, lamb = calculate_Lame(E, nu)
    boundary_idxs = set(boundary.ravel())
    
    K = np.zeros((2*N, 2*N)) # stiffness matrix
    M = np.zeros((2*N, 2*N)) # mass matrix
    B = np.zeros((2*N, 2*N)) # gradient matrix
    D = np.zeros((2*N, 2*N)) # elasticity matrix
    F = np.zeros(2*N) # load vector
    for e in faces: 
        element = points[e]
        element_mass_matrix = calculate_element_elastic_mass_matrix(element)
        element_stiffness_matrix = calculate_element_elastic_stiffness_matrix(element, mu, lamb)
        forces = np.apply_along_axis(body_force, axis=1, arr=element).flatten()
        

        e_idxs = np.array([2*e, 2*e+1]).T.flatten()
        M[np.ix_(e_idxs, e_idxs)] += element_mass_matrix
        K[np.ix_(e_idxs, e_idxs)] += element_stiffness_matrix
        F[np.ix_(e_idxs)] += element_mass_matrix @ forces

        # TODO: in development
        # B, D = calculate_something(element, mu, lamb)
        # dT = 10
        # F[np.ix_(e_idxs)] += alpha * (3*lamb + 2*mu) * (dT) * B

    if neumann_bc is not None:
        neumann_boundary, neumann_force = neumann_bc
        for b in neumann_boundary:
            element = points[b]
            E = np.linalg.norm(element[0] - element[1])
            forces = np.apply_along_axis(neumann_force, axis=1, arr=element).flatten()
            b_idxs = np.array([2*b, 2*b+1]).T.flatten()
            F[np.ix_(b_idxs)] += 1/2 * forces * E

    d = np.zeros(2*N) # displacement vector

    # boundary conditions
    if dirichlet_bc is not None:
        fixed_idxs, fixed_displacements = dirichlet_bc

        fixed = np.array([2*fixed_idxs, 2*fixed_idxs+1]).T.flatten()
        d[fixed] = np.array(fixed_displacements).flatten()
        free = list(set(range(2*N)) - set(fixed))

        F_temp = F[free] - K[np.ix_(free, fixed)] @ d[fixed]
        K_temp = K[np.ix_(free, free)]
        d[free] = np.linalg.solve(K_temp, F_temp)
    else:
        d = np.linalg.solve(K, F) # is this allowed

    eps_faces = np.zeros((len(faces), 3))
    sigma_faces = np.zeros((len(faces), 3))

    for face_idx, face in enumerate(faces):
        element = points[face]
        e_idxs = np.array([2*face, 2*face+1]).T.flatten()
        B, D = calculate_something(element, mu, lamb)
        eps = B @ d[e_idxs]
        sigma = D @ B @ d[e_idxs]
        eps_faces[face_idx] = eps
        sigma_faces[face_idx] = sigma
        pass

    return d.reshape((-1, 2)), eps_faces, sigma_faces
    

def calculate_Lame(E, nu):
    # E = 2 * mu * (1 + nu)
    # nu = E / (2 * mu) - 1
    mu = E / (2 * (1 + nu))
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lamb

# for debugging convenience
def print_min_max(array):
    print(f'Min: {np.min(array)}')
    print(f'Max: {np.max(array)}')

def move_to_origin(points):
    return points - np.min(points, axis=0)


def demo_elastic_solver():
    # convenient idxs
    top_idxs = np.array([idx for idx in boundary_idxs if points[idx][1] > h-1e-6])
    bottom_idxs = np.array([idx for idx in boundary_idxs if points[idx][1] < 1e-6])
    left_idxs = np.array([idx for idx in boundary_idxs if points[idx][0] < 1e-6])
    right_idxs = np.array([idx for idx in boundary_idxs if points[idx][0] > w-1e-6])
    top_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, top_idxs))])
    bottom_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, bottom_idxs))])
    left_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, left_idxs))])
    right_boundary = np.array([boundary for boundary in boundary if np.all(np.isin(boundary, right_idxs))])

    # material properties
    E, nu = 100, 0.4
    print(f'Material properties: E = {E}, nu = {nu}')

    # dirichlet
    dirichlet_idxs = left_idxs
    dirichlet_displacements = [[0, 0] for idx in dirichlet_idxs] # fixed displacements
    dirichlet_bc = dirichlet_idxs, dirichlet_displacements

    # neumann
    neumann_boundary = right_boundary
    neumann_force = lambda x: np.array([[0, 0.01]]) # boundary force
    neumann_bc = neumann_boundary, neumann_force

    # body force
    body_force = lambda x: np.array([[0, 0.002]])

    results = elastic_solver(points, faces, boundary, 
                            body_force=body_force, 
                            dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc, 
                            E=E, nu=nu)

    d, eps_faces, sigma_faces = results
    new_points = points + d

    # results analysis
    vonmises = np.sqrt(sigma_faces[:, 0]**2 + sigma_faces[:, 1]**2 - sigma_faces[:, 0] * sigma_faces[:, 1] + 3 * sigma_faces[:, 2]**2)

    print('_________________FEM SOLUTION_________________')
    print('Stress (avg):', np.mean(vonmises))
    print('Stress (max):', np.max(vonmises))
    print('Strain (avg):', np.mean(eps_faces))
    
    print()
    plot_mesh(points, faces, title='Undeformed Mesh', show=True)
    plot_colored_mesh(new_points, faces, vonmises, title='Deformed Mesh', show=True, cbar_label='von Mises Stress')

    # plot results
    fig, ax = plt.subplots(1, 2)
    plot_mesh(points, faces, ax=ax[0], title='Undeformed Mesh', show=False)
    plot_colored_mesh(new_points, faces, vonmises, title='Deformed Mesh', ax=ax[1], show=False, cbar_label='von Mises Stress')
    
    for a in ax:
        a.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    # Choose quadrature rule
    calc_quadrature = simpsons_quadrature

    # MESH
    MESH_FILE = 'meshes/bracedcantilever_mesh.pkl'
    points, faces, boundary = load_mesh(MESH_FILE, plot=False)

    # POINTS (alternative) - mesh is generated from triangulation of points
    # POINTS_FILE = 'meshes/points.pkl'
    # points, faces, boundary = load_points(POINTS_FILE, plot=False)

    # for convenience
    points = move_to_origin(points) 
    # points = normalize_points(points)
    w, h = np.max(points[:, 0]), np.max(points[:, 1])

    # plot_mesh(points, faces)
    boundary_idxs = list(set(boundary.ravel()))

    # linear elastics
    demo_elastic_solver()
    
    # BOUNDARY CONDITIONS
    def K(point): # robin boundary condition weight
        if not point[0] < 1e-6 and not point[1] < 1e-6 and not point[1] > h - 1e-6 and not point[0] > w - 1e-6:
            return 10**6
        return 0

    def g_D(point): # dirichlet boundary condition
        return 0

    def g_N(point): # neumann boundary condition
        if point[0] < 1e-6:
            return 3
        elif point[0] > w - 1e-6:
            return -1
        return 0

    # LOAD FUNCTION
    # f = lambda x: -np.sqrt(x[0]**2 + x[1]**2)
    f = lambda x: 0 # no load

    # def f(point):
    #     x, y = point - np.array([0.5, 0.5])
    #     return np.sin(40*(x**2+y**2))/30

    # ASSEMBLY
    A = assemble_matrix(points, faces, calculate_element_stiffness_matrix)
    M = assemble_matrix(points, faces, calculate_element_mass_matrix)
    b = assemble_vector(points, faces, calculate_element_load_vector, f)

    # SOLVING - projection, different boundary conditions
    # u = solve_projection(A, M, b, points, boundary)
    # u = solve_dirichlet(A, M, b, points, boundary)
    # u = solve_robin(A, M, b, points, boundary)
    # u, force = solve_neumann(A, M, b, points, boundary)
    # plot_colored_mesh(points, faces, u, contour=True, title='Contour plot of velocity potential \n(Poisson\'s equation w/ Robin BC)')
    # plot_surface_mesh(points, faces, u, contour=True)

    # SOLVING - heat equation
    heat_center = np.mean(points, axis=0) + np.array([0.4*w, 0.4*h])
    # u_initial = np.array([100 if np.linalg.norm(points[i] - heat_center) < 0.2*min(w, h) else 0 for i in range(len(points))])
    u_initial = np.array([100 if points[i][1] < 1e-6 else 0 for i in range(len(points))])
    u = solve_heat_equation(A, M, b, points, boundary, u_initial, dt=0.01, num_iterations=10, plotting=True, plotting_freq=4)

    # SOLVING - wave equation, TODO: in development
    # wave_center = np.mean(points, axis=0) + np.array([0.4, -0.2])
    # u_initial = np.zeros(len(points))
    # dudt_initial = np.zeros(len(points))
    # u, dudt = solve_wave_equation(A, M, b, points, boundary, u_initial, dudt_initial, dt=0.02, num_iterations=10, plotting=True)

    print('finite_element_2d.py done')