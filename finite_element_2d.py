import numpy as np
from math import sin, cos, pi
from utils.file_io import *
from utils.plotting import *
from projection_solver import *
from poisson_solver import *
from heat_solver import *
from wave_solver import *
from linear_elastic_solver import *

np.set_printoptions(suppress=True)

# TODO:
# add BC, load func to heat/wave solver
# better plotting for heat/wave
# one time inheritance for matrices
# improve BC specfication in linear elastic and others
# figure out meshes names and improve meshing -> fem project mesh sharing
# add heat/wave equation to README
# utils folder

# gradient plot
# add measures
# add adaptive meshing
# variable time stepping?
# check plane strain vs plane stress


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


def demo_elastic_solver():
    boundary_idxs = list(set(boundary.ravel()))
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
    dirichlet_displacements = np.array([[0, 0] for idx in dirichlet_idxs]) # fixed displacements
    dirichlet_bc = dirichlet_idxs, dirichlet_displacements

    # neumann
    neumann_boundary = right_boundary
    neumann_force = lambda x: np.array([[0, 0.01]]) # boundary force
    neumann_bc = neumann_boundary, neumann_force

    # body force
    body_force = lambda x: np.array([[0, 0.002]])

    solver = LinearElasticSolver(points, faces, boundary, E=E, nu=nu)
    solver.solve(body_force=body_force, dirichlet_bc=None, neumann_bc=None)
    d = solver.result.displacement
    eps_faces = solver.result.eps_faces
    sigma_faces = solver.result.sigma_faces

    new_points = points + d

    # results analysis
    vonmises = np.sqrt(sigma_faces[:, 0]**2 + sigma_faces[:, 1]**2 - sigma_faces[:, 0] * sigma_faces[:, 1] + 3 * sigma_faces[:, 2]**2)

    print('_________________FEM SOLUTION_________________')
    print('Stress (avg):', np.mean(vonmises))
    print('Stress (max):', np.max(vonmises))
    print('Strain (avg):', np.mean(eps_faces))
    
    print()
    # plot_mesh(points, faces, title='Undeformed Mesh', show=True)
    # plot_colored_mesh(new_points, faces, vonmises, title='Deformed Mesh', show=True, cbar_label='von Mises Stress')

    # plot results
    fig, ax = plt.subplots(1, 2)
    plot_mesh(points, faces, ax=ax[0], title='Undeformed Mesh', show=False)
    plot_colored_mesh(new_points, faces, vonmises, title='Deformed Mesh', ax=ax[1], show=False, cbar_label='von Mises Stress')
    
    for a in ax:
        a.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    # MESH
    MESH_FILE = 'meshes/densehole_mesh.pkl'
    points, faces, boundary = load_mesh(MESH_FILE, plot=False)

    # POINTS (alternative) - mesh is generated from triangulation of points
    # POINTS_FILE = 'meshes/points.pkl'
    # points, faces, boundary = load_points(POINTS_FILE, plot=False)

    # for convenience
    points = move_to_origin(points) 
    points = normalize_points(points)
    w, h = np.max(points[:, 0]), np.max(points[:, 1])

    # plot_mesh(points, faces)
    
    # BOUNDARY CONDITIONS
    def K(point): # robin boundary condition weight
        if not point[0] < 1e-6 and not point[1] < 1e-6 and not point[1] > h - 1e-6 and not point[0] > w - 1e-6:
            return 10**6
        return 0

    def g_D(point): # dirichlet boundary condition
        # if point[0] < 1e-6:
        #     return 10
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
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return np.sin(40*(x**2+y**2))/30

    
    # L2 PROJECTION
    projection_solver = ProjectionSolver(points, faces, boundary)
    projection_solver.solve(load_function=cool_f)
    projection_solver.plot_result()

    # POISSON'S EQUATION
    poisson_solver = PoissonSolver(points, faces, boundary)
    poisson_solver.solve(robin_bc=(K, g_D, g_N), load_function=f)
    # poisson_solver.solve(neumann_bc=g_N, load_function=f) # other bc
    # poisson_solver.solve(dirichlet_bc=g_D, load_function=f)
    poisson_solver.plot_result()

    # HEAT EQUATION
    heat_center = np.mean(points, axis=0) + np.array([0.4*w, 0.4*h])
    u_initial = np.array([100*np.cos(np.pi/2*np.linalg.norm(points[i] - heat_center)) if np.linalg.norm(points[i] - heat_center) < 0.2*min(w, h) else 0 for i in range(len(points))])
    heat_solver = HeatSolver(points, faces, boundary, dt=0.01, num_iterations=20)
    heat_solver.solve(u_initial)
    heat_solver.plot_result()    

    # WAVE EQUATION
    dt, num_iterations = 0.01, 15
    wave_center = np.mean(points, axis=0) + np.array([0.4*w, -0.2*h])
    u_initial = np.array([np.cos(np.pi/2*np.linalg.norm(points[i] - wave_center) / (0.2*min(w, h)))**2 if np.linalg.norm(points[i] - wave_center) < 0.2*min(w, h) else 0 for i in range(len(points))])
    dudt_initial = np.zeros(len(points))
    wave_solver = WaveSolver(points, faces, boundary, dt=dt, num_iterations=num_iterations)
    wave_solver.solve(u_initial, dudt_initial)
    wave_solver.plot_result()

    # LINEAR ELASTICS
    # demo_elastic_solver() # TODO: need to refactor/fix

    # TODO: testing gradient
    # gradient = calculate_gradient(points, faces, u)
    # plot_colored_mesh(points, faces, np.linalg.norm(gradient, axis=1), contour=False, title='Gradient of velocity potential')

    print('finite_element_2d.py done')