import numpy as np
from math import sin, cos, pi
from utils.file_io import *
from utils.helper import *
from utils.plotting import *
from projection_solver import *
from poisson_solver import *
from heat_solver import *
from wave_solver import *
from linear_elastic_solver import *

np.set_printoptions(suppress=True)

# TODO:
# add BC, load func to wave solver
# improve BC specfication in linear elastic and others
# figure out meshes names and improve meshing -> fem project mesh sharing
# gauss quadrature?
# transport equation
# fix mesh editor
# BC as functions or values at indices
# slider animation

# add measures
# add adaptive meshing
# variable time stepping?
# check plane strain vs plane stress

# minor TODO:
# u vs. u_values

if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    # MESH
    MESH_FILE = 'meshes/densesquare_mesh.pkl'
    points, faces, boundary = load_mesh(MESH_FILE, plot=False)

    # POINTS (alternative) - mesh is generated from triangulation of points
    # POINTS_FILE = 'meshes/points.pkl'
    # points, faces, boundary = load_points(POINTS_FILE, plot=False)

    # for convenience
    points = move_to_origin(points) 
    points = normalize_points(points)
    w, h = np.max(points[:, 0]), np.max(points[:, 1])

    # plot_mesh(points, faces)
    
    # BOUNDARY CONDITIONS - examples
    def g_D(point): # dirichlet boundary condition
        # if point[0] < 1e-6:
        #     return 10
        return 0

    def g_N(point): # neumann boundary condition
        # interior = 1e-6 < point[0] < w - 1e-6 and 1e-6 < point[1] < h - 1e-6
        # if interior:
        #     return 0
        # return 1
        if point[0] < 1e-6:
            return 3
        elif point[0] > w - 1e-6:
            return -1
        return 0

    def W(point): # robin boundary condition weight
        if point[0] < 1e-6:
            return 1000
        return 0

    # LOAD FUNCTION
    # f = lambda x: -np.sqrt(x[0]**2 + x[1]**2)
    f = lambda x: 0 # no load
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return np.sin(40*(x**2+y**2))/30

    # BASE SOLVER - calculates matrices
    base_solver = BaseSolver(points, faces, boundary)
    
    # # L2 PROJECTION
    projection_solver = ProjectionSolver.from_base_solver(base_solver)
    projection_solver.solve(load_function=cool_f)
    projection_solver.plot_result()

    # # POISSON'S EQUATION
    poisson_solver = PoissonSolver.from_base_solver(base_solver)
    poisson_solver.solve(robin_bc=(W, g_D, g_N), load_function=f)
    # poisson_solver.solve(neumann_bc=g_N, load_function=f) # other bc
    # poisson_solver.solve(dirichlet_bc=g_D, load_function=f)
    poisson_solver.plot_result(gradient=True)

    # HEAT EQUATION
    heat_center = np.mean(points, axis=0) + np.array([0.4*w, 0.4*h])
    u_initial = bump_function(points, heat_center, mag=50, size=0.3*min(w, h)) + 300
    heat_solver = HeatSolver.from_base_solver(base_solver, dt=0.001, num_iterations=10)
    heat_solver.solve(u_initial, robin_bc=None, load_function=None)
    heat_solver.plot_result(fixed_cbar=True)    

    # WAVE EQUATION
    wave_center = np.mean(points, axis=0) + np.array([0.4*w, -0.2*h])
    u_initial = bump_function(points, wave_center, size=0.2*min(w, h))
    dudt_initial = np.zeros(len(points))
    wave_solver = WaveSolver.from_base_solver(base_solver, dt=0.02, num_iterations=5)
    wave_solver.solve(u_initial, dudt_initial)
    wave_solver.plot_result()

    # LINEAR ELASTICS
    convenient_indices = ConvenientIndices(points, boundary)
    linear_elastic_solver = LinearElasticSolver(points, faces, boundary, E=100, nu=0.4)
    body_force = lambda x: np.array([[0, 0.002]]) # gravity
    dirichlet_bc = convenient_indices.left_idxs, np.array([[0, 0] for idx in convenient_indices.left_idxs]) # fixed displacements
    neumann_bc = convenient_indices.right_boundary, lambda x: np.array([[0, 0.01]]) # boundary force
    linear_elastic_solver.solve(body_force=body_force, dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
    linear_elastic_solver.plot_result()

    print('finite_element_2d.py done')