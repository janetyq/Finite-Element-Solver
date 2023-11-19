import numpy as np
from math import sin, cos, pi

from projection_solver import *
from poisson_solver import *
from heat_solver import *
from wave_solver import *
from linear_elastic_solver import *
from boundary_conditions import *

from utils.base_solver import *
from utils.helper import *
from utils.mesh import *

np.set_printoptions(suppress=True)

# TODO:
# improve BC
# add BC specfication to all solvers
# transport equation
# understand units
# why robin when you can do both neumann and dirichlet

# difference between elastic (2d) and the rest (1d)
# wave energy not conserved
# add adaptive meshing
# variable time stepping?
# check plane strain vs plane stress

# FUTURE IDEAS
# higher order elements


# QUICKER STUFF
# ADD BC separately and load function, call solve to solve
# add refinement to solve
# residual for different function
# inner idxs used?
# plot hovering
# more exact meshing num triangles
# write equation for each
# u vs. u_values
# gauss quadrature?
# slider animation

# NOW
# fix BC add to solvers separately
# add load func to solver separately
# call solve without params
# adaptive refinement with num iterations



if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    # MESH
    MESH_FILE = '../shared_meshes/square100_mesh.pkl'
    mesh = Mesh.load(MESH_FILE)
    points, faces, boundary = mesh.get_info()
    boundary_idxs = list(set(boundary.ravel()))

    # POINTS (alternative) - mesh is generated from triangulation of points
    # POINTS_FILE = 'meshes/points.pkl'
    # points, faces, boundary = load_points(POINTS_FILE, plot=False)

    # for convenience
    # points = move_to_origin(points) 
    # points = normalize_points(points)
    w, h = np.max(points[:, 0]), np.max(points[:, 1])

    # plot_mesh(points, faces)
    
    # BOUNDARY CONDITIONS - examples
    def g_D(point): # dirichlet boundary condition
        # if point[0] < 1e-6:
        #     return 10
        return 0

    def g_N(point): # neumann boundary condition
        return -1
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
    f = lambda x: 1/((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
    # f = lambda x: 0 # no load
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return np.sin(40*(x**2+y**2))/30

    initial_mesh = mesh.copy()

    # L2 PROJECTION
    projection_solver = ProjectionSolver(mesh)
    projection_solver.solve(load_function=cool_f)
    projection_solver.plot_result()

    # # POISSON'S EQUATION
    poisson_solver = PoissonSolver(mesh)
    # poisson_solver.solve(robin_bc=(W, g_D, g_N), load_function=f)
    # poisson_solver.solve(neumann_bc=g_N, load_function=f) # other bc
    poisson_solver.solve(dirichlet_bc=g_D, load_function=f)
    poisson_solver.plot_result(gradient=True, contour=5)
    initial_residuals = poisson_solver.calculate_face_residuals()
    
    poisson_solver.adaptive_refinement(3)

    final_mesh = poisson_solver.mesh
    final_residuals = poisson_solver.calculate_face_residuals()

    fig, ax = plt.subplots(2, 2)
    initial_mesh.plot_colored(initial_residuals, title='Initial Residuals', ax=ax[0][0], show=False)
    final_mesh.plot_colored(final_residuals, title='Final Residuals', ax=ax[0][1], show=False)
    initial_mesh.plot(ax=ax[1][0], show=False)
    final_mesh.plot(ax=ax[1][1], show=False)
    plt.show()

    # poisson_solver2.solve(dirichlet_bc=g_D, load_function=f)
    # poisson_solver2.plot_result(gradient=True)
    

    # HEAT EQUATION
    heat_center = np.mean(points, axis=0) + np.array([0.4*w, 0.4*h])
    u_initial = bump_function(points, heat_center, mag=50, size=0.3*min(w, h)) + 300
    heat_solver = HeatSolver(mesh, dt=0.001, num_iterations=10)
    heat_solver.solve(u_initial, robin_bc=None, load_function=None)
    heat_solver.plot_result(fixed_cbar=True)    

    # WAVE EQUATION
    wave_center = np.mean(points, axis=0) + np.array([0.4*w, -0.2*h])
    u_initial = bump_function(points, wave_center, size=0.2*min(w, h))
    dudt_initial = np.zeros(len(points))
    wave_solver = WaveSolver(mesh, dt=0.02, num_iterations=6)
    wave_solver.solve(u_initial, dudt_initial)
    wave_solver.plot_result()


    # LINEAR ELASTICS
    body_force = lambda x: np.array([[0, 0]]) # gravity
    boundary_conditions = BoundaryConditions(points, boundary, boundary_idxs)
    boundary_conditions.set_dirichlet_bc(boundary_conditions.convenient_indices.left_idxs, np.array([[0, 0] for idx in boundary_conditions.convenient_indices.left_idxs])) # fixed displacements
    def right_force(point):
        if point[0] > w - 1e-6:
            return [10, 0]
        return [0, 0]
    boundary_conditions.set_neumann_bc_func(right_force) # boundary force
    # boundary_conditions.set_neumann_bc(boundary_conditions.convenient_indices.right_idxs, [np.array([10, 0]) for idx in boundary_conditions.convenient_indices.right_idxs])

    linear_elastic_solver = LinearElasticSolver(mesh, E=100, nu=0.4)
    linear_elastic_solver.solve(body_force=body_force, boundary_conditions=boundary_conditions)
    linear_elastic_solver.plot_result()

    print('finite_element_2d.py done')