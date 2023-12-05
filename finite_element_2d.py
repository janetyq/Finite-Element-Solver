import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from base_solver import *
from boundary_conditions import *
from projection_solver import *
from poisson_solver import *
from heat_solver import *
from wave_solver import *
from linear_elastic_solver import *
from topology_optimization import *

from utils.helper import *
from utils.mesh import *

np.set_printoptions(suppress=True)

# TODO:
# detect convergence - research
# add BC specfication to all solvers - WAVE
# improve README, more references
# incorporate mesh data structures better
# test buckling
# efficiency of mesh stuff

# save results with animation together, gif?, save function separate from plot


# intuition for alpha
# improve results saving
# regular mesh generator - make official
# half edge - make part of mesh and w/ insertion

# automatic boundary detection, topo opt remove material

# clean up topo opt code

# adaptive refinement for other equations
# transport equation
# understand units / physical interpretations
# wave energy not conserved
# variable time stepping?
# check plane strain vs plane stress
# efficiency, np stuff
# contour weird after refinement
# residual for different function
# linear elastics - corners seem too pulled
# buckling effect
# label boundary conditions

# plot hovering
# slider animation
# reorder class methods

# FUTURE IDEAS
# higher order elements
# gauss quadrature?
# periodic BC


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    # MESH
    MESH_FILE = '../shared_meshes/regular_mesh3.pkl'
    mesh = Mesh.load(MESH_FILE)
    points, faces, boundary = mesh.get_info()
    boundary_idxs = list(set(boundary.ravel()))

    # mesh.plot()

    w, h = np.max(mesh.points[:, 0]), np.max(mesh.points[:, 1])

    # EXAMPLE LOAD FUNCTIONS
    f = lambda x: 1/((x[0] - 0.5)**2 + (x[1] - 0.5)**2 + 0.01)
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return np.sin(40*(x**2+y**2)) / 10

    # # EXAMPLE BOUNDARY CONDITIONS
    # # no bc, equivalent to neumann = 0 on all boundaries (for poisson and heat)
    # no_bc = BoundaryConditions(mesh)
    
    # # dirichlet u=0 on boundary
    # zero_bc = BoundaryConditions(mesh)
    # zero_bc.add('dirichlet', boundary_idxs, [0 for idx in boundary_idxs])

    # # neumann du = 1 on right boundary
    # right_pull_bc = BoundaryConditions(mesh)
    # right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
    # right_pull_bc.add('neumann', right_idxs, [100 for idx in right_idxs])

    # # mixed dirichlet and neumann
    # mixed_bc = BoundaryConditions(mesh)
    # mixed_bc.add('dirichlet', boundary_idxs, [0 for idx in boundary_idxs])
    # mixed_bc.add('neumann', right_idxs, [1000 for idx in right_idxs])

    # # fluid flow
    # fluid_bc = BoundaryConditions(mesh)
    # inner_idxs = [idx for idx in boundary_idxs if 1e-6 < points[idx][0] < w-1e-6 and 1e-6 < points[idx][1] < h-1e-6]
    # left_idxs = [idx for idx in boundary_idxs if points[idx][0] < 1e-6]
    # right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
    # bottom_idxs = [idx for idx in boundary_idxs if points[idx][1] < 1e-6 and points[idx][0] > 1e-6]
    # fluid_bc.add('dirichlet', inner_idxs, [0 for idx in inner_idxs])
    # fluid_bc.add('neumann', left_idxs, [10 for idx in left_idxs])
    # fluid_bc.add('neumann', bottom_idxs, [10 for idx in bottom_idxs])


    # # L2 PROJECTION
    # projection_solver = ProjectionSolver(mesh)
    # projection_solver.initialize(boundary_conditions=no_bc, load_function=cool_f)
    # projection_solver.solve()
    # projection_solver.plot_result(title='L2 Projection of sin(40*r^2)/10')

    # # POISSON'S EQUATION
    # def test_function(point):
    #     a = 50
    #     x, y = point - np.array([0.5, 0.5])
    #     r2 = x**2 + y**2
    #     return 4*a*a*(1-a*r2)*e**(-a*r2)

    # poisson_solver = PoissonSolver(mesh)
    # poisson_solver.initialize(boundary_conditions=fluid_bc, load_function=None)
    # poisson_solver.solve()
    # poisson_solver.plot_result(gradient=True, contour=5)
    
    # # ADAPTIVE REFINEMENT TEST
    # initial_mesh = poisson_solver.mesh
    # initial_residuals = poisson_solver.calculate_face_residuals()
    # initial_result = poisson_solver.result
    
    # poisson_solver.adaptive_refinement(max_triangles=400)

    # final_mesh = poisson_solver.mesh
    # final_residuals = poisson_solver.calculate_face_residuals()
    # final_result = poisson_solver.result

    # fig, ax = plt.subplots(2, 2)
    # initial_mesh.plot_colored(initial_residuals, title='Initial Residuals', ax=ax[0][0], show=False)
    # final_mesh.plot_colored(final_residuals, title='Final Residuals', ax=ax[0][1], show=False)
    # initial_mesh.plot(title='Initial Mesh', ax=ax[1][0], show=False)
    # final_mesh.plot(title='Final Mesh', ax=ax[1][1], show=False)
    # plt.show()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # initial_mesh.plot_surface(initial_result.u_values, title='Initial Solution', ax=ax1, show=False)
    # ax2 = fig.add_subplot(122, projection='3d')
    # final_mesh.plot_surface(final_result.u_values, title='Final Solution', ax=ax2, show=False)
    # plt.show()

    # # HEAT EQUATION
    # heat_center = np.mean(points, axis=0)
    # u_initial = bump_function(points, heat_center, mag=50, size=0.3*min(w, h)) + 300
    # heat_solver = HeatSolver(mesh, dt=0.001, num_iterations=10)
    # heat_solver.initialize(u_initial, boundary_conditions=right_pull_bc)
    # heat_solver.solve()
    # heat_solver.plot_result(fixed_cbar=True)    

    # # WAVE EQUATION
    # wave_center = np.mean(points, axis=0) + np.array([0.4*w, -0.2*h])
    # u_initial = bump_function(points, wave_center, size=0.2*min(w, h))
    # dudt_initial = np.zeros(len(points))
    # wave_solver = WaveSolver(mesh, dt=0.02, num_iterations=10)
    # wave_solver.initialize(u_initial, dudt_initial, boundary_conditions=no_bc)
    # wave_solver.solve()
    # wave_solver.plot_result()

    # LINEAR ELASTICS
    body_force = lambda x: np.array([[0, 0]]) # gravity
    boundary_conditions = BoundaryConditions(mesh)
    d_idxs = []
    for idx in boundary_idxs:
        if points[idx][0] < 1e-6:
            d_idxs.append(idx)
    boundary_conditions.add('dirichlet', d_idxs, [[0, 0] for idx in d_idxs]) # fixed displacements

    n_idxs = []
    for idx in boundary_idxs:
        if points[idx][0] > w-1e-6 and h/2-1e-3 < points[idx][1] < h/2+1e-3:
            n_idxs.append(idx)
            print(idx, points[idx])
    boundary_conditions.add('neumann', n_idxs, [[0, -50] for idx in n_idxs]) # boundary force

    linear_elastic_solver = LinearElasticSolver(mesh)
    linear_elastic_solver.initialize_material(E=125, nu=0.4)
    linear_elastic_solver.initialize_bc(boundary_conditions=boundary_conditions, load_function=body_force)
    original_result = linear_elastic_solver.solve()
    original_result.plot(title='Original Solution', variable='stress')

    # TOPOLOGY OPTIMIZATION
    rho_initial = np.full(len(mesh.faces), 0.5)
    topopt_results = topology_optimization(
        linear_elastic_solver,  
        E_0=125, nu=0.4, penalization=3,               # material
        boundary_conditions=boundary_conditions, load_function=body_force,  # bc
        rho=rho_initial, target_fraction=0.5,           # density
        num_iterations=10, alpha=0.0002, beta=0.6,      # optimization parameters
        plotting_freq=0                                 # debugging
    )

    # save_topopt_results(topopt_results, name='topopt_1', fps=30)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    original_result.deformed_mesh.plot_colored(np.full(len(mesh.faces), 1), title='Original', ax=ax[0], show=False, cbar_lim=[0, 1], cbar_label='Density')
    final_result.deformed_mesh.plot_colored(final_result.values['rho'], title=f'Optimized ({volume_fraction*100:.0f}% of material)', ax=ax[1], show=False, cbar_lim=[0, 1], cbar_label='Density')
    plt.show()

    # new_mesh = linear_elastic_solver2.mesh.copy()
    # new_mesh.faces = new_mesh.faces[(results[-1].values['rho'] > 0.5)]
    # new_mesh.plot()
    # new_mesh.save('topopt_mesh.pkl')
    
    print('finite_element_2d.py done')