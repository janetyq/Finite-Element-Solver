import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from BoundaryConditions import *

from Solver import *
from TopologyOptimizer import *

from utils.helper import *
from Mesh import *

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
    # MESH_FILE = 'meshes/hole1000_mesh.pkl'
    MESH_FILE = 'meshes/square1000_mesh.pkl'
    # MESH_FILE = 'meshes/regular_mesh3.pkl'
    # MESH_FILE = 'meshes/easy_rectangle.pkl'
    mesh = Mesh.load(MESH_FILE)
    points, faces, boundary = mesh.get_info()
    boundary_idxs = list(set(boundary.ravel()))
    # mesh.plot()

    mesh.points -= np.min(mesh.points, axis=0) # move mesh to origin
    w, h = np.max(mesh.points[:, 0]), np.max(mesh.points[:, 1])

    # EXAMPLE LOAD FUNCTIONS
    f = lambda x: 1/((x[0] - 0.5)**2 + (x[1] - 0.5)**2 + 0.01)
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return np.sin(40*(x**2+y**2)) / 10

    # # EXAMPLE BOUNDARY CONDITIONS
    # no bc, equivalent to neumann = 0
    no_bc = BoundaryConditions(mesh)
    
    # # dirichlet u=0 on boundary
    zero_bc = BoundaryConditions(mesh)
    zero_bc.add('dirichlet', boundary_idxs, [0 for idx in boundary_idxs])

    # # both dirichlet and neumann
    # flow out of right
    mixed_bc = BoundaryConditions(mesh)
    right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
    other_idxs = [idx for idx in boundary_idxs if idx not in right_idxs]
    mixed_bc.add('neumann', right_idxs, [1000 for idx in right_idxs])
    mixed_bc.add('dirichlet', other_idxs, [0 for idx in other_idxs])

    # fluid flow
    # fluid_bc = BoundaryConditions(mesh)
    # inner_idxs = [idx for idx in boundary_idxs if 1e-6 < points[idx][0] < w-1e-6 and 1e-6 < points[idx][1] < h-1e-6]
    # left_idxs = [idx for idx in boundary_idxs if points[idx][0] < 1e-6]
    # right_idxs = [idx for idx in boundary_idxs if points[idx][0] > w-1e-6]
    # bottom_idxs = [idx for idx in boundary_idxs if points[idx][1] < 1e-6 and points[idx][0] > 1e-6]
    # fluid_bc.add('dirichlet', inner_idxs, [0 for idx in inner_idxs])
    # fluid_bc.add('neumann', left_idxs, [10 for idx in left_idxs])
    # fluid_bc.add('neumann', bottom_idxs, [10 for idx in bottom_idxs])

    # # L2 PROJECTION
    equation = Equation('projection')
    new_solver = Solver(mesh, equation, no_bc, cool_f)
    new_solver.solve()
    new_solver.solution.plot_surface('u', title='L2 Projection')

    # # POISSON'S EQUATION
    def test_function(point):
        a = 50
        x, y = point - np.array([0.5, 0.5])
        r2 = x**2 + y**2
        return 4*a*a*(1-a*r2)*e**(-a*r2)

    # equation = Equation('poisson')
    # new_solver = Solver(mesh, equation, zero_bc, test_function)
    # new_solver.solve()
    # new_solver.solution.plot_colored('u', title='Poisson Solution', contour=5)
    
    # # ADAPTIVE REFINEMENT TEST
    # initial_mesh = new_solver.mesh
    # initial_result = new_solver.solution
    # initial_residuals = initial_result.values['face_residuals']

    # new_solver.adaptive_refinement(max_iters=1, max_triangles=2000)
    # new_solver.solve()
    # final_mesh = new_solver.mesh
    # final_result = new_solver.solution
    # final_residuals = final_result.values['face_residuals']

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

    # HEAT EQUATION
    heat_center = np.mean(points, axis=0)
    u_initial = bump_function(points, heat_center, mag=50, size=0.3*min(w, h)) + 300
    
    equation = Equation('heat', {'u_initial': u_initial.copy(), 'iters': 50, 'dt': 0.01})
    solver = Solver(mesh, equation, mixed_bc)
    solver.solve()
    # solver.solution.plot_colored('u_values', idx=5, contour=20, show=True)
    solver.solution.plot_animation('u_values')
   

    # # WAVE EQUATION
    wave_center = np.mean(points, axis=0)
    u_initial = bump_function(points, wave_center, size=0.2*min(w, h))
    dudt_initial = np.zeros(len(points))
    
    equation = Equation('wave', {'u_initial': u_initial, 'dudt_initial': dudt_initial, 'c': 1, 'dt': 0.02, 'iters': 10})
    solver = Solver(mesh, equation, None)
    solver.solve()
    solver.solution.plot_animation('u_values')


    # LINEAR ELASTICS
    def right_force(point):
        if point[0] > w-1e-6:
            return np.array([[0, -100]])
        return np.array([[0, 0]])

    beam_bc = BoundaryConditions(mesh)
    left_idxs = [idx for idx in boundary_idxs if points[idx][0] < 1e-6]
    beam_bc.add('dirichlet', left_idxs, [[0, 0] for idx in left_idxs])

    equation = Equation('linear_elastic', {'E': 125, 'nu': 0.4})
    solver = Solver(mesh, equation, beam_bc, load_function=right_force)
    solver.solve()
    # solver.solution.plot_deformed('stress')

    # TOPOLOGY OPTIMIZATION
    topoptimizer = TopologyOptimizer(solver, {'iters': 50, 'volume_frac': 0.5, 'alpha': 0.002, 'beta': 0.6})
    topoptimizer.solve()
    topoptimizer.solution.plot_animation('rhos')

    # save_topopt_results(topopt_results, name='topopt_1', fps=30)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # original_result.deformed_mesh.plot_colored(np.full(len(mesh.faces), 1), title='Original', ax=ax[0], show=False, cbar_lim=[0, 1], cbar_label='Density')
    # final_result.deformed_mesh.plot_colored(final_result.values['rho'], title=f'Optimized ({volume_fraction*100:.0f}% of material)', ax=ax[1], show=False, cbar_lim=[0, 1], cbar_label='Density')
    # plt.show()

    # new_mesh = linear_elastic_solver2.mesh.copy()
    # new_mesh.faces = new_mesh.faces[(results[-1].values['rho'] > 0.5)]
    # new_mesh.plot()
    # new_mesh.save('topopt_mesh.pkl')
    
    print('finite_element_2d.py done')