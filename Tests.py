import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from utils.helper import *

from BoundaryConditions import *
from Mesh import *
from Plotter import *
from Solver import *
from TopologyOptimizer import *
from EnergySolver import *

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

def test_plot_mesh():
    plotter = Plotter(mesh, options={'title': 'Mesh'})
    plotter.plot_mesh(mode='wireframe', color_vertices=[('red', mesh.boundary_idxs, 'boundary')])
    plotter.plot_mesh(mode='solid')

def test_l2_projection():
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return [np.sin(40*(x**2+y**2))]
    equation = Equation('projection')
    bc = BoundaryConditions(mesh)
    bc.add_force(cool_f)
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    solution.plot('u', mode='surface', options={'title': 'L2 Projection'})

def test_poisson_equation():
    equation = Equation('poisson')
    bc = BoundaryConditions(mesh)
    bc.add('dirichlet', mesh.boundary_idxs, [0])
    bc.add_force(lambda point: [1])
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    solution.plot('u', mode='surface', options={'title': 'Poisson Solution'})
    gradient = solution.calculate_gradient('u')
    Plotter(mesh, options={'title': 'Gradient'}).plot_values(gradient, mode='arrows')

def test_heat_equation():
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.5*min(w, h)) + 300
    
    equation = Equation('heat', {'u_initial': u_initial.copy(), 'iters': 5, 'dt': 0.01})
    solver = Solver(mesh, equation)
    solution = solver.solve()
    u_values = solution.get_values('u_values')
    t_values = solution.get_values('t_values')

    plotter = Plotter(mesh, options={'title': 'Heat Equation', 'cbar_lim': (300, 330), 'cbar_label': 'Temperature'})
    plotter.plot_animation(u_values, t_values, mode='surface')
    plotter.options['save'] = 'results/heat.gif'
    plotter.plot_animation(u_values, t_values, mode='colored')

def test_wave_equation(): # TODO: Wave energy not fully implemented
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    wave_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, wave_center, size=0.3*min(w, h))
    dudt_initial = np.zeros(len(mesh.vertices))
    
    equation = Equation('wave', {'u_initial': u_initial, 'dudt_initial': dudt_initial, 'c': 1, 'dt': 0.04, 'iters': 10})
    solver = Solver(mesh, equation)
    solution = solver.solve()

    last_value = solution.get_values('u_values', iter_idx=-1)
    all_values = solution.get_values('u_values')
    plotter = Plotter(mesh, options={'title': 'Wave Equation'})
    plotter.plot_values(last_value)
    plotter.options['save'] = 'results/wave.gif'
    plotter.plot_animation(all_values, mode='surface')

def test_linear_elastic():
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    print(w, h)
    bc = BoundaryConditions(mesh)
    left_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] < 1e-6]
    right_middle_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] > w-1e-6 and 0.2 < mesh.vertices[v_idx][1] < 0.8]
    bc.add('dirichlet', left_idxs, [0, 0])
    bc.add('neumann', right_middle_idxs, [50, 0]) # stress
    # bc.plot()

    equation = Equation('linear_elastic', {'E': 200, 'nu': 0.4})
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()

    solver.solution.plot('stress', deformed=True)

def test_topology_optimization():
    def down_force(point):
        return np.array([0, -0.5])

    bc = BoundaryConditions(mesh)
    left_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] < 1e-6]
    bc.add('dirichlet', left_idxs, [0, 0])
    bc.add_force(down_force)

    equation = Equation('linear_elastic', {'E': 200, 'nu': 0.4})
    topopt = TopologyOptimizer(mesh, equation, bc, iters=10, volume_frac=0.5)
    solution = topopt.solve(plot=True)
    
    # Plotter(topopt._get_deformed_mesh(5), options={'title': 'TopoOpt iter 5'}).plot_values(solution.get_values('rhos', iter_idx=5))
    options = {'title': 'Topology Optimization', 'cbar_lim': [0, 1], 'cbar_label': 'Density', 'save': 'results/topopt.gif'}
    topopt.plot('rho_list', deformed=False, options=options) # animation

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    Plotter(topopt._get_deformed_mesh(), fig=fig, ax=ax[0], options={'title': 'Final Density', 'show': False}).plot_values(solution.get_values('rho_list', iter_idx=-1))
    Plotter(topopt._get_deformed_mesh(), fig=fig, ax=ax[1], options={'title': 'Final Stress'}).plot_values(solution.get_values('stress_list', iter_idx=-1))

def test_adaptive_refinement():
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    def test_function(point):
        # return [1]
        a = 50
        x, y = point - np.array([w/2, h/2])
        r2 = x**2 + y**2
        return [4*a*a*(1-a*r2)*e**(-a*r2)] # TODO: list thing is awkward

    equation = Equation('poisson')
    bc = BoundaryConditions(mesh)
    bc.add_force(test_function)
    bc.add("dirichlet", mesh.boundary_idxs, [0])
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    Plotter(mesh, options={'title': 'Poisson Solution'}).plot_values(solution.get_values('u'), mode='surface')
    solution.calculate_gradient('u')
    Plotter(mesh, options={'title': 'gradient'}).plot_values(solution.get_values('grad_u'), mode='arrows')

    raise NotImplementedError('Adaptive refinement demo is not implemented') # TODO

    # solution_init = solver.solve()
    # solver.adaptive_refinement()
    # solution_final = solver.solve()
    # u_init = solution_init.get_values('u')
    # u_final = solution_final.get_values('u')
    # r_init = solution_init.get_values('residuals')
    # r_final = solution_final.get_values('residuals')

    # fig = plt.figure(figsize=(10, 5))
    # axs = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122)]
    # Plotter(mesh, options={'title': 'Initial Solution', 'show': False}).plot_values(u_init, mode='surface')
    # Plotter(mesh, options={'title': 'Final Solution', 'show': False}).plot_values(u_final, mode='surface')
    # plt.show()

    # fig, ax = plt.subplots(2, 2)
    # Plotter(mesh, options={'title': 'Initial Residuals', 'show': False}).plot_values(r_init, mode='colored')
    # Plotter(mesh, options={'title': 'Final Residuals', 'show': False}).plot_values(r_final, mode='colored')
    # Plotter(mesh, options={'title': 'Initial Mesh', 'show': False}).plot_mesh(mode='wireframe')
    # Plotter(mesh, options={'title': 'Final Mesh', 'show': False}).plot_mesh(mode='wireframe')
    # plt.show()

def test_energy_solver(): # TODO: add support for force bc
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    equation = Equation('linear_elastic', {'E': 200, 'nu': 0.4})
    bc = BoundaryConditions(mesh)
    left_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] < 1e-6]
    right_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] > w-1e-6]
    bc.add('dirichlet', left_idxs, [0, 0])
    bc.add('dirichlet', right_idxs, [0.001, 0])
    # bc.plot()

    energy_solver = EnergySolver(mesh, equation, bc)
    solution = energy_solver.solve()
    vertices = mesh.vertices + solution.get_values('u').reshape(-1, 2)
    deformed_mesh = Mesh(vertices, mesh.elements, mesh.boundary)
    deformed_mesh.plot()

    u1 = solution.get_values('u')
    energy1 = energy_solver.energy(u1)

    solver2 = Solver(mesh, equation, bc)
    solution2 = solver2.solve()
    vertices = mesh.vertices + solution2.get_values('u').reshape(-1, 2)
    deformed_mesh2 = Mesh(vertices, mesh.elements, mesh.boundary)
    deformed_mesh2.plot()

    u2 = solution2.get_values('u')
    energy2 = energy_solver.energy(u2)

    # e1_energy_density = energy_solver.element_energy(0, u2.reshape(-1, 2)[mesh.elements[0]])
    # e1_energy = energy_solver.mesh.areas[0] * e1_energy_density
    # s1_energy = solution2.get_values('compliance')[0]

    # e1_strain = energy_solver.energy_density.S
    # e_idx = 0
    # u_element = u2.reshape(-1, 2)[mesh.elements[e_idx]]

    print(energy1, energy2)

if __name__ == "__main__":
    MESH_FILE = 'meshes/10x10.pkl'
    mesh = Mesh.load(MESH_FILE)

    # test_plot_mesh()
    # test_l2_projection()
    # test_poisson_equation()
    # test_heat_equation()
    # test_wave_equation() # TODO: running test_wave after test_heat seems to have plotting issues
    # test_linear_elastic()
    # test_topology_optimization()
    # test_energy_solver()
    # test_adaptive_refinement()
