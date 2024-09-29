import numpy as np
from math import sin, cos, pi, e
import matplotlib.pyplot as plt

from utils.helper import *

from BoundaryConditions import *
from FEMesh import *
from Plotter import *
from Solver import *
from TopologyOptimizer import *
from EnergySolver import *

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

def test_plot_mesh():
    plotter = Plotter(title='Mesh Plot')
    plotter.plot(mesh, mode='mesh')
    plotter.plot(mesh, mode='boundary')
    plotter.plot_highlights(mesh, [mesh.boundary_idxs], ['red'], ['boundary'])
    plotter.show()

def test_l2_projection():
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return [np.sin(40*(x**2+y**2))]
    equation = Equation('projection')
    bc = BoundaryConditions(mesh)
    bc.add_force(cool_f)
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()

    plotter = Plotter(title='L2 Projection')
    plotter.plot(mesh, solution.get_values('u'), mode='surface')
    plotter.show()

def test_poisson_equation():
    w = np.max(mesh.vertices[:, 0])
    right_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] > w-1e-6]

    equation = Equation('poisson')
    bc = BoundaryConditions(mesh)
    bc.add('dirichlet', [idx for idx in mesh.boundary_idxs if idx not in right_idxs], [0])
    bc.add('neumann', right_idxs, [1])
    bc.add_force(lambda point: [1])

    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    gradient = mesh.calculate_gradient(solution.get_values('u'))

    plotter = Plotter(1, 3, title='Poisson Equation')
    plotter.plot(mesh, solution.get_values('u'), mode='surface', title='Solution', idx=(0, 0))
    plotter.plot(mesh, gradient, mode='arrows', title='Gradient', idx=(0, 1))
    plotter.plot(mesh, np.linalg.norm(gradient, axis=1), mode='surface', title='Gradient Norm', idx=(0, 2))
    plotter.show()

def test_heat_equation():
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.5*min(w, h)) + 300
    
    equation = Equation('heat', {'u_initial': u_initial.copy(), 'iters': 5, 'dt': 0.01})
    solver = Solver(mesh, equation)
    solution = solver.solve()
    u_values = solution.get_values('u_values')
    t_values = solution.get_values('t_values')

    plotter = Plotter(1, 2, title='Heat Equation')
    plotter.plot_animation(mesh, u_values, mode='colored', titles=[f'Color t={t}' for t in t_values], idx=(0, 0))
    plotter.plot_animation(mesh, u_values, mode='surface', titles=[f'Surface t={t}' for t in t_values], idx=(0, 1))
    plotter.show()

def test_wave_equation(): # TODO: Wave energy not fully implemented
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    wave_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, wave_center, size=0.3*min(w, h))
    dudt_initial = np.zeros(len(mesh.vertices))
    
    equation = Equation('wave', {'u_initial': u_initial, 'dudt_initial': dudt_initial, 'c': 1, 'dt': 0.04, 'iters': 10})
    solver = Solver(mesh, equation)
    solution = solver.solve()
    u_values = solution.get_values('u_values')
    t_values = solution.get_values('t_values')

    plotter = Plotter(1, 1, title='Wave Equation')
    plotter.plot_animation(mesh, u_values, mode='surface', titles=[f'Surface t={t}' for t in t_values], idx=(0, 0))
    plotter.show()

def test_linear_elastic():
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    bc = BoundaryConditions(mesh)
    left_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] < 1e-6]
    right_middle_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] > w-1e-6 and 0.2 < mesh.vertices[v_idx][1] < 0.8]
    bc.add('dirichlet', left_idxs, [0, 0])
    bc.add('neumann', right_middle_idxs, [50, 0]) # stress
    # bc.plot()

    equation = Equation('linear_elastic', {'E': 200, 'nu': 0.4})
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    deformed_mesh = solution.get_deformed_mesh()
    displacements = np.linalg.norm(solution.get_values('u').reshape(-1, 2), axis=1)

    plotter = Plotter(1, 2, title='Linear Elasticity')
    plotter.plot(deformed_mesh, solution.get_values('stress'), mode='colored', title='Stress', idx=(0, 0))
    plotter.plot(mesh, displacements, mode='colored', title='Displacement', idx=(0, 1))
    plotter.show()


def test_topology_optimization(iters=10):
    def down_force(point):
        return np.array([0, -0.5])

    bc = BoundaryConditions(mesh)
    left_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] < 1e-6]
    bc.add('dirichlet', left_idxs, [0, 0])
    bc.add_force(down_force)

    equation = Equation('linear_elastic', {'E': 200, 'nu': 0.4})
    topopt = TopologyOptimizer(mesh, equation, bc, iters=iters, volume_frac=0.5)
    solution = topopt.solve(plot=False)
    deformed_mesh = topopt._get_deformed_mesh()

    plotter = Plotter(title='Topology Optimization')
    plotter.plot_animation(mesh, solution.get_values('rho_list'), mode='colored') # TODO: have mesh deform during animation, title
    plotter.show()

    rho_final = solution.get_values('rho_list', iter_idx=-1)
    stress_final = solution.get_values('stress_list', iter_idx=-1)
    plotter = Plotter(1, 2, title='Topology Optimization')
    plotter.plot(deformed_mesh, rho_final, mode='colored', title='Final Density', idx=(0, 0))
    plotter.plot(deformed_mesh, stress_final, mode='colored', title='Final Stress', idx=(0, 1))
    plotter.show()

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
    u = solution.get_values('u')
    u_gradient = mesh.calculate_gradient(u)

    plotter = Plotter(1, 2, title='Adaptive Refinement')
    plotter.plot(mesh, u, mode='surface', title='Poisson Solution', idx=(0, 0))
    plotter.plot(mesh, u_gradient, mode='arrows', title='Gradient', idx=(0, 1))
    plotter.show()

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
    bc.add('dirichlet', right_idxs, [0.5, 0])

    energy_solver = EnergySolver(mesh, equation, bc)
    solution = energy_solver.solve()
    vertices = mesh.vertices + solution.get_values('u').reshape(-1, 2)
    mesh_final = FEMesh(vertices, mesh.elements, mesh.boundary)

    plotter = Plotter(title='Energy Solver')
    plotter.plot(mesh_final, mode='mesh', title='Final')
    plotter.show()

def test_boundary_conditions():
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    
    # Linear Elastic Example
    equation = Equation('linear_elastic', {'E': 1, 'nu': 0.4})
    bc = BoundaryConditions(mesh)
    left_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] < 1e-6]
    right_idxs = [v_idx for v_idx in mesh.boundary_idxs if mesh.vertices[v_idx][0] > w-1e-6]
    bc.add('dirichlet', left_idxs, [0, 0])
    bc.add('neumann', right_idxs, [[0.1, 0] for idx in right_idxs])

    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    deformed_mesh = solution.get_deformed_mesh()

    plotter = Plotter(1, 2, title='Linear Elastic')
    plotter.plot(mesh, bc=bc, mode='bc', title='Boundary Conditions')
    plotter.plot(deformed_mesh, solution.get_values('stress'), mode='colored', title='Solution', idx=(0, 1))
    plotter.show()

    # Poisson Example
    equation = Equation('poisson')
    bc = BoundaryConditions(mesh)
    bc.add('dirichlet', [idx for idx in mesh.boundary_idxs if idx not in right_idxs], [0])
    bc.add('neumann', right_idxs, [1])
    bc.add_force(lambda point: [1])

    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    u = solution.get_values('u')
    u_gradient = mesh.calculate_gradient(u)
    u_gradient_norm = np.linalg.norm(u_gradient, axis=1)

    plotter = Plotter(1, 2, title='Boundary Conditions')
    plotter.plot(mesh, u, mode='surface', title='Poisson Solution', idx=(0, 0))
    plotter.plot(mesh, u_gradient_norm, mode='surface', title='Gradient Norm', idx=(0, 1))
    plotter.show()

if __name__ == "__main__":
    MESH_FILE = 'files/mesh_80x40.json'
    mesh = FEMesh.load(MESH_FILE)

    test_plot_mesh()
    test_l2_projection()
    test_poisson_equation()
    test_heat_equation()
    test_wave_equation()
    test_linear_elastic()
    test_topology_optimization(iters=15) # TODO: save animation, smoothing
    test_energy_solver()
    test_boundary_conditions()
    # test_adaptive_refinement()
