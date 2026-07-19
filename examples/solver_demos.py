"""Solver demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
"""
import numpy as np
from math import e

from fem.numerics import bump_function
from fem.mesh.femesh import FEMesh
from fem.elements import LinearTriangleElement, LinearTetrahedralElement
from fem.boundary import BoundaryConditions, BCType
from fem.regions import everywhere, on_plane, in_box, intersect
from fem.plot.plotter import Plotter
from fem.plot.tet import create_rect_tetmesh, plot_tetmesh_animation
from fem.solver import Solver, Projection, Poisson, Heat, Wave, LinearElastic
from fem.topology import TopologyOptimizer
from fem.energy_solver import EnergySolver

from demo_registry import Demo

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

def demo_plot_mesh(mesh):
    """Plot the mesh and highlight its boundary vertices."""
    plotter = Plotter(title='Mesh Plot')
    plotter.plot(mesh, mode='mesh')
    plotter.plot(mesh, mode='boundary')
    plotter.plot_highlights(mesh, [mesh.boundary_idxs], ['red'], ['boundary'])
    return plotter

def demo_l2_projection(mesh):
    """L2-project an oscillatory function onto the mesh's finite element space."""
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return [np.sin(40*(x**2+y**2))]
    equation = Projection(source=cool_f)
    solver = Solver(mesh, equation)
    solution = solver.solve()

    plotter = Plotter(title='L2 Projection')
    plotter.plot(mesh, solution.get_values('u'), mode='surface')
    return plotter

def demo_poisson_equation(mesh):
    """Solve Poisson's equation with zero Dirichlet BCs and a constant force."""
    equation = Poisson(source=1)
    bc = BoundaryConditions()
    # bc.add(BCType.NEUMANN, on_plane(0, np.max(mesh.vertices[:, 0])), [1])
    bc.add(BCType.DIRICHLET, everywhere(), 0)

    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    gradient = mesh.calculate_gradient(solution.get_values('u'))

    plotter = Plotter(1, 3, title='Poisson Equation')
    plotter.plot(mesh, solution.get_values('u'), mode='surface', title='Solution', idx=(0, 0))
    plotter.plot(mesh, gradient, mode='arrows', title='Gradient', idx=(0, 1))
    plotter.plot(mesh, np.linalg.norm(gradient, axis=1), mode='surface', title='Gradient Norm', idx=(0, 2))
    return plotter

def demo_heat_equation(mesh):
    """Animate transient heat diffusion from a hot bump initial condition."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.5*min(w, h)) + 300

    equation = Heat(u_initial=u_initial.copy(), iters=5, dt=0.01)
    solver = Solver(mesh, equation)
    solution = solver.solve()
    u_values = solution.get_values('u_values')
    t_values = solution.get_values('t_values')

    plotter = Plotter(1, 2, title='Heat Equation')
    plotter.plot_animation(mesh, u_values, mode='colored', titles=[f'Color t={t}' for t in t_values], idx=(0, 0))
    plotter.plot_animation(mesh, u_values, mode='surface', titles=[f'Surface t={t}' for t in t_values], idx=(0, 1))
    return plotter

def demo_wave_equation(mesh):  # TODO: Wave energy not fully implemented
    """Animate wave propagation from a bump initial condition, then show late frames individually."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    wave_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, wave_center, size=0.25*min(w, h))
    dudt_initial = np.zeros(len(mesh.vertices))

    equation = Wave(u_initial=u_initial, dudt_initial=dudt_initial, c=1, dt=0.03, iters=20)
    solver = Solver(mesh, equation)
    solution = solver.solve()
    u_values = solution.get_values('u_values')
    t_values = solution.get_values('t_values')

    plotter = Plotter(1, 1, title='Wave Equation')
    plotter.plot_animation(mesh, u_values, mode='surface', titles=[f'Surface t={t}' for t in t_values], idx=(0, 0))
    plotters = [plotter]

    for i in range(6, len(u_values)):
        frame_plotter = Plotter(1, 1, title='Wave Equation')
        frame_plotter.plot(mesh, u_values[i], mode='surface', empty=True)
        plotters.append(frame_plotter)

    return plotters

def demo_linear_elastic(mesh):
    """Solve linear elasticity for a cantilever fixed on the left with a traction load."""
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN,  # stress, on the middle band of the right edge
           intersect(on_plane(0, w), in_box([None, 0.2], [None, 0.8])),
           [50, 0])
    bc.plot(mesh)

    equation = LinearElastic(E=200, nu=0.4)
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    deformed_mesh = solution.get_deformed_mesh()
    displacements = np.linalg.norm(solution.get_values('u').reshape(-1, 2), axis=1)

    plotter = Plotter(1, 2, title='Linear Elasticity')
    plotter.plot(deformed_mesh, solution.get_values('stress'), mode='colored', title='Stress', idx=(0, 0))
    plotter.plot(mesh, displacements, mode='colored', title='Displacement', idx=(0, 1))
    return plotter

def demo_topology_optimization(mesh, iters=10):
    """Run SIMP topology optimization on a cantilever under a downward force."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    equation = LinearElastic(E=200, nu=0.4, source=[0, -0.5])
    topopt = TopologyOptimizer(mesh, equation, bc, iters=iters, volume_frac=0.5)
    solution = topopt.solve(plot=False)
    deformed_mesh = topopt._get_deformed_mesh()

    animation_plotter = Plotter(title='Topology Optimization')
    animation_plotter.plot_animation(mesh, solution.get_values('rho_list'), mode='colored') # TODO: have mesh deform during animation, title

    rho_final = solution.get_values('rho_list', iter_idx=-1)
    stress_final = solution.get_values('stress_list', iter_idx=-1)
    final_plotter = Plotter(1, 2, title='Topology Optimization')
    final_plotter.plot(deformed_mesh, rho_final, mode='colored', title='Topology Optimized Structure', idx=(0, 0), empty=True)
    final_plotter.plot(deformed_mesh, stress_final, mode='colored', title='Final Stress', idx=(0, 1))
    return [animation_plotter, final_plotter]

def demo_adaptive_refinement(mesh):
    """Attempt adaptive refinement of a Poisson solve (currently blocked, see BACKLOG.md)."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    def test_function(point):
        # return [1]
        a = 50
        x, y = point - np.array([w/2, h/2])
        r2 = x**2 + y**2
        return [4*a*a*(1-a*r2)*e**(-a*r2)] # TODO: list thing is awkward

    equation = Poisson(source=test_function)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0)
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    u = solution.get_values('u')
    u_gradient = mesh.calculate_gradient(u)

    plotter = Plotter(1, 2, title='Adaptive Refinement')
    plotter.plot(mesh, u, mode='surface', title='Poisson Solution', idx=(0, 0))
    plotter.plot(mesh, u_gradient, mode='arrows', title='Gradient', idx=(0, 1))
    plotter.show()  # shown directly: this demo always raises below, so there's no return to show it via

    # solver.adaptive_refinement now drives the loop correctly, but this demo is
    # still blocked on two open pieces: a real a-posteriori error estimator to
    # pass in, and position-based Dirichlet conditions (the ones added above are
    # index-based, so they cannot survive the vertex renumbering a refinement
    # does). See BACKLOG.md.
    raise NotImplementedError(
        'Adaptive refinement demo needs an error estimator and remeshable Dirichlet BCs'
    )

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

def demo_energy_solver(mesh):  # displacement-driven: EnergySolver rejects a source term
    """Minimize elastic energy directly (Newton solve) instead of the linear FEM system."""
    w = np.max(mesh.vertices[:, 0])
    equation = LinearElastic(E=200, nu=0.4)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, w), [0.5, 0])

    energy_solver = EnergySolver(mesh, equation, bc)
    solution = energy_solver.solve()
    vertices = mesh.vertices + solution.get_values('u').reshape(-1, 2)
    mesh_final = FEMesh(vertices, mesh.elements, mesh.boundary, element_type=LinearTriangleElement)
    solution.get_values('energy')
    stresses = np.linalg.norm(solution.get_values('gradient').reshape(-1, 2), axis=1)

    plotter = Plotter(title='Energy Solver')
    plotter.plot(mesh_final, stresses, mode='colored', title='Final')
    return plotter

def demo_3d():
    """Solve transient heat diffusion on a 3D tetrahedral mesh (renders via PyVista)."""
    mesh = create_rect_tetmesh(x_lim=[0, 4], y_lim=[0, 1], z_lim=[0, 1], subdividisions=2, plot=False)
    mesh = FEMesh(mesh.vertices, mesh.elements, mesh.boundary, element_type=LinearTetrahedralElement)

    w = max(mesh.vertices.flatten()) - min(mesh.vertices.flatten())
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.3*w) + 300

    equation = Heat(u_initial=u_initial.copy(), iters=20, dt=0.04)
    solver = Solver(mesh, equation)
    solution = solver.solve()
    u_values = solution.get_values('u_values')
    solution.get_values('t_values')

    plot_tetmesh_animation(mesh, np.array(u_values), title='Heat Diffusion')


DEMOS = [
    Demo('plot_mesh', demo_plot_mesh),
    Demo('l2_projection', demo_l2_projection),
    Demo('poisson', demo_poisson_equation),
    Demo('heat', demo_heat_equation),
    Demo('wave', demo_wave_equation),
    Demo('linear_elastic', demo_linear_elastic),
    Demo('topology_optimization', demo_topology_optimization),
    Demo('adaptive_refinement', demo_adaptive_refinement, returns_plotter=False),
    Demo('energy_solver', demo_energy_solver),
    Demo('3d', demo_3d, needs_mesh=False, returns_plotter=False),
]
