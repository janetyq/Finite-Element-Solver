"""Solver demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
"""
import numpy as np
from functools import partial
from math import e
from pathlib import Path

from fem.numerics import bump_function
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import ConvergenceStudy, poisson_convergence
from fem.regions import everywhere, on_plane, in_box, intersect
from fem.plot.plotter import Plotter
from fem.equations import Projection, Poisson, LinearElastic, StrainMeasure
from fem.solver import Solver
from fem.problem import heat, wave
from fem.integrators import NewmarkMethod, ThetaMethod
from fem.topology import TopologyOptimizer
from fem.energy_solver import EnergySolver

from demo_registry import Demo, DemoResult, Figure
from domains import beam, square

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

def demo_plot_mesh(mesh):
    """Plot the mesh and highlight its boundary vertices."""
    plotter = Plotter(title='Mesh Plot', axis_labels=False)
    plotter.plot(mesh, mode='mesh')
    plotter.plot(mesh, mode='boundary')
    plotter.plot_highlights(mesh, [mesh.boundary_idxs], ['red'], ['boundary'])
    return DemoResult([Figure(plotter, 'Mesh edges with the boundary vertices marked.')])

def demo_l2_projection(mesh):
    """L2-project an oscillatory function onto the mesh's finite element space."""
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return [np.sin(40*(x**2+y**2))]
    equation = Projection(source=cool_f)
    solver = Solver(mesh, equation)
    solution = solver.solve()

    plotter = Plotter(title='L2 Projection')
    plotter.plot(mesh, solution.u, mode='surface')
    return DemoResult([Figure(
        plotter,
        'sin(40 r^2) projected onto the P1 space: the mesh resolves the inner rings '
        'and loses the outer ones.')])

def demo_poisson_equation(mesh):
    """Solve Poisson's equation with zero Dirichlet BCs and a constant force."""
    equation = Poisson(source=1)
    bc = BoundaryConditions()
    # bc.add(BCType.NEUMANN, on_plane(0, np.max(mesh.vertices[:, 0])), [1])
    bc.add(BCType.DIRICHLET, everywhere(), 0)

    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    gradient = solver.space.gradient(solution.u)

    plotter = Plotter(1, 3, title='Poisson Equation')
    plotter.plot(mesh, solution.u, mode='surface', title='Solution', idx=(0, 0))
    plotter.plot(mesh, gradient, mode='arrows', title='Gradient', idx=(0, 1))
    plotter.plot(mesh, np.linalg.norm(gradient, axis=1), mode='surface', title='Gradient Norm', idx=(0, 2))
    return DemoResult([Figure(
        plotter,
        'A constant unit source pinned at every boundary node, with the gradient '
        'recovered from the solution beside it.')])

def demo_convergence(resolutions=(11, 21, 41, 81)):
    """Measure the solver's own error against an exactly known solution, and read off
    the convergence rate (Method of Manufactured Solutions)."""
    # The one demo that does not show what the solver computed, but how wrong it was.
    # Every other figure here is checked by eye; this is the claim that survives being
    # looked at properly -- and it is the claim P1 elements make: halve h, quarter the
    # error. The same study runs as an assertion in tests/test_convergence.py.
    solves = poisson_convergence(resolutions)
    study = ConvergenceStudy.from_solves(solves)
    finest = solves[-1]

    plotter = Plotter(1, 2, title='Convergence against a manufactured solution')
    plotter.plot(finest.mesh, finest.pointwise_error, mode='colored', idx=(0, 0),
                 label='u_h - u_exact', title=f'Error field at h={finest.h:.3g}')

    ax = plotter.chart_ax(idx=(0, 1), xlabel='h', ylabel='L2 error')
    ax.loglog(study.h, study.error, 'o-', color='tab:blue',
              label=f'measured (order {study.fitted_order:.2f})')
    # Anchored at the coarsest point, so the two lines start together and any
    # divergence downward is the measured rate beating h^2 rather than an offset.
    ax.loglog(study.h, study.error[0] * (study.h / study.h[0])**2, '--', color='gray',
              label='h^2')
    ax.set_title('Error vs mesh size')
    ax.grid(True, which='both', alpha=0.3)
    # The mesh sizes themselves, rather than the decade ticks a log axis defaults to:
    # the sequence spans well under one decade, so the minor labels ran together.
    ax.set_xticks(study.h, [f'{h:g}' for h in study.h])
    ax.set_xticks([], minor=True)

    rows = ['     h      L2 error   order', *(
        f'{h:8.4f}  {e:10.3e}  {"-" if i == 0 else f"{study.orders[i-1]:6.2f}"}'
        for i, (h, e) in enumerate(zip(study.h, study.error))
    )]
    return DemoResult(
        [Figure(plotter,
                f'The error is smooth and one-signed: zero on the boundary, where the '
                f'solution is pinned exactly, and deepest at the centre, where the '
                f'solution itself peaks and the piecewise-linear space has the most to '
                f'miss. Halving h divides it by about four -- a fitted order of '
                f'{study.fitted_order:.2f} against the 2 that P1 elements promise.')],
        text='\n'.join(rows),
    )

def demo_robin_bc(mesh):
    """Cool a heated plate through a convective boundary, sweeping the Robin coefficient."""
    # du/dn + kappa*(u - u_ambient) = 0: heat generated inside escapes through a boundary
    # film, and kappa says how freely. The other two condition types are its limits --
    # kappa -> 0 is insulated (Neumann) and kappa -> infinity pins u to ambient
    # (Dirichlet) -- so the sweep ends on a Dirichlet solve the last Robin panel should
    # already look like.
    u_ambient = 300.0
    equation = Poisson(source=50.0)
    kappas = [0.5, 5.0, 500.0]

    plotter = Plotter(1, len(kappas) + 1, title='Robin BCs: convective cooling')
    for i, kappa in enumerate(kappas):
        bc = BoundaryConditions()
        bc.add_robin(everywhere(), kappa=kappa, g=kappa*u_ambient)
        u = Solver(mesh, equation, bc).solve().u
        plotter.plot(mesh, u, mode='colored', idx=(0, i), label='temperature',
                     title=f'kappa={kappa:g}\n{u.min():.1f} - {u.max():.1f}')

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), u_ambient)
    u = Solver(mesh, equation, bc).solve().u
    plotter.plot(mesh, u, mode='colored', idx=(0, len(kappas)), label='temperature',
                 title=f'Dirichlet limit\n{u.min():.1f} - {u.max():.1f}')
    return DemoResult([Figure(
        plotter,
        'Convective cooling at three film coefficients. The last Robin panel and the '
        'Dirichlet solve beside it agree to the digit -- the limit, computed both ways.')])

def demo_nonlinear_elastic(mesh, stretch=0.5):
    """Stretch a block hard, comparing small-strain elasticity against St Venant-Kirchhoff."""
    # Same material, same imposed displacement, different strain measure: the linear
    # path uses eps = (grad u + grad u^T)/2, which is only the leading term of the
    # Green-Lagrange strain S = (F^T F - I)/2 that St Venant-Kirchhoff uses. Under a
    # uniaxial stretch lambda the two read (lambda - 1) and (lambda^2 - 1)/2, so the
    # finite-strain model stiffens as the stretch grows and the linear one cannot.
    # Both solutions report Cauchy stress (see LinearElasticForm/EnergyForm
    # .derived_fields), so the two von Mises fields are the same measure and the gap
    # between them is physics rather than bookkeeping. The peak sits at the clamped
    # corners, where the imposed displacement is singular, so the median is quoted
    # beside it as the bulk figure.
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, w), [stretch*w, 0])

    small = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve()
    finite = EnergySolver(
        mesh, LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE), bc
    ).solve()

    plotter = Plotter(1, 2, title=f'Small strain vs St Venant-Kirchhoff ({stretch:.0%} stretch)')
    for i, (name, solution) in enumerate([('Small strain', small), ('Green-Lagrange', finite)]):
        vm = solution.von_mises
        plotter.plot(solution.deformed_mesh(), vm, mode='colored', idx=(0, i),
                     label='von Mises stress',
                     title=f'{name}\nvon Mises: median {np.median(vm):.0f}, peak {vm.max():.0f}')
    return DemoResult([Figure(
        plotter,
        'The same 50% stretch under both strain measures. Green-Lagrange stiffens as '
        'the stretch grows; small strain cannot.')])

def demo_stress_invariants(mesh):
    """Show the four rotation-invariant stress measures recovered from one elastic solve."""
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, intersect(on_plane(0, w), in_box([None, 0.2], [None, 0.8])), [0, -0.5])

    solution = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve()
    deformed = solution.deformed_mesh()

    # Each is a different question asked of the same stress tensor: distortion, mean
    # normal stress, the Tresca measure, and the largest tensile principal value.
    fields = [
        ('Von Mises', solution.von_mises),
        ('Pressure', solution.pressure),
        ('Max shear', solution.max_shear),
        ('Max principal', solution.principal_stress[:, -1]),
    ]
    plotter = Plotter(2, 2, title='Stress invariants of one solve', panel_aspect=4.0)
    for i, (name, values) in enumerate(fields):
        plotter.plot(deformed, values, mode='colored', idx=divmod(i, 2), title=name)
    return DemoResult([Figure(
        plotter,
        'Four rotation-invariant reductions of one stress tensor: distortion, mean '
        'normal stress, the Tresca measure, and the largest tensile principal value.')])

def demo_heat_equation(mesh):
    """Animate transient heat diffusion from a hot bump initial condition."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.5*min(w, h)) + 300

    # dt sized to the bump's decay, not to a round number: the corner bump loses 99% of
    # its contrast by t=0.4, so a run that long is three quarters flat square. Over
    # t=0.08 the same 40 frames spread the decay out and still reach near-uniform.
    solution = ThetaMethod(dt=0.002, steps=40).run(heat(mesh), u_initial.copy())
    u_values = solution.u
    t_values = solution.t

    animation = Plotter(1, 2, title='Heat Equation')
    animation.plot_animation(mesh, u_values, mode='colored', label='temperature',
                             titles=[f'Color t={t:.3f}' for t in t_values], idx=(0, 0))
    animation.plot_animation(mesh, u_values, mode='surface',
                             titles=[f'Surface t={t:.3f}' for t in t_values], idx=(0, 1))

    # The animation renders only on show(), so the diffusion needs a still form too --
    # otherwise this demo contributes nothing to a saved gallery.
    snapshots = Plotter(2, 3, title='Heat Equation: diffusion from the corner')
    for panel, i in enumerate(np.linspace(0, len(u_values) - 1, 6).astype(int)):
        snapshots.plot(mesh, u_values[i], mode='colored', idx=divmod(panel, 3),
                       label='temperature', title=f't={t_values[i]:.3f}')

    return DemoResult([
        Figure(animation, 'Crank-Nicolson diffusion, coloured and as a surface.', 'animation'),
        Figure(snapshots,
               'The same run sampled at six times: the corner bump spreads and the '
               'plate approaches a uniform temperature.', 'snapshots'),
    ])

def demo_wave_equation(mesh):  # TODO: Wave energy not fully implemented
    """Animate wave propagation from a bump initial condition, plus a grid of late snapshots."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    wave_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, wave_center, size=0.25*min(w, h))
    dudt_initial = np.zeros(len(mesh.vertices))

    solution = NewmarkMethod(dt=0.03, steps=40).run(wave(mesh, c=1), u_initial, dudt_initial)
    u_values = solution.u
    t_values = solution.t

    animation = Plotter(1, 1, title='Wave Equation')
    animation.plot_animation(mesh, u_values, mode='surface',
                             titles=[f'Surface t={t:.2f}' for t in t_values], idx=(0, 0))

    # Snapshots from the second half of the run, once the pulse has reflected off the
    # boundary and started interfering with itself. One grid, rather than the window
    # per frame this used to open.
    snapshots = Plotter(2, 3, title='Wave Equation: reflection and interference')
    for panel, i in enumerate(np.linspace(len(u_values)//2, len(u_values) - 1, 6).astype(int)):
        snapshots.plot(mesh, u_values[i], mode='surface', idx=divmod(panel, 3),
                       title=f't={t_values[i]:.2f}')

    return DemoResult([
        Figure(animation, 'Newmark time integration of a pulse on a fixed membrane.',
               'animation'),
        Figure(snapshots,
               'Six times from the second half of the run, after the pulse has reflected '
               'off the boundary and begun interfering with itself.', 'snapshots'),
    ])

def demo_linear_elastic(mesh):
    """Solve linear elasticity for a cantilever fixed on the left with a traction load."""
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    # Transverse, so the beam bends: an axial pull is much the same solve on any
    # domain, where a tip load is what makes a cantilever one. Sized for a tip
    # deflection near 9% of the span -- a 4:1 beam is compliant enough that the load
    # this demo used to apply axially would bend it through more than its own length,
    # well outside the small-strain regime the solver assumes.
    bc.add(BCType.NEUMANN,
           intersect(on_plane(0, w), in_box([None, 0.2], [None, 0.8])),
           [0, -0.5])

    equation = LinearElastic(E=200, nu=0.4)
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    deformed_mesh = solution.deformed_mesh()
    displacements = np.linalg.norm(solution.u.reshape(-1, 2), axis=1)

    plotter = Plotter(1, 3, title='Linear Elasticity', panel_aspect=4.0)
    plotter.plot(mesh, mode='bc', bc=bc, title='Boundary conditions', idx=(0, 0))
    plotter.plot(deformed_mesh, solution.von_mises, mode='colored', title='Von Mises stress',
                 label='von Mises stress', idx=(0, 1))
    plotter.plot(mesh, displacements, mode='colored', title='Displacement',
                 label='|u|', idx=(0, 2))
    return DemoResult([Figure(
        plotter,
        'A cantilever clamped at the left and loaded downwards at the tip. The bending '
        'stress is largest at the clamp and splits top from bottom -- tension over the '
        'neutral axis, compression under it.')])

def demo_topology_optimization(mesh, iters=40):
    """Run SIMP topology optimization on a cantilever under a downward force."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    equation = LinearElastic(E=200, nu=0.4, source=[0, -0.5])
    topopt = TopologyOptimizer(mesh, equation, bc, iters=iters, volume_frac=0.5)
    history = topopt.solve()
    deformed_mesh = topopt.deformed_mesh()

    animation_plotter = Plotter(title='Topology Optimization', panel_aspect=2.0)
    animation_plotter.plot_animation(mesh, history.rho, mode='colored', label='density') # TODO: have mesh deform during animation, title

    rho_final = history.rho[-1]
    stress_final = history.von_mises[-1]
    final_plotter = Plotter(1, 2, title='Topology Optimization', panel_aspect=2.0)
    final_plotter.plot(deformed_mesh, rho_final, mode='colored', title='Topology Optimized Structure',
                       label='density', idx=(0, 0), empty=True)
    final_plotter.plot(deformed_mesh, stress_final, mode='colored', title='Final von Mises stress',
                       label='von Mises stress', idx=(0, 1))
    return DemoResult([
        Figure(animation_plotter, 'Density evolving over the SIMP iterations.', 'animation'),
        Figure(final_plotter,
               'The converged structure and its stress: material has migrated into a '
               'truss carrying the load back to the supported edge.', 'final'),
    ])

def demo_adaptive_refinement(mesh):
    """Solve the peaked Poisson problem adaptive refinement is meant for (refinement itself
    is still blocked, see BACKLOG.md)."""
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
    u = solution.u
    u_gradient = solver.space.gradient(u)

    plotter = Plotter(1, 2, title='Adaptive Refinement: the problem, not yet the refinement')
    plotter.plot(mesh, u, mode='surface', title='Poisson Solution', idx=(0, 0))
    plotter.plot(mesh, u_gradient, mode='arrows', title='Gradient', idx=(0, 1))

    # AdaptiveRefinement(solver, estimator).run() drives the loop correctly, but calling
    # it here is still blocked on two open pieces: a real a-posteriori error estimator to
    # pass in, and position-based Dirichlet conditions (the ones added above are
    # index-based, so they cannot survive the vertex renumbering a refinement does).
    # Until then this shows the problem that motivates refinement -- a source with one
    # sharp interior peak, which a uniform mesh spends most of its elements not
    # resolving -- rather than raising and showing nothing. See BACKLOG.md.
    return DemoResult([Figure(
        plotter,
        'A Poisson source with one sharp interior peak: the case for refining where the '
        'error is. The refinement loop itself is not run -- it needs an error estimator.')])

    # from fem.adaptivity import AdaptiveRefinement
    # solution_final = AdaptiveRefinement(solver, estimator).run()
    # u_init = solution_init.u
    # u_final = solution_final.u
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

    # EnergySolver returns the same ElasticSolution the linear path does, so the
    # recovered stress is read the same way -- the parity is the point of the demo.
    plotter = Plotter(title=f'Energy Solver (minimised energy {energy_solver.energy(solution.u):.4g})')
    plotter.plot(solution.deformed_mesh(), solution.von_mises, mode='colored',
                 title='Von Mises stress', label='von Mises stress')
    return DemoResult([Figure(
        plotter,
        'A displacement-driven stretch solved by minimising energy rather than by '
        'assembling a linear system; the recovered stress reads the same way.')])

def demo_3d(steps=20, save_file='tetmesh_animation.gif'):
    """Solve transient heat diffusion on a 3D tetrahedral mesh (renders via PyVista)."""
    # `fem.plot.tet` needs the optional viz3d extra, so it is imported where it runs. A
    # module-level import takes down every demo in this file, and cli.py with them, on
    # the default install CI uses -- see tests/test_examples_import.py.
    from fem.plot.tet import create_rect_tetmesh, plot_tetmesh_animation

    mesh = create_rect_tetmesh(x_lim=[0, 4], y_lim=[0, 1], z_lim=[0, 1], subdividisions=2, plot=False)

    w = max(mesh.vertices.flatten()) - min(mesh.vertices.flatten())
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.3*w) + 300

    solution = ThetaMethod(dt=0.04, steps=steps).run(heat(mesh), u_initial.copy())

    # PyVista writes the frames as a GIF; the path is returned rather than left for the
    # caller to guess, which is what makes this demo collectable like any other.
    plot_tetmesh_animation(mesh, np.array(solution.u), save_file=save_file, title='Heat Diffusion')
    return DemoResult(artifacts=[Path(save_file)])


DEMOS = [
    # Coarse enough that individual edges and boundary vertices are legible.
    Demo('plot_mesh', demo_plot_mesh, domain=partial(square, 20)),
    # The point is which oscillations of sin(40 r^2) the space can represent, so this
    # one is meshed finer than the rest: at 40 a side the inner rings alias too.
    Demo('l2_projection', demo_l2_projection, domain=partial(square, 70)),
    Demo('poisson', demo_poisson_equation, domain=square),
    # Builds its own sequence of meshes rather than taking a domain: the refinement
    # sequence *is* the demo. The smoke run keeps the two coarsest -- an order needs
    # two points, and the 81x81 solve is most of the cost.
    Demo('convergence', demo_convergence, smoke_kwargs={'resolutions': (11, 21)}),
    Demo('robin', demo_robin_bc, domain=square),
    Demo('heat', demo_heat_equation, domain=square),
    Demo('wave', demo_wave_equation, domain=square),
    # A cantilever is a beam. On the square this used to load, the "bending" was a
    # square bulging sideways, and the stress concentration had nowhere to run to.
    Demo('linear_elastic', demo_linear_elastic, domain=partial(beam, 4.0, 1.0, 80)),
    Demo('stress_invariants', demo_stress_invariants, domain=partial(beam, 4.0, 1.0, 80)),
    # Stretched end to end, so the domain is incidental; a square keeps the deformed
    # and undeformed shapes comparable at a glance.
    Demo('nonlinear_elastic', demo_nonlinear_elastic, domain=square),
    # 2:1 at ~1600 vertices: the aspect ratio is what makes SIMP produce the truss it
    # is known for, and the vertex count is what keeps 40 iterations affordable in the
    # gallery workflow.
    Demo('topology_optimization', demo_topology_optimization, domain=partial(beam, 2.0, 1.0, 56)),
    Demo('adaptive_refinement', demo_adaptive_refinement, domain=square),
    Demo('energy_solver', demo_energy_solver, domain=square),
    # 20 steps of tet rendering is ~4.4s against ~1.9s for 3; the frames are identical
    # work, so the test takes the short run.
    Demo('3d', demo_3d, smoke_requires='pyvista', smoke_kwargs={'steps': 3}),
]
