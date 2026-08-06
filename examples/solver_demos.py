"""Solver demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
"""
import numpy as np
from functools import partial

from matplotlib.collections import LineCollection

from fem.adaptivity import AdaptiveRefinement
from fem.backends import IterativeBackend
from fem.geometry import calculate_triangle_min_angle
from fem.numerics import bump_function
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import (
    ConvergenceStudy, elastic_convergence, poisson_convergence, theta_convergence,
)
from fem.regions import everywhere, on_plane, in_box, intersect
from fem.plot.plotter import Plotter
from fem.equations import Projection, Poisson, LinearElastic, StrainMeasure
from fem.solver import Solver
from fem.mesh.ruppert import RuppertsAlgorithm, create_box_mesh
from fem.problem import heat, wave
from fem.integrators import NewmarkMethod, ThetaMethod
from fem.topology import TopologyOptimizer
from fem.energy_solver import EnergySolver

from demo_registry import Demo, DemoResult, Figure
from domains import beam, plate_with_hole_pslg, square

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

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

    conditions = Plotter()
    conditions.plot(mesh, mode='bc', bc=bc)

    plotter = Plotter(1, 3, title='Poisson Equation')
    plotter.plot(mesh, solution.u, mode='surface', title='Solution', idx=(0, 0))
    plotter.plot(mesh, gradient, mode='arrows', title='Gradient', idx=(0, 1))
    plotter.plot(mesh, np.linalg.norm(gradient, axis=1), mode='surface', title='Gradient Norm', idx=(0, 2))
    return DemoResult([
        Figure(plotter,
               'A constant unit source, with the gradient recovered from the solution '
               'beside it.',
               'fields'),
        Figure(conditions,
               'Pinned at every boundary node, so no part of this boundary is left to '
               'the natural condition -- which is what makes the solution vanish all '
               'the way round.',
               'conditions', setup=True),
    ])

def _plot_study(ax, study, label, colour, reference_order, xlabel):
    """One measured curve plus the power law it is being held to."""
    ax.loglog(study.step, study.error, 'o-', color=colour,
              label=f'{label} (order {study.fitted_order:.2f})')
    # Anchored at the coarsest point, so the two lines start together and any gap is
    # the measured rate differing from the reference rather than an offset between them.
    reference = study.error[0] * (study.step / study.step[0])**reference_order
    ax.loglog(study.step, reference, '--', color=colour, alpha=0.4,
              label=f'{xlabel}^{reference_order}')


def _tidy_log_axis(ax, steps):
    """Label the axis with the steps actually used.

    These sequences span well under a decade, where a log axis falls back to minor
    ticks like 2x10^-2 -- which run into each other.
    """
    ax.grid(True, which='both', alpha=0.3)
    # Plain decimals below a thousandth run to more digits than they are worth.
    fmt = '{:.1e}' if min(steps) < 1e-3 else '{:g}'
    ax.set_xticks(steps, [fmt.format(s) for s in steps])
    ax.set_xticks([], minor=True)


def demo_convergence(resolutions=(11, 21, 41, 81), elastic_resolutions=(9, 17, 33),
                     step_counts=(16, 32, 64, 128)):
    """Measure the solver's own error against exactly known solutions, and read off the
    convergence rates in space and in time (Method of Manufactured Solutions)."""
    # The one demo that does not show what the solver computed, but how wrong it was.
    # Every other figure here is checked by eye; these are the claims that survive
    # being looked at properly, and they are the two the discretization makes:
    #
    #   in space  P1 elements are O(h^2) -- halve h, quarter the error -- for a scalar
    #             unknown and for a coupled vector one alike;
    #   in time   the theta method's order is theta's to choose: 1 at backward Euler,
    #             2 at Crank-Nicolson, which is the default.
    #
    # All of it runs as assertions in tests/test_convergence{,_elasticity,_heat}.py;
    # this draws the same studies rather than a second implementation of them.
    solves = poisson_convergence(resolutions)
    poisson_study = ConvergenceStudy.from_solves(solves)
    elastic_study = ConvergenceStudy.from_solves(elastic_convergence(elastic_resolutions))
    # Step counts chosen to sit in the asymptotic band: over coarser steps
    # Crank-Nicolson reads an order near 3, because lambda*dt is not yet small and
    # the leading error term is not yet the one that dominates.
    crank_nicolson = theta_convergence(0.5, step_counts)
    backward_euler = theta_convergence(1.0, step_counts)
    finest = solves[-1]

    plotter = Plotter(1, 3, title='Convergence against manufactured solutions')
    plotter.plot(finest.mesh, finest.pointwise_error, mode='colored', idx=(0, 0),
                 label='u_h - u_exact', title=f'Poisson error at h={finest.h:.3g}')

    space = plotter.chart_ax(idx=(0, 1), xlabel='h', ylabel='L2 error')
    _plot_study(space, poisson_study, 'Poisson', 'tab:blue', 2, 'h')
    _plot_study(space, elastic_study, 'Elasticity', 'tab:green', 2, 'h')
    space.set_title('Space: P1 is second order')
    _tidy_log_axis(space, poisson_study.step)

    time = plotter.chart_ax(idx=(0, 2), xlabel='dt', ylabel='L2 error')
    _plot_study(time, crank_nicolson, 'Crank-Nicolson', 'tab:blue', 2, 'dt')
    _plot_study(time, backward_euler, 'Backward Euler', 'tab:red', 1, 'dt')
    time.set_title('Time: the order is theta\'s to choose')
    _tidy_log_axis(time, crank_nicolson.step)

    rows = ['                      fitted order   expected']
    for name, study, expected in (('Poisson (h)', poisson_study, 2),
                                  ('Elasticity (h)', elastic_study, 2),
                                  ('Crank-Nicolson (dt)', crank_nicolson, 2),
                                  ('Backward Euler (dt)', backward_euler, 1)):
        rows.append(f'{name:<22}{study.fitted_order:>9.2f}{expected:>11}')
    return DemoResult(
        [Figure(plotter,
                'Left: the Poisson error is smooth and one-signed -- zero on the boundary '
                'where the solution is pinned exactly, deepest at the centre where the '
                'piecewise-linear space has the most to miss. Middle: halving h quarters '
                'that error, for a scalar unknown and for a coupled vector one alike. '
                'Right: the same measurement against the time step instead, where the '
                'order is not a property of the elements but a choice -- backward Euler '
                'buys first order, Crank-Nicolson second, for the same cost per step.')],
        text='\n'.join(rows),
    )

def demo_stress_concentration(traction=1.0, length=6.0, height=3.0, radius=0.3,
                              min_angle=25, max_area_fraction=0.0005):
    """Take a plate with a hole from an outline through meshing and boundary conditions
    to the stress concentration at its rim, measured against the textbook factor of 3."""
    # The one demo that runs the whole pipeline, and so the only one that builds its own
    # mesh instead of being handed one: what the outline was, what Ruppert's was asked
    # for, and where the conditions went are each part of what this is showing, and a
    # domain factory would put all three somewhere the gallery page does not print.
    #
    # A domain with a hole in it has no structured triangulation, so this is a generated
    # mesh going straight into the solver. Nothing downstream of the meshing knows that:
    # the conditions are written against coordinates rather than vertex numbers, so they
    # resolve against whatever triangulation arrives.
    pslg = plate_with_hole_pslg(length, height, radius)
    pslg.validate()
    # The angle bound constrains element *shape* and says nothing about size, so the
    # area cap is what sets resolution. Neither asks for refinement at the hole; the
    # mesh grades there anyway, because the polygonalised rim is built from short input
    # segments and Ruppert's honours them.
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area())
    mesh = rupperts.refine()

    # The rim takes no condition at all, and that is the point: a free surface is the
    # natural boundary condition of the weak form, so "traction-free" is what an edge
    # means when nothing is said about it. The conditions panel draws it, rather than
    # leaving a reader to notice an absence.
    #
    # Kirsch's factor of 3 is the *infinite*-plate limit, and this plate is finite, so
    # the measured peak sits above it -- the hole removes section, which raises the
    # stress the remaining material carries. It falls toward 3 as the hole shrinks
    # relative to the plate: around 3.3 at a hole 0.20 of the height, 3.27 at 0.15,
    # 3.22 at 0.12.
    #
    # Only one digit of that is worth quoting. The peak is read at element centroids,
    # and Ruppert's lays down a different triangulation at every size cap, so how close
    # the nearest centroid falls to the rim -- where the gradient is steepest -- varies
    # between runs. Refining does not settle it monotonically: 3.35, 3.56, 3.34, 3.34
    # over 1182, 2074, 3233 and 5272 elements. Reading the true rim value would mean
    # extrapolating to the boundary rather than sampling near it.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, length), [traction, 0])

    equation = LinearElastic(E=200, nu=0.3)
    solution = Solver(mesh, equation, bc).solve()
    sigma_xx = solution.stress[:, 0, 0]

    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    # A vertical strip through the hole's centre: the line the concentration decays
    # along, from the rim out to the far field. The geometry is known here rather than
    # measured back off the mesh, which is what building the domain in the demo buys.
    strip = np.abs(centroids[:, 0] - length/2) < 0.4*radius
    order = np.argsort(centroids[strip, 1])
    y_strip, ratio_strip = centroids[strip, 1][order], (sigma_xx[strip] / traction)[order]
    peak = ratio_strip.max()

    # Two figures rather than one: five panels of a 2:1 plate and a chart do not share a
    # sensible aspect ratio, and a grid that size thumbnails to nothing legible.
    built = Plotter(1, 2, title='From an outline to a solvable mesh', panel_aspect=2.0)
    built.plot(mesh, mode='mesh', idx=(0, 0),
               title=f'{len(mesh.elements)} triangles\n'
                     f'min_angle={min_angle} deg, max_area={max_area_fraction:.2%}')
    # The input segments over the triangulation, rather than the outline in a panel of
    # its own: four corners and a polygonalised circle is an almost empty picture alone,
    # and drawn here it shows which of them the mesher kept and which it split.
    built.get_ax((0, 0)).add_collection(LineCollection(
        rupperts.vertices[rupperts.segments], colors='blue', linewidths=1.0))
    built.plot(mesh, mode='bc', bc=bc, idx=(0, 1), title='Boundary conditions')

    plotter = Plotter(1, 2, title='Stress concentration around a hole', panel_aspect=3.0)
    plotter.plot(mesh, sigma_xx, mode='colored', idx=(0, 0), label='sigma_xx',
                 title=f'{len(mesh.elements)} generated triangles')
    ax = plotter.chart_ax(idx=(0, 1), xlabel='y', ylabel='sigma_xx / applied')
    # Drawn as two runs, below the hole and above it. One run joins them straight
    # across the gap, which reads as a stress the hole does not have.
    below = y_strip < height/2
    ax.plot(y_strip[below], ratio_strip[below], 'o-', color='tab:blue', markersize=3,
            label='through the hole centre')
    ax.plot(y_strip[~below], ratio_strip[~below], 'o-', color='tab:blue', markersize=3)
    ax.axhline(3.0, color='tab:red', linestyle='--', label='Kirsch: 3x at the rim')
    ax.axhline(1.0, color='gray', linestyle=':', label='far field')
    ax.set_title(f'Peak {peak:.1f}x the applied stress')
    ax.grid(alpha=0.3)
    # Below the curve: the peak is what this panel exists to show, and a default-placed
    # legend sat on top of it.
    ax.legend(loc='lower center', fontsize='small')

    # Adaptive refinement, driven by this same equation's residual estimator
    # (LinearElastic.error_estimate) rather than by the uniform area cap above.
    # Starts from a copy of the mesh already built, so the comparison is the same
    # physics at the same starting resolution.
    #
    # It does end up concentrated at the rim -- but not right away. A fully
    # clamped edge is its own source of error: it resists the plate's natural
    # sideways contraction as it stretches, and that resistance is itself poorly
    # resolved close to x=0. The first rounds go there and to the four corners,
    # where the clamp or the pulled edge meets a free one -- boundary conditions
    # changing type is a standard weak singularity, independent of the hole.
    # Only once those settle does the rim pull ahead: over the ten rounds run
    # here, roughly four-fifths of the new triangles land within one hole-radius
    # of it. `BACKLOG.md` has the fix for the clamp's share of that -- a roller
    # condition in place of the full clamp -- which today's boundary conditions
    # cannot express.
    adaptive_solver = Solver(mesh.copy(), equation, bc)
    adaptive_solution = AdaptiveRefinement(
        adaptive_solver, equation.error_estimate,
        max_triangles=len(mesh.elements) + 2000, max_iters=10,
    ).run()
    adaptive_mesh = adaptive_solver.mesh
    adaptive_sigma_xx = adaptive_solution.stress[:, 0, 0]
    adaptive_centroids = adaptive_mesh.vertices[adaptive_mesh.elements].mean(axis=1)
    adaptive_strip = np.abs(adaptive_centroids[:, 0] - length/2) < 0.4*radius
    adaptive_peak = float((adaptive_sigma_xx[adaptive_strip] / traction).max())

    adaptive_plotter = Plotter(1, 2, title='Adaptive refinement, driven by the residual estimator',
                               panel_aspect=2.0)
    adaptive_plotter.plot(adaptive_mesh, mode='mesh', idx=(0, 0),
                          title=f'{len(adaptive_mesh.elements)} triangles\n'
                                f'(started from {len(mesh.elements)})')
    adaptive_plotter.plot(adaptive_mesh, adaptive_sigma_xx, mode='colored', idx=(0, 1),
                          label='sigma_xx', title=f'peak {adaptive_peak:.1f}x the applied stress')

    worst_angle = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()
    rim_facets = int(np.sum(rupperts.boundary_loops == 1))
    return DemoResult(
        [Figure(built,
                'Left: the mesh, with the outline it was refined from in blue. Nothing '
                'asked for smaller elements at the hole -- the area cap is uniform -- '
                'but the rim is polygonalised into short segments, and honouring them '
                'is what grades the mesh there. Right: where the conditions went. The '
                'rim and the long edges carry none, which is not an omission but the '
                'natural condition of the weak form: an edge nothing is said about is '
                'traction-free. Every one of these is written against coordinates '
                'rather than vertex numbers, which is what lets them be placed on a '
                'mesh no one laid out by hand.',
                'built'),
         Figure(plotter,
                f'A plate pulled from the right, with the hole left traction-free. The '
                f'stress crowds into the material either side of the hole and relaxes '
                f'to the applied value within about a diameter, peaking at {peak:.1f}x '
                f'the applied stress. Kirsch gives 3x -- for a hole in an *infinite* '
                f'plate. This one is three hole-diameters tall, so the hole removes '
                f'enough section to push the peak above that limit, and the excess '
                f'shrinks as the hole does. A textbook constant is a limit, not a '
                f'target -- and one digit is all this measurement supports, since the '
                f'peak is sampled at element centroids near the steepest gradient in '
                f'the field.',
                'stress', thumbnail=True),
         Figure(adaptive_plotter,
                f'The same problem, refined by LinearElastic.error_estimate instead of a '
                f'uniform area cap: {len(mesh.elements)} triangles growing to '
                f'{len(adaptive_mesh.elements)} over ten rounds, peaking at '
                f'{adaptive_peak:.1f}x. Left: where the extra triangles went -- mostly the '
                f'rim, but not only it. Right: the stress they now resolve. The estimator '
                f'also spends triangles at the four corners and along the clamped edge, a '
                f'real, separate source of error here (see the code comment above) rather '
                f'than something specific to the hole.',
                'adaptive')],
        text=(f'outline points           {len(pslg.vertices)}  '
              f'(rectangle + polygonalised rim)\n'
              f'generated elements       {len(mesh.elements)}\n'
              f'worst angle              {worst_angle:.1f} deg   '
              f'(asked for {min_angle})\n'
              f'boundary edges           {len(mesh.boundary)}   '
              f'({rim_facets} of them the hole rim)\n'
              f'applied traction         {traction:.3g}\n'
              f'hole diameter / height   {2*radius/height:.2f}\n'
              f'peak sigma_xx / applied  {peak:.2f}   (Kirsch, infinite plate: 3)\n'
              f'adaptive elements        {len(adaptive_mesh.elements)}\n'
              f'adaptive peak sigma_xx / applied  {adaptive_peak:.2f}'),
    )

def demo_elastic_3d(n=17):
    """Bend a 3D cantilever beam of tetrahedra, drawn as its boundary surface."""
    # The package solves in 3D throughout -- the same assembly, the same element
    # hierarchy, `Solver` reading the element type off the connectivity. `heat_3d` draws
    # the same way, through `plot_solid` (`fem/plot/helpers.py`).
    mesh = create_box_mesh(corners=[[0, 0, 0], [4, 1, 1]], resolution=(4*n//2, n//2, n//2))

    # The two 3D demos are the only solves here with no conditions panel: `plot_bc`
    # draws boundary facets as line segments, and in 3D they are triangles. The clamp
    # and the tip load are the 2D cantilever's, which does show them.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 4.0), [0, 0, -0.5])

    # AMG-preconditioned CG rather than the direct factorization: a 3D elastic solve
    # is where fill-in starts to hurt, which `backends` measures.
    solution = Solver(mesh, LinearElastic(E=200, nu=0.3), bc,
                      backend=IterativeBackend()).solve()
    deformed = solution.deformed_mesh()
    tip = np.abs(solution.u.reshape(-1, 3)[:, 2]).max()

    plotter = Plotter(1, 2, figsize=(11.0, 4.5), title='A 3D cantilever in tetrahedra')
    plotter.plot(mesh, mode='solid', idx=(0, 0),
                 title=f'{len(mesh.elements)} tetrahedra')
    plotter.plot(deformed, solution.von_mises, mode='solid', idx=(0, 1),
                 label='von Mises stress', title='Loaded and deformed')
    return DemoResult(
        [Figure(plotter,
                'The same clamp-and-load as the 2D cantilever, one dimension up: a '
                'tetrahedral mesh, a three-component displacement, and stress recovered '
                'the same way. Only the boundary surface is drawn -- the inside of a '
                'solid is not visible, and there are several times more tets than '
                'surface triangles.')],
        text=(f'tetrahedra          {len(mesh.elements)}\n'
              f'degrees of freedom  {3*len(mesh.vertices)}\n'
              f'peak deflection     {tip:.4f}'),
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

    # The sweep varies kappa, not where it applies, so one conditions figure covers all
    # three: the whole boundary is a film, and each result panel's title says how free
    # a one.
    first = BoundaryConditions()
    first.add_robin(everywhere(), kappa=kappas[0], g=kappas[0]*u_ambient)
    conditions = Plotter()
    conditions.plot(mesh, mode='bc', bc=first)

    solves = []
    for kappa in kappas:
        bc = BoundaryConditions()
        bc.add_robin(everywhere(), kappa=kappa, g=kappa*u_ambient)
        solves.append((f'kappa={kappa:g}', Solver(mesh, equation, bc).solve().u))

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), u_ambient)
    solves.append(('Dirichlet limit', Solver(mesh, equation, bc).solve().u))

    # One scale across the sweep, which is what the demo is claiming with. Renormalized
    # per panel the four look alike and the reader has to compare colorbar ticks; shared,
    # the plate visibly cools towards ambient as kappa rises, and the last two are the
    # same picture -- which *is* the claim that the Robin limit is the Dirichlet solve.
    span = (min(float(u.min()) for _, u in solves), max(float(u.max()) for _, u in solves))
    plotter = Plotter(1, len(solves), title='Robin BCs: convective cooling')
    for i, (name, u) in enumerate(solves):
        plotter.plot(mesh, u, mode='colored', idx=(0, i), label='temperature',
                     title=f'{name}\n{u.min():.1f} - {u.max():.1f}', clim=span)
    return DemoResult([
        Figure(plotter,
               'Convective cooling at three film coefficients, all four on one colour '
               'scale -- so the plate is seen to cool towards ambient as the film opens '
               'up. The last Robin panel and the Dirichlet solve beside it are the same '
               'picture and agree to the digit: the limit, computed both ways.',
               'sweep'),
        Figure(conditions,
               'Robin the whole way round, at the first of the three coefficients. Only '
               'kappa changes across the sweep -- where the condition applies does not.',
               'conditions', setup=True),
    ])

def demo_elasticity_models(mesh, stretch=0.5):
    """Stretch one clamped block three ways: a linear solve, the same physics by energy
    minimisation, and finite strain."""
    # One setup, three paths, and the two comparisons worth making sit side by side.
    #
    # Panels 1 and 2 are the same physics reached differently -- assembling and solving
    # K u = f, against driving Newton on the elastic energy whose stationary point that
    # system is. The displacements come out identical to machine precision, which is
    # what says the energy path is wired up right, and the demo prints the difference
    # rather than asserting it.
    #
    # Their *stress* is not identical, and that is not a discrepancy: the two recover
    # different measures. `LinearElasticForm` reports sigma = D:eps; `EnergyForm`
    # reports the true Cauchy stress J^-1 P F^T at the deformed configuration. Those
    # agree only to O(||grad u||) -- see EnergyForm.derived_fields -- and a 50% stretch
    # is nowhere near that limit.
    #
    # Panel 3 changes the physics rather than the solve. The small-strain measure
    # eps = (grad u + grad u^T)/2 is only the leading term of S = (F^T F - I)/2; under a
    # uniaxial stretch lambda the two read (lambda - 1) and (lambda^2 - 1)/2, so the
    # finite-strain model stiffens as the stretch grows and the linear one cannot.
    #
    # The stress peak sits at the clamped corners, where the imposed displacement is
    # singular, so the median is quoted beside it as the bulk figure.
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, w), [stretch*w, 0])

    linear = LinearElastic(E=200, nu=0.4)
    energy_solver = EnergySolver(mesh, linear, bc)
    solutions = [
        ('Linear solve\n(small strain)', Solver(mesh, linear, bc).solve()),
        ('Energy minimisation\n(small strain)', energy_solver.solve()),
        ('Energy minimisation\n(Green-Lagrange)', EnergySolver(
            mesh, LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE), bc
        ).solve()),
    ]

    conditions = Plotter()
    conditions.plot(mesh, mode='bc', bc=bc)

    plotter = Plotter(1, 3, title=f'One {stretch:.0%} stretch, three ways to solve it')
    for i, (name, solution) in enumerate(solutions):
        vm = solution.von_mises
        plotter.plot(solution.deformed_mesh(), vm, mode='colored', idx=(0, i),
                     label='von Mises stress',
                     title=f'{name}\nmedian {np.median(vm):.0f}, peak {vm.max():.0f}')
    linear_u, energy_u = solutions[0][1].u, solutions[1][1].u
    drift = np.linalg.norm(energy_u - linear_u) / np.linalg.norm(linear_u)
    return DemoResult(
        [Figure(plotter,
                'The first two are the same physics reached two ways -- a linear system, '
                'and Newton on the energy that system is the stationary point of. Their '
                'displacements are identical to machine precision (below); their stress '
                'is not, because the two recover different measures -- sigma = D:eps '
                'against the true Cauchy stress at the deformed configuration, which '
                'agree only for small gradients. The third changes the physics rather '
                'than the solve: Green-Lagrange stiffens as the stretch grows, which '
                'small strain cannot.',
                'stress'),
         Figure(conditions,
                'Both ends are Dirichlet, and the difference between them is the whole '
                'problem: the left is held at zero, the right is displaced to '
                f'{stretch:.0%} of the width. Nothing is loaded -- the stress above is '
                'what it costs to hold that shape.',
                'conditions', setup=True)],
        text=(f'displacement, linear solve vs energy minimisation: '
              f'relative difference {drift:.1e}\n'
              f'minimised elastic energy: {energy_solver.energy(energy_u):.4g}'),
    )

def demo_heat_equation(mesh):
    """Animate transient heat diffusion from a hot bump initial condition."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.5*min(w, h)) + 300

    # Empty on purpose, and stated rather than left as a `None` default: every edge is
    # insulated, which is why the plate levels off at the mean of its initial state
    # instead of cooling towards anything.
    bc = BoundaryConditions()

    # dt sized to the bump's decay, not to a round number: the corner bump loses 99% of
    # its contrast by t=0.4, so a run that long is three quarters flat square. Over
    # t=0.08 the same 40 frames spread the decay out and still reach near-uniform.
    solution = ThetaMethod(dt=0.002, steps=40).run(heat(mesh, bc=bc), u_initial.copy())
    u_values = solution.u
    t_values = solution.t

    # One animated panel, not two. The second was the same field as a 3D surface, and
    # `plot_trisurf` re-tessellates the whole mesh every frame -- it was the single most
    # expensive thing in a gallery build, for a second view of a field the snapshots
    # below already show at six times.
    # A transient problem is posed by two things, and the initial state is the one that
    # decides what the picture looks like.
    setup = Plotter(1, 2)
    setup.plot(mesh, mode='bc', bc=bc, title='Boundary conditions', idx=(0, 0))
    setup.plot(mesh, u_initial, mode='colored', idx=(0, 1), label='temperature',
               title=f'Initial condition u(x, 0)\n{u_initial.min():.1f} - {u_initial.max():.1f}')

    animation = Plotter(1, 1, title='Heat Equation')
    animation.plot_animation(mesh, u_values, mode='colored', label='temperature',
                             titles=[f't={t:.3f}' for t in t_values], idx=(0, 0))

    # The animation renders only on show(), so the diffusion needs a still form too --
    # otherwise this demo contributes nothing to a saved gallery.
    # One scale across the six, spanning the whole run. Renormalized per panel, a field
    # losing 70% of its contrast drew as six near-identical squares under a caption
    # promising it approaches uniform -- the decay was in the colorbars and nowhere else.
    span = (float(np.min(u_values)), float(np.max(u_values)))
    snapshots = Plotter(2, 3, title='Heat Equation: diffusion from the corner')
    for panel, i in enumerate(np.linspace(0, len(u_values) - 1, 6).astype(int)):
        snapshots.plot(mesh, u_values[i], mode='colored', idx=divmod(panel, 3),
                       label='temperature', title=f't={t_values[i]:.3f}', clim=span)

    return DemoResult([
        Figure(animation, 'Crank-Nicolson diffusion of the corner bump.', 'animation'),
        Figure(snapshots,
               'The same run sampled at six times: the corner bump spreads and the '
               'plate approaches a uniform temperature.', 'snapshots'),
        Figure(setup,
               'A hot bump in one corner of a plate whose every edge is insulated. '
               'du/dn = 0 is not an omission but the condition the weak form imposes '
               'where nothing else is written, and here it means no heat can leave -- '
               'so the total is conserved and the plate must level off at the mean of '
               'where it started rather than cooling towards anything.',
               'conditions', setup=True),
    ])

def demo_wave_equation(mesh):  # TODO: Wave energy not fully implemented
    """Animate wave propagation from a bump initial condition, plus a grid of late snapshots."""
    w, h = np.max(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 1])
    wave_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, wave_center, size=0.25*min(w, h))
    dudt_initial = np.zeros(len(mesh.vertices))

    # Empty, so the edges are free rather than pinned. It is the difference the
    # snapshots are of: a free edge reflects a pulse back the same way up, where a
    # clamped one would invert it.
    bc = BoundaryConditions()

    solution = NewmarkMethod(dt=0.03, steps=40).run(wave(mesh, c=1, bc=bc),
                                                    u_initial, dudt_initial)
    u_values = solution.u
    t_values = solution.t

    # Newmark is second order in time, so it is posed by two initial conditions, not one.
    # Both are drawn: the velocity is identically zero, and a panel of nothing is what
    # says the membrane starts at rest -- which is why the pulse spreads outwards in
    # every direction rather than travelling in one.
    setup = Plotter(1, 3)
    setup.plot(mesh, mode='bc', bc=bc, title='Boundary conditions', idx=(0, 0))
    setup.plot(mesh, u_initial, mode='colored', idx=(0, 1), label='displacement',
               title='Initial displacement u(x, 0)')
    setup.plot(mesh, dudt_initial, mode='colored', idx=(0, 2), label='velocity',
               title='Initial velocity du/dt(x, 0) = 0')

    animation = Plotter(1, 1, title='Wave Equation')
    animation.plot_animation(mesh, u_values, mode='surface',
                             titles=[f'Surface t={t:.2f}' for t in t_values], idx=(0, 0))

    # Snapshots from the second half of the run, once the pulse has reflected off the
    # boundary and started interfering with itself. One grid, rather than the window
    # per frame this used to open.
    # Shared z limits, so the six are the same membrane seen at six times rather than
    # six differently-scaled drawings: autoscaled, a pulse that has spread out is drawn
    # to the same height as the one that has not. Spanned over the frames shown and not
    # the whole run -- the tallest thing here is the initial pulse before it disperses,
    # which none of these panels contains, and scaling to it drew every one of them at
    # about a third of its axis.
    shown = [int(i) for i in np.linspace(len(u_values)//2, len(u_values) - 1, 6)]
    span = (min(float(u_values[i].min()) for i in shown),
            max(float(u_values[i].max()) for i in shown))
    snapshots = Plotter(2, 3, title='Wave Equation: reflection and interference')
    for panel, i in enumerate(shown):
        snapshots.plot(mesh, u_values[i], mode='surface', idx=divmod(panel, 3),
                       title=f't={t_values[i]:.2f}', clim=span)

    return DemoResult([
        Figure(animation, 'Newmark time integration of the pulse.', 'animation'),
        Figure(snapshots,
               'Six times from the second half of the run, after the pulse has reflected '
               'off the boundary and begun interfering with itself.', 'snapshots'),
        Figure(setup,
               'A pulse released from rest on a membrane whose edges carry the natural '
               'condition, du/dn = 0. That is a free edge rather than a clamped one, '
               'which is why the pulse reflects the same way up instead of inverted.',
               'conditions', setup=True),
    ])

def demo_linear_elastic(mesh):
    """Solve linear elasticity for a cantilever fixed on the left with a traction load,
    then read four rotation-invariant stress measures off that one solve."""
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

    conditions = Plotter(panel_aspect=4.0)
    conditions.plot(mesh, mode='bc', bc=bc)

    plotter = Plotter(1, 2, title='Linear Elasticity', panel_aspect=4.0)
    plotter.plot(deformed_mesh, solution.von_mises, mode='colored', title='Von Mises stress',
                 label='von Mises stress', idx=(0, 0))
    plotter.plot(mesh, displacements, mode='colored', title='Displacement',
                 label='|u|', idx=(0, 1))

    # The same stress tensor admits other rotation-invariant reductions besides von
    # Mises: mean normal stress, the Tresca measure, and the largest tensile principal
    # value. Each is its own question asked of one solve, not a different problem.
    invariant_fields = [
        ('Von Mises', solution.von_mises),
        ('Pressure', solution.pressure),
        ('Max shear', solution.max_shear),
        ('Max principal', solution.principal_stress[:, -1]),
    ]
    invariants = Plotter(2, 2, title='Stress invariants of the same solve', panel_aspect=4.0)
    for i, (name, values) in enumerate(invariant_fields):
        invariants.plot(deformed_mesh, values, mode='colored', idx=divmod(i, 2), title=name)

    return DemoResult([
        Figure(plotter,
               'The bending stress is largest at the clamp and splits top from bottom '
               '-- tension over the neutral axis, compression under it.',
               'fields'),
        Figure(invariants,
               'Four rotation-invariant reductions of that same stress tensor: distortion, '
               'mean normal stress, the Tresca measure, and the largest tensile principal '
               'value.',
               'invariants'),
        Figure(conditions,
               'Clamped along the left edge, pulled down over the middle of the right '
               'one. Everything between is traction-free, which is what makes this a '
               'cantilever rather than a beam being squeezed.',
               'conditions', setup=True),
    ])

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
    # Only the clamp is a boundary condition here: the load is a body force over every
    # element, not a traction on an edge, so the rest of the boundary is natural.
    conditions = Plotter(panel_aspect=2.0)
    conditions.plot(mesh, mode='bc', bc=bc)

    final_plotter = Plotter(1, 2, title='Topology Optimization', panel_aspect=2.0)
    final_plotter.plot(deformed_mesh, rho_final, mode='colored', title='Topology Optimized Structure',
                       label='density', idx=(0, 0), empty=True)
    final_plotter.plot(deformed_mesh, stress_final, mode='colored', title='Final von Mises stress',
                       label='von Mises stress', idx=(0, 1))
    return DemoResult([
        Figure(animation_plotter, 'Density evolving over the SIMP iterations.',
               'animation'),
        Figure(final_plotter,
               'The converged structure and its stress: material has migrated into a '
               'truss carrying the load back to the supported edge.', 'final'),
        Figure(conditions,
               'Clamped on the left and nothing else: the load here is a body force over '
               'every element rather than a traction on an edge, so it is the one thing '
               'imposed that a picture of the boundary cannot show.',
               'conditions', setup=True),
    ])

def demo_heat_3d(steps=20, n=17):
    """Animate transient heat diffusion on a 3D tetrahedral box, drawn as its boundary surface."""
    # Same box and resolution convention as `elastic_3d`.
    mesh = create_box_mesh(corners=[[0, 0, 0], [4, 1, 1]], resolution=(4*n//2, n//2, n//2))

    w = max(mesh.vertices.flatten()) - min(mesh.vertices.flatten())
    heat_center = np.max(mesh.vertices, axis=0)
    u_initial = bump_function(mesh.vertices, heat_center, mag=50, size=0.3*w) + 300

    solution = ThetaMethod(dt=0.04, steps=steps).run(heat(mesh), u_initial.copy())
    u_values = solution.u
    t_values = solution.t

    animation = Plotter(1, 1, title='Heat Diffusion')
    animation.plot_animation(mesh, u_values, mode='solid', label='temperature',
                             titles=[f't={t:.2f}' for t in t_values], idx=(0, 0))

    return DemoResult([Figure(
        animation,
        'Heat diffusing from a hot corner through a tetrahedral box -- the same solve '
        '`heat` runs in 2D, one dimension up. Only the boundary surface is drawn, so '
        'the interior is not directly visible, but the same diffusion reaches it.')])


SOLVING = 'Solving PDEs'
SOLIDS = 'Solids & structures'
ACCURACY = 'Accuracy & performance'

DEMOS = [
    Demo('poisson', demo_poisson_equation, section=SOLVING, domain=partial(square, 80)),
    Demo('heat', demo_heat_equation, section=SOLVING, domain=square),
    Demo('heat_3d', demo_heat_3d, section=SOLVING, smoke_kwargs={'steps': 3, 'n': 5}),
    Demo('wave', demo_wave_equation, section=SOLVING, domain=square),
    Demo('robin', demo_robin_bc, section=SOLVING, domain=partial(square, 80)),

    # A cantilever is a beam. On the square this used to load, the "bending" was a
    # square bulging sideways, and the stress concentration had nowhere to run to.
    Demo('linear_elastic', demo_linear_elastic, section=SOLIDS,
         domain=partial(beam, 4.0, 1.0, 140)),
    # Stretched end to end, so the domain is incidental; a square keeps the deformed
    # and undeformed shapes comparable at a glance.
    Demo('elasticity_models', demo_elasticity_models, section=SOLIDS,
         domain=partial(square, 60)),
    # Builds its own domain rather than taking one, because the meshing is part of what
    # it shows -- the pipeline demo, from an outline through to a stress. The smoke run
    # loosens the size cap, which is where all of its cost is.
    Demo('stress_concentration', demo_stress_concentration, section=SOLIDS,
         smoke_kwargs={'max_area_fraction': 0.05}),
    # Builds its own box: the only 3D domain, and the tet count is what sets the
    # cost, so the smoke run takes a coarser one.
    Demo('elastic_3d', demo_elastic_3d, section=SOLIDS, smoke_kwargs={'n': 5}),
    # 2:1, because the aspect ratio is what makes SIMP produce the truss it is known
    # for. The resolution is now set by what the *filter* needs rather than by what 40
    # iterations cost: `smoothing_radius` is a fixed physical length, so refining
    # resolves the same structure more finely instead of growing thinner members. At
    # 56 a side that radius spanned about three elements, which is thin cover for the
    # thing keeping the design off a checkerboard; at 140 it spans seven.
    Demo('topology_optimization', demo_topology_optimization, section=SOLIDS,
         domain=partial(beam, 2.0, 1.0, 140)),

    # The point is which oscillations of sin(40 r^2) the space can represent, so this
    # one is meshed finer than the rest: at 40 a side the inner rings alias too. It
    # leads the accuracy section because representation error is what the rest measures.
    Demo('l2_projection', demo_l2_projection, section=ACCURACY, domain=partial(square, 120)),
    # Builds its own sequence of meshes rather than taking a domain: the refinement
    # sequence *is* the demo. The smoke run keeps the two coarsest -- an order needs
    # two points, and the 81x81 solve is most of the cost.
    Demo('convergence', demo_convergence, section=ACCURACY,
         smoke_kwargs={'resolutions': (11, 21), 'elastic_resolutions': (9, 17),
              'step_counts': (16, 32)}),
]
