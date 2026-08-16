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
    LOAD_MMS_FREQUENCY, ConvergenceStudy, elastic_convergence, load_comparison_convergence,
    oscillatory_exact, poisson_convergence, poisson_p2_convergence, solve_load_comparison,
    theta_convergence,
)
from fem.elements import QuadraticTriangleElement
from fem.estimators import residual_estimator
from fem.space import FunctionSpace
from fem.regions import everywhere, on_plane, in_box, intersect
from fem.plot.plotter import Plotter
from fem.equations import Projection, Poisson, LinearElastic, StrainMeasure
from fem.solver import Solver
from fem.mesh.ruppert import RuppertsAlgorithm, create_box_mesh, create_rect_mesh
from fem.problem import heat, wave
from fem.integrators import NewmarkMethod, ThetaMethod
from fem.topology import TopologyOptimizer
from fem.energy_solver import EnergySolver
from fem.buckling import BucklingSolver
from fem.modal import ModalSolver

from demo_registry import Demo, DemoResult, Figure
from domains import beam, column, plate_with_hole_pslg, square, tuning_fork_pslg

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
               'the natural condition, which makes the solution vanish all '
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
    ticks like 2x10^-2, which run into each other.
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
    #   in space  P1 elements are O(h^2) (halve h, quarter the error) for a scalar
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
                'Left: the Poisson error is smooth and one-signed: zero on the boundary '
                'where the solution is pinned exactly, deepest at the centre where the '
                'piecewise-linear space has the most to miss. Middle: halving h quarters '
                'that error, for a scalar unknown and for a coupled vector one alike. '
                'Right: the same measurement against the time step instead, where the '
                'order is not a property of the elements but a choice: backward Euler '
                'buys first order, Crank-Nicolson second, for the same cost per step.')],
        text='\n'.join(rows),
    )

def demo_higher_order(resolutions=(11, 21, 41, 81)):
    """Compare P1 and P2 elements on the same manufactured Poisson problem: quadratic
    elements are third order in L2 where linear ones are second, and so reach a given
    accuracy at far fewer degrees of freedom."""
    # Same problem, two element orders. P2 carries the extra edge-midpoint DOFs that
    # let its solution curve within an element; the rate is what that buys.
    p1_solves = poisson_convergence(resolutions)
    p2_solves = poisson_p2_convergence(resolutions)
    p1 = ConvergenceStudy.from_solves(p1_solves)
    p2 = ConvergenceStudy.from_solves(p2_solves)
    # DOF counts drive the accuracy-per-cost view: P2 spends more unknowns per element,
    # and the question is whether its faster rate pays that back.
    p1_dofs = np.array([FunctionSpace(s.mesh).n_dofs for s in p1_solves])
    p2_dofs = np.array([FunctionSpace(s.mesh, QuadraticTriangleElement).n_dofs
                        for s in p2_solves])

    plotter = Plotter(1, 2, title='Higher-order accuracy: P1 vs P2')
    rate = plotter.chart_ax(idx=(0, 0), xlabel='h', ylabel='L2 error')
    _plot_study(rate, p1, 'P1', 'tab:blue', 2, 'h')
    _plot_study(rate, p2, 'P2', 'tab:orange', 3, 'h')
    rate.set_title('Rate: P2 is third order, P1 second')
    _tidy_log_axis(rate, p1.step)

    cost = plotter.chart_ax(idx=(0, 1), xlabel='degrees of freedom', ylabel='L2 error')
    cost.loglog(p1_dofs, p1.error, 'o-', color='tab:blue', label='P1')
    cost.loglog(p2_dofs, p2.error, 'o-', color='tab:orange', label='P2')
    cost.set_title('Cost: P2 reaches a given accuracy first')
    cost.grid(True, which='both', alpha=0.3)

    rows = ['            fitted order   expected']
    for name, study, expected in (('P1', p1, 2), ('P2', p2, 3)):
        rows.append(f'{name:<12}{study.fitted_order:>9.2f}{expected:>11}')
    return DemoResult(
        [Figure(plotter,
                'Left: on the same meshes, halving h quarters the P1 error (order 2) but '
                'divides the P2 error by eight (order 3): the steeper line is the whole '
                'point of a higher-order element. Right: the same errors against the '
                'number of unknowns. P2 spends more DOFs per element, yet reaches a given '
                'accuracy well to the left of P1, so it is the cheaper choice where the '
                'solution is smooth.')],
        text='\n'.join(rows),
    )

def demo_quadrature_load(resolutions=(11, 21, 41, 81)):
    """Show what sampling the load at the quadrature points buys. Both solves use the same
    P1 elements; they differ only in how the source f becomes the right-hand side: read
    at the vertices, or at the interior quadrature points. The vertex shortcut undershoots
    wherever f swings within an element."""
    k = LOAD_MMS_FREQUENCY

    # Setup: the load and the solution it drives, on a fine mesh so these are the ideal
    # shapes rather than a coarse approximation of them.
    fine = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(41, 41))
    u_fine = oscillatory_exact(fine.vertices)
    f_fine = 2 * (k * np.pi) ** 2 * u_fine     # the source is proportional to u here
    setup = Plotter(1, 2, title='The problem: a source f drives a solution u')
    setup.plot(fine, f_fine, mode='colored', idx=(0, 0), label='source f',
               title='The load: source f')
    setup.plot(fine, u_fine, mode='surface', idx=(0, 1), title='The solution: u')

    # Convergence over the sequence, plus one coarse mesh reused for the slice and the
    # error fields, so the 1D cut is literally a row through the 2D error.
    loads = load_comparison_convergence(resolutions)
    steps = np.array([lc.h for lc in loads])
    nodal = ConvergenceStudy(steps, np.array([lc.nodal_error for lc in loads]))
    sampled = ConvergenceStudy(steps, np.array([lc.sampled_error for lc in loads]))
    cut_lc = solve_load_comparison(15)

    n = cut_lc.n
    xs = np.linspace(0, 1, n)
    j = int(np.argmin(np.abs(xs - 1 / (2 * k))))   # a row through the bump peaks
    row = slice(j * n, (j + 1) * n)
    xf = np.linspace(0, 1, 400)
    u_line = np.sin(k * np.pi * xf) * np.sin(k * np.pi * xs[j])
    nodal_err = np.abs(cut_lc.nodal - cut_lc.exact)
    sampled_err = np.abs(cut_lc.sampled - cut_lc.exact)
    emax = float(max(nodal_err.max(), sampled_err.max()))

    comp = Plotter(2, 2, title='One P1 problem, two ways to build the load')
    cut = comp.chart_ax(idx=(0, 0), xlabel='x', ylabel='u')
    cut.plot(xf, u_line, '-', color='0.45', label='exact u')
    cut.plot(xs, cut_lc.nodal[row], 'o-', color='tab:red', ms=4,
             label='nodal load (f at vertices)')
    cut.plot(xs, cut_lc.sampled[row], 's-', color='tab:blue', ms=4,
             label='sampled load (f at quad. pts)')
    cut.set_title(f'Solution on a slice at y={xs[j]:.2g} (both P1)')

    conv = comp.chart_ax(idx=(0, 1), xlabel='h', ylabel='L2 error')
    _plot_study(conv, nodal, 'nodal load', 'tab:red', 2, 'h')
    _plot_study(conv, sampled, 'sampled load', 'tab:blue', 2, 'h')
    conv.set_title('L2 error: both order 2, sampling wins the constant')
    _tidy_log_axis(conv, steps)

    comp.plot(cut_lc.mesh, nodal_err, mode='colored', idx=(1, 0), clim=(0, emax),
              label='|u_h - u|', title=f'Error with nodal load (h={cut_lc.h:.2g})')
    comp.plot(cut_lc.mesh, sampled_err, mode='colored', idx=(1, 1), clim=(0, emax),
              label='|u_h - u|', title='Error with sampled load')

    coarse = loads[0]
    rows = [f'coarsest mesh (h={coarse.h:.3g}):',
            f'  nodal load    L2 error {coarse.nodal_error:.3e}',
            f'  sampled load  L2 error {coarse.sampled_error:.3e}',
            f'  the nodal shortcut is {coarse.nodal_error / coarse.sampled_error:.1f}x worse']
    return DemoResult(
        [Figure(comp,
                'Both solves use the same P1 (linear) elements; neither is higher order. '
                'They differ only in how the source f becomes the load: the nodal load reads '
                'f at the vertices only (integrating its linear interpolant), the sampled '
                'load reads f at the interior quadrature points. Top-left: on a slice '
                'through a row of bumps, the exact solution (grey) against the two P1 '
                'solutions: the nodal load undershoots each peak. Top-right: both converge '
                'at order 2, the sampled load about 3x lower. Bottom: the absolute error '
                'over the mesh for each, at the same colour scale.',
                'comparison'),
         Figure(setup,
                'The manufactured problem: -div(grad u) = f on the unit square, zero on the '
                'boundary. The load is the source f (left), an oscillating field of '
                'sources and sinks; the solution u (right) is what it drives, a grid of '
                'bumps pinned to zero all around. Both solves in the comparison target this '
                'u and differ only in how f is sampled to build the load.',
                'setup', setup=True)],
        text='\n'.join(rows),
    )

def demo_stress_concentration(traction=1.0, length=6.0, height=3.0, radius=0.3,
                              min_angle=25, max_area_fraction=0.01, circle_segments=192,
                              refinement_iters=34, refinement_budget=11000):
    """Take a plate with a hole from an outline through meshing, boundary conditions and
    adaptive refinement to the stress concentration at its rim, measured against the
    textbook factor of 3."""
    # The one demo that runs the whole pipeline, and so the only one that builds its own
    # mesh instead of being handed one: what the outline was, what Ruppert's was asked
    # for, and where the conditions went are each part of what this is showing, and a
    # domain factory would put all three somewhere the gallery page does not print.
    #
    # A domain with a hole in it has no structured triangulation, so this is a generated
    # mesh going straight into the solver. Nothing downstream of the meshing knows that:
    # the conditions are written against coordinates rather than vertex numbers, so they
    # resolve against whatever triangulation arrives, including the one adaptive
    # refinement (below) rebuilds several times over.
    #
    # The circle needs many more segments than its default here specifically because
    # this mesh is going to be refined well past what the area cap alone would produce:
    # honouring a 48-gon down to elements smaller than one of its own straight sides
    # subdivides the polygon rather than resolving a rounder hole. Pushed as far as it
    # is below (some 4000 triangles from a deliberately coarse start), even a 48-gon's
    # chord (about 0.039 here) is far bigger than the smallest elements it would be
    # asked to sit under; 192 segments (chord about 0.010) is what keeps the polygon
    # ahead of the triangulation rather than the other way around.
    pslg = plate_with_hole_pslg(length, height, radius, segments=circle_segments)
    pslg.validate()
    # Deliberately coarse: the angle bound constrains element shape and says nothing
    # about size, and this area cap is generous rather than tight, because resolving
    # the rim is adaptive refinement's job below, not this uniform pass's. The rim still
    # grades finer than the interior even here, since the polygonalised rim is built
    # from short input segments and Ruppert's honours them; adaptive refinement starts
    # from that head start rather than from scratch.
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area())
    mesh = rupperts.refine()
    n_initial = len(mesh.elements)
    initial_worst_angle = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()

    # The rim takes no condition at all, and that is the point: a free surface is the
    # natural boundary condition of the weak form, so "traction-free" is what an edge
    # means when nothing is said about it. The conditions panel draws it, rather than
    # leaving a reader to notice an absence.
    #
    # The left edge is a roller, not a clamp: pinned normal to itself (x = 0) so the
    # plate cannot drift, free tangentially (y) so it can still narrow as it stretches
    # -- the Poisson contraction a real clamp would resist. That resistance is itself a
    # local disturbance with nothing to do with the hole, and it used to compete with
    # the hole for the adaptive-refinement estimator's attention (see below). A roller
    # cannot be built from one `add` alone: pinning y anywhere along the edge would
    # resist the same contraction a clamp does, so a second condition pins y at just
    # the one corner point the two conditions share, removing the last rigid-body mode
    # without adding back what the roller exists to avoid.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, None])
    bc.add(BCType.DIRICHLET, intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0])
    bc.add(BCType.NEUMANN, on_plane(0, length), [traction, 0])

    # Adaptive refinement, driven by this same equation's residual estimator
    # (residual_estimator, from fem.estimators), replaces the uniform mesh above with one built
    # by repeatedly re-solving and splitting wherever the estimator finds the most
    # error: everything the rest of this demo plots and measures is read off the
    # result of this loop, not off the coarse mesh it started from.
    #
    # Thirty-four rounds from a starting mesh this coarse spend a real share of their
    # budget bringing the whole plate up to a baseline before they can behave like
    # they are chasing the hole specifically; a finer starting mesh reaches that
    # point in far fewer rounds (see BACKLOG.md), but starting coarse and letting
    # this loop do more of the work is the trade made here. The ceiling on pushing
    # this further isn't the budget, it's the circle above: past roughly this many
    # rounds the smallest elements at the rim start to undercut even a 192-gon's own
    # chord length, at which point more refinement would be subdividing the polygon
    # rather than resolving a rounder hole.
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc)
    solution = AdaptiveRefinement(
        solver, residual_estimator(equation),
        max_triangles=n_initial + refinement_budget, max_iters=refinement_iters,
    ).run()
    mesh = solver.mesh
    sigma_xx = solution.stress[:, 0, 0]

    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    # A vertical strip through the hole's centre: the line the concentration decays
    # along, from the rim out to the far field. The geometry is known here rather than
    # measured back off the mesh, which building the domain in the demo buys.
    strip = np.abs(centroids[:, 0] - length/2) < 0.4*radius
    order = np.argsort(centroids[strip, 1])
    y_strip, ratio_strip = centroids[strip, 1][order], (sigma_xx[strip] / traction)[order]
    peak = ratio_strip.max()

    # Kirsch's factor of 3 is the infinite-plate limit, and this plate is finite, so
    # the measured peak sits above it: the hole removes section, which raises the
    # stress the remaining material carries. Sampled at three hole/height ratios (0.20,
    # 0.15, 0.12) it reads 3.24, 3.16, 3.12: falling toward 3 as the hole shrinks, and
    # cleanly enough to say so: adaptive sampling reads the peak from elements the
    # estimator put exactly where the gradient is steepest, rather than from whichever
    # elements a uniform cap happened to land nearby.
    #
    # Even so, only one digit of that is worth quoting. Refining further does not
    # settle it much closer than this: 3.23, 3.24, 3.24, 3.24 over 3600, 4249, 5209 and
    # 5614 elements. That is tighter than a uniform mesh manages at any single size
    # (the point of putting the resolution where the gradient is steepest), but reading
    # the true rim value would still mean extrapolating to the boundary rather than
    # sampling near it.

    # Two figures rather than one: five panels of a 2:1 plate and a chart do not share a
    # sensible aspect ratio, and a grid that size thumbnails to nothing legible.
    built = Plotter(1, 2, title='From an outline to a solvable mesh', panel_aspect=2.0)
    built.plot(mesh, mode='mesh', idx=(0, 0),
               title=f'{len(mesh.elements)} triangles\n'
                     f'(adaptively refined from {n_initial})')
    # The input segments over the triangulation, rather than the outline in a panel of
    # its own: four corners and a polygonalised circle is an almost empty picture alone,
    # and drawn here it shows which of them the mesher kept and which it split. Fixed
    # data from the original PSLG, so it overlays the refined mesh as validly as it did
    # the coarse one.
    built.get_ax((0, 0)).add_collection(LineCollection(
        rupperts.vertices[rupperts.segments], colors='blue', linewidths=1.0))
    built.plot(mesh, mode='bc', bc=bc, idx=(0, 1), title='Boundary conditions')

    plotter = Plotter(1, 2, title='Stress concentration around a hole', panel_aspect=3.0)
    plotter.plot(mesh, sigma_xx, mode='colored', idx=(0, 0), label='sigma_xx',
                 title=f'{len(mesh.elements)} triangles after adaptive refinement')
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

    # Ruppert's own quality guarantee does not survive: `min_angle` bounds what
    # RuppertsAlgorithm builds, but red-green refinement bisects existing triangles
    # rather than re-triangulating for shape, so it is not a Delaunay construction and
    # carries no angle guarantee of its own. Worth reporting rather than hiding:
    # `tests/test_refinement_conformity.py` covers that the mesh stays conforming
    # through this, not that it stays well-shaped.
    worst_angle = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()
    # rupperts.boundary_loops describes the initial triangulation's boundary facets,
    # not the refined mesh's; adaptive refinement does not carry it forward (see
    # BACKLOG.md), so this counts the rim facets Ruppert's produced, before refinement
    # added more of its own.
    rim_facets = int(np.sum(rupperts.boundary_loops == 1))
    return DemoResult(
        [Figure(built,
                f'Left: the mesh after adaptive refinement, with the outline it started '
                f'from in blue: {n_initial} triangles from a deliberately coarse '
                f'uniform pass, grown to {len(mesh.elements)} by putting the rest where '
                f'the residual estimator found the most error. Right: where the '
                f'conditions went. The rim and the long edges carry none, which is not '
                f'an omission but the natural condition of the weak form: an edge '
                f'nothing is said about is traction-free. Every one of these is written '
                f'against coordinates rather than vertex numbers, which lets '
                f'them be placed on a mesh no one laid out by hand, including the one '
                f'adaptive refinement rebuilds several times over.',
                'built'),
         Figure(plotter,
                f'A plate pulled from the right, with the hole left traction-free. The '
                f'stress crowds into the material either side of the hole and relaxes '
                f'to the applied value within about a diameter, peaking at {peak:.1f}x '
                f'the applied stress, just above the classic Kirsch factor of 3x that '
                f'holds for a hole in an infinite plate.',
                'stress', thumbnail=True)],
        text=(f'outline points           {len(pslg.vertices)}  '
              f'(rectangle + polygonalised rim)\n'
              f'initial elements         {n_initial}\n'
              f'adaptively refined to    {len(mesh.elements)}\n'
              f'worst angle, initial     {initial_worst_angle:.1f} deg   '
              f'(asked for {min_angle})\n'
              f'worst angle, refined     {worst_angle:.1f} deg   '
              f'(red-green carries no angle guarantee)\n'
              f'boundary edges           {len(mesh.boundary)}   '
              f'({rim_facets} of them the hole rim, before refinement)\n'
              f'applied traction         {traction:.3g}\n'
              f'hole diameter / height   {2*radius/height:.2f}\n'
              f'peak sigma_xx / applied  {peak:.2f}   (Kirsch, infinite plate: 3)'),
    )

def demo_elastic_3d(n=17):
    """Bend a 3D cantilever beam of tetrahedra, drawn as its boundary surface."""
    # The package solves in 3D throughout: the same assembly, the same element
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
                'the same way. Only the boundary surface is drawn: the inside of a '
                'solid is not visible, and there are several times more tets than '
                'surface triangles.')],
        text=(f'tetrahedra          {len(mesh.elements)}\n'
              f'degrees of freedom  {3*len(mesh.vertices)}\n'
              f'peak deflection     {tip:.4f}'),
    )

def demo_robin_bc(mesh):
    """Cool a heated plate through a convective boundary, sweeping the Robin coefficient."""
    # du/dn + kappa*(u - u_ambient) = 0: heat generated inside escapes through a boundary
    # film, and kappa says how freely. The other two condition types are its limits:
    # kappa -> 0 is insulated (Neumann) and kappa -> infinity pins u to ambient
    # (Dirichlet), so the sweep ends on a Dirichlet solve the last Robin panel should
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
    # same picture, which is the claim that the Robin limit is the Dirichlet solve.
    span = (min(float(u.min()) for _, u in solves), max(float(u.max()) for _, u in solves))
    plotter = Plotter(1, len(solves), title='Robin BCs: convective cooling')
    for i, (name, u) in enumerate(solves):
        plotter.plot(mesh, u, mode='colored', idx=(0, i), label='temperature',
                     title=f'{name}\n{u.min():.1f} - {u.max():.1f}', clim=span)
    return DemoResult([
        Figure(plotter,
               'Convective cooling at three film coefficients, all four on one colour '
               'scale, so the plate is seen to cool towards ambient as the film opens '
               'up. The last Robin panel and the Dirichlet solve beside it are the same '
               'picture and agree to the digit: the limit, computed both ways.',
               'sweep'),
        Figure(conditions,
               'Robin the whole way round, at the first of the three coefficients. Only '
               'kappa changes across the sweep; where the condition applies does not.',
               'conditions', setup=True),
    ])

def demo_elasticity_models(mesh, stretch=0.5):
    """Stretch one clamped block three ways: a linear solve, the same physics by energy
    minimisation, and finite strain."""
    # One setup, three paths, and the two comparisons worth making sit side by side.
    #
    # Panels 1 and 2 are the same physics reached differently: assembling and solving
    # K u = f, against driving Newton on the elastic energy whose stationary point that
    # system is. The displacements come out identical to machine precision, which
    # says the energy path is wired up right, and the demo prints the difference
    # rather than asserting it.
    #
    # Their stress is not identical, and that is not a discrepancy: the two recover
    # different measures. `LinearElasticForm` reports sigma = D:eps; `EnergyForm`
    # reports the true Cauchy stress J^-1 P F^T at the deformed configuration. Those
    # agree only to O(||grad u||) (see EnergyForm.derived_fields), and a 50% stretch
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
                'The first two are the same physics reached two ways: a linear system, '
                'and Newton on the energy that system is the stationary point of. Their '
                'displacements are identical to machine precision (below); their stress '
                'is not, because the two recover different measures: sigma = D:eps '
                'against the true Cauchy stress at the deformed configuration, which '
                'agree only for small gradients. The third changes the physics rather '
                'than the solve: Green-Lagrange stiffens as the stretch grows, which '
                'small strain cannot.',
                'stress'),
         Figure(conditions,
                'Both ends are Dirichlet, and the difference between them is the whole '
                'problem: the left is held at zero, the right is displaced to '
                f'{stretch:.0%} of the width. Nothing is loaded; the stress above is '
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
    # `plot_trisurf` re-tessellates the whole mesh every frame; it was the single most
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

    # The animation renders only on show(), so the diffusion needs a still form too;
    # otherwise this demo contributes nothing to a saved gallery.
    # One scale across the six, spanning the whole run. Renormalized per panel, a field
    # losing 70% of its contrast drew as six near-identical squares under a caption
    # promising it approaches uniform; the decay was in the colorbars and nowhere else.
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
               'where nothing else is written, and here it means no heat can leave, '
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
    # says the membrane starts at rest, which is why the pulse spreads outwards in
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
    # the whole run: the tallest thing here is the initial pulse before it disperses,
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
    # deflection near 9% of the span: a 4:1 beam is compliant enough that the load
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

def demo_buckling(length=24.0, height=1.0, n_length=48, n_across=6, n_modes=3,
                  sweep_lengths=(16.0, 20.0, 28.0, 40.0)):
    """Find the loads at which a slender column buckles and the shapes it buckles into,
    then check them against Euler three ways: the mode shapes of a pinned column, the
    effective-length factors of four end conditions, and the 1/L^2 slenderness law.

    Euler's column formula (Leonhard Euler, 1744) is the exact critical load of an ideal
    slender elastic column, P_cr = pi^2 E* I / (K L)^2, and plays the role the manufactured
    solution does for the steady solvers: the analytic answer the computed one is held to."""
    # Buckling is an eigenproblem, not a K u = b solve: a reference load puts the column
    # under a prestress, and BucklingSolver assembles the geometric stiffness K_g from it
    # and solves K phi = -lambda K_g phi. The load factor lambda multiplies the reference
    # load to reach the buckling load. Quadratic (P2) elements throughout: the
    # constant-strain triangle locks in bending and would need a mesh refined hard through
    # the thickness to reach Euler, where P2 matches it on this coarse one.
    E, nu = 200.0, 0.3
    E_star = E / (1 - nu**2)     # plane-strain effective modulus, the one bending sees
    moment = height**3 / 12      # second moment of area of the rectangular section
    n_across += n_across % 2      # a vertex on the neutral axis, for the pinned anchor
    equation = LinearElastic(E, nu)

    def solve_buckling(mesh, bc, span, modes=n_modes):
        solver = BucklingSolver(mesh, equation, bc, n_modes=modes,
                                element_type=QuadraticTriangleElement)
        solution = solver.solve()
        # The load factor multiplies the reference load; the physical buckling load is
        # that factor times the actual axial force the column carries, read at mid-span
        # where it is uniform and clear of the end disturbances.
        centroids = mesh.vertices[mesh.elements].mean(axis=1)
        dy = span / (len(np.unique(mesh.vertices[:, 1])) - 1)
        midspan = np.abs(centroids[:, 1] - span / 2) < dy
        axial = -float(np.mean(solver.reference.stress[midspan, 1, 1])) * height
        return solution, solution.load_factors * axial

    # The four classic end conditions. What sets an end's effective-length factor is
    # whether it can rotate, and in a continuum that is the axial DOF: a traction-loaded
    # edge (u_y free) rotates (a pin or a free end) while an imposed uniform axial
    # displacement (u_y fixed) cannot (a clamp). u_x = 0 along a whole edge holds the end
    # transversely without touching its rotation, which is a pin rather than a point load.
    # The column stands along y, so the ends are at y = 0 and y = span and the load pushes
    # down the axis in -y.
    def cantilever(span):   # fixed-free, K = 2
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])
        bc.add(BCType.NEUMANN, on_plane(1, span), [0, -1.0])
        return bc

    def pinned(span):       # pinned-pinned, K = 1
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, None])
        bc.add(BCType.DIRICHLET, intersect(on_plane(1, 0.0), on_plane(0, height / 2)), [0, 0])
        bc.add(BCType.DIRICHLET, on_plane(1, span), [0, None])
        bc.add(BCType.NEUMANN, on_plane(1, span), [0, -1.0])
        return bc

    def fixed(span):        # fixed-fixed, K = 1/2
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])
        bc.add(BCType.DIRICHLET, on_plane(1, span), [0, -0.02 * span])
        return bc

    def fixed_pinned(span):  # fixed-pinned, K ~ 0.7
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])
        bc.add(BCType.DIRICHLET, on_plane(1, span), [0, None])
        bc.add(BCType.NEUMANN, on_plane(1, span), [0, -1.0])
        return bc

    ends = [('Cantilever\n(fixed-free)', cantilever, 2.0),
            ('Pinned-pinned', pinned, 1.0),
            ('Fixed-fixed', fixed, 0.5),
            ('Fixed-pinned', fixed_pinned, 0.699)]

    mesh = column(length, height, n_length, n_across)

    def buckled(solution, i, span):
        """The mesh deformed by mode `i`, scaled so its bow is a fixed fraction of span,
        and the signed transverse displacement to colour it by."""
        n_v = len(mesh.vertices)
        transverse = solution.modes[i].reshape(-1, 2)[:n_v, 0]
        scale = 0.14 * span / np.abs(transverse).max()
        return solution.mode_mesh(i, scale), scale * transverse

    # -- 1. Mode shapes of a pinned column: the buckling analogue of vibration modes ----
    # Upright columns in a row, so the half-waves of successive modes sit side by side, with
    # one glyph-and-colour key below all of them (fig.supxlabel) rather than per panel.
    pinned_solution, pinned_loads = solve_buckling(mesh, pinned(length), length)
    pinned_bc = pinned(length)
    modes = Plotter(1, n_modes, figsize=(2.4 * n_modes, 6.6), axis_labels=False,
                    title='Buckling modes of a pinned-pinned column')
    for i in range(n_modes):
        shape, colour = buckled(pinned_solution, i, length)
        modes.plot(shape, colour, mode='colored', idx=(0, i), cmap='coolwarm', colorbar=False,
                   title=f'Mode {i+1}: P_cr = {pinned_loads[i]:.3g}\n'
                         f'({i+1} half-wave{"s" if i else ""})')
        # The pin/load glyphs, on the deformed shape so the load rides the moving end.
        modes.overlay_supports(mesh, pinned_bc, idx=(0, i), coords=shape.vertices)
    modes.fig.supxlabel(
        'Blue triangles: the pinned ends, held sideways but free to rotate. Red arrow: the '
        'compressive load. Colour: sideways deflection, whose sign and amplitude are '
        'arbitrary, so read the shape and the load, not the colour direction or size.',
        fontsize='small')

    # -- 2. Effective length: the same column, four ways to hold its ends ---------------
    measured = {}
    factor_plots = Plotter(1, len(ends), figsize=(2.4 * len(ends), 6.6), axis_labels=False,
                           title='End conditions set the effective length')
    for col, (name, make_bc, K_ideal) in enumerate(ends):
        end_bc = make_bc(length)
        solution, loads = solve_buckling(mesh, end_bc, length, modes=1)
        K_measured = np.pi / length * np.sqrt(E_star * moment / loads[0])
        measured[name] = (K_measured, K_ideal, loads[0])
        shape, colour = buckled(solution, 0, length)
        factor_plots.plot(shape, colour, mode='colored', idx=(0, col), cmap='coolwarm', colorbar=False,
                          title=f'{name.splitlines()[0]}\nK = {K_measured:.2f} (Euler {K_ideal:g})\n'
                                f'P_cr = {loads[0]:.3g}')
        # Each end's supports drawn on it: a wall clamps, triangles pin, arrows load.
        factor_plots.overlay_supports(mesh, end_bc, idx=(0, col), coords=shape.vertices)

    # -- 3. Euler's laws: the 1/L^2 slenderness curve and the effective-length factors ---
    sweep = [(L, solve_buckling(column(L, height, max(32, int(2 * L)), n_across),
                                pinned(L), L, modes=1)[1][0]) for L in sweep_lengths]
    sweep_L = np.array([L for L, _ in sweep])
    sweep_P = np.array([P for _, P in sweep])
    slope = np.polyfit(np.log(sweep_L), np.log(sweep_P), 1)[0]

    laws = Plotter(1, 2, title="Against Euler's column theory")
    curve = laws.chart_ax(idx=(0, 0), xlabel='length L', ylabel='critical load P_cr')
    curve.loglog(sweep_L, sweep_P, 'o', color='tab:blue', label=f'computed (slope {slope:.2f})')
    dense_L = np.linspace(sweep_L.min(), sweep_L.max(), 100)
    curve.loglog(dense_L, np.pi**2 * E_star * moment / dense_L**2, '-', color='tab:red',
                 alpha=0.6, label='Euler  pi^2 E* I / L^2')
    curve.set_title('Pinned column: P_cr goes as 1/L^2')
    curve.grid(True, which='both', alpha=0.3)

    names = [n.splitlines()[0] for n, _, _ in ends]
    K_meas = [measured[n][0] for n, _, _ in ends]
    K_ideal = [measured[n][1] for n, _, _ in ends]
    bars = laws.chart_ax(idx=(0, 1), xlabel='', ylabel='effective-length factor K')
    x = np.arange(len(names))
    bars.bar(x - 0.2, K_ideal, 0.4, color='tab:red', alpha=0.6, label='Euler')
    bars.bar(x + 0.2, K_meas, 0.4, color='tab:blue', label='computed')
    bars.set_xticks(x, names, rotation=20, ha='right', fontsize='small')
    bars.set_title('Effective-length factor by end condition')
    bars.grid(True, axis='y', alpha=0.3)

    # -- 4. How the pinned column is posed ----------------------------------------------
    conditions = Plotter(panel_aspect=0.7)   # tall and narrow, matching the upright column
    conditions.plot(mesh, mode='bc', bc=pinned(length))

    ratios = '   '.join(f'{n.splitlines()[0]}/pinned {measured[n][2] / measured["Pinned-pinned"][2]:.2f}'
                        for n, _, _ in ends if n != 'Pinned-pinned')
    text = ('Euler (1744): an ideal slender column buckles at P_cr = pi^2 E* I / (K L)^2.\n'
            'This demo reproduces it three ways: mode shapes, end conditions, slenderness.\n\n'
            'effective-length factor K (measured vs Euler):\n'
            + '\n'.join(f'  {n.splitlines()[0]:<14} {measured[n][0]:.3f}  (Euler {measured[n][1]:g})'
                        for n, _, _ in ends)
            + f'\nslenderness law    P_cr ~ L^{slope:.2f}   (Euler exponent -2)\n'
            + f'buckling-load ratios (Euler 0.25 : 4 : 2.05):  {ratios}')

    return DemoResult([
        Figure(modes,
               'A pinned column buckles into half-sine waves. Mode 1 is a single half-wave '
               'at the lowest load, the shape a real column takes. Each higher mode adds a '
               'half-wave and costs n^2 as much (mode 2 is ~4x mode 1), and is reached only '
               'if the lower ones are braced out: a support at mid-span, a node of mode 2 '
               'but not mode 1, buys the jump to it. This is the buckling analogue of '
               'vibration modes, one K phi = -lambda K_g phi eigenproblem: the shapes are '
               'its eigenvectors and the load factors its eigenvalues.',
               'modes', thumbnail=True),
        Figure(factor_plots,
               'The same slender column, its ends held four ways (a blue hatched wall clamps '
               'an end against rotation, blue triangles pin it (free to rotate), red arrows '
               'are the load) buckles at loads spanning 16x. Clamping shortens the effective '
               'length K*L '
               'the column buckles over, from 2L free-standing down to L/2 with both ends '
               'fixed, and the load goes as 1/K^2. The measured K sits within a few percent '
               'of Euler\'s 2, 1, 1/2 and ~0.7; the small excess is a real continuum effect, a '
               'clamp in a solid adding a little Saint-Venant stiffening an ideal beam has none of.',
               'end_conditions'),
        Figure(laws,
               'Euler\'s column formula (1744) is the exact buckling load of an ideal slender '
               'elastic column, P_cr = pi^2 E* I / (K L)^2, the analytic truth this whole '
               'demo checks against. Left: sweeping the length of a pinned column, the critical '
               'load falls as 1/L^2 (a slope of -2 on log-log) and lands on it, with '
               'E* = E/(1-nu^2) the plane-strain modulus a 2D solve sees. Right: the '
               'effective-length factor K read back from each end condition\'s buckling load, '
               'against the textbook values.',
               'laws'),
        Figure(conditions,
               'A pinned-pinned column: both ends held across their width (u_y = 0) so they '
               'stay in line but can still rotate, one point anchoring the axial slide, and a '
               'compressive traction on the right. The transverse support and the axial load '
               'share the loaded edge (a roller carrying a tangential traction) which the '
               'buckling column is the case that first needs.',
               'conditions', setup=True),
    ], text=text)


def demo_modal(tine_length=0.088, tine_thickness=0.004, n_across_tine=5, min_angle=27,
               n_modes=6, n_shown=4, sweep_lengths=(0.075, 0.088, 0.105, 0.125),
               n_frames=24):
    """Find the natural frequencies and mode shapes of a steel tuning fork, meshed from
    its own outline, and check them against beam theory: the fork is tuned to concert A
    by the cantilever-tine formula, and its voice is read back against the 1/L^2 law.

    Modal analysis is load-free: unlike buckling, no reference solve enters. The natural
    frequencies solve `K phi = omega^2 M phi` (elastic stiffness against consistent mass)
    and are a property of the structure alone (its shape, material, and supports), the
    way a bell's pitch is the bell's and not the striker's."""
    # Real SI steel, so the frequencies come out in Hz a musician would recognise: this is
    # the one demo that names a pitch, and the abstract E=200 the others use would not land
    # on 440. E* = E/(1-nu^2) is the plane-strain modulus a 2D solve sees, the same
    # effective modulus the buckling demo's bending uses.
    E, NU, RHO = 2.0e11, 0.3, 7850.0             # Young's (Pa), Poisson, density (kg/m^3)
    E_STAR = E / (1 - NU**2)
    BETA1_SQ = 1.875104**2                        # first fixed-free beam root, squared

    def cantilever_hz(length, thickness=tine_thickness):
        """The ideal clamped-free tine's fundamental (Hz): a bare beam, with no base."""
        return BETA1_SQ / (2*np.pi) * (thickness / length**2) * np.sqrt(E_STAR / (12*RHO))

    def clamp():
        """Grounded at the stem base: the fork's node, held without damping the voice."""
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), [0, 0])
        return bc

    def solve_fork(length, modes, across=n_across_tine):
        """Mesh a fork of tine length `length` from its outline, and solve its modes."""
        pslg = tuning_fork_pslg(tine_length=length, tine_thickness=tine_thickness)
        pslg.validate()
        # Element size is set by resolving the thin tine, not the fork's overall extent:
        # bending curves across the tine, and too few elements there under-resolve the
        # very mode the demo is about.
        mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=0.5*(tine_thickness/across)**2).refine()
        solution = ModalSolver(mesh, LinearElastic(E, NU), clamp(), n_modes=modes,
                               element_type=QuadraticTriangleElement, density=RHO).solve()
        return mesh, solution

    def voice_index(mesh, solution):
        """The acoustic mode: the lowest whose two tine tips swing in opposite directions.

        A clamped fork's low modes come in pairs: the tips moving together (a rocking
        that shakes the stem, damped the moment the fork is held there) or oppositely.
        The oppositely-moving one keeps the stem still and rings; it is the lowest with
        the tip transverse motions of opposite sign.
        """
        verts = mesh.vertices
        tips = verts[:, 1] > verts[:, 1].max() - 0.2*tine_length
        left, right = tips & (verts[:, 0] < 0), tips & (verts[:, 0] > 0)
        for i in range(len(solution.frequencies)):
            u_x = solution.modes[i].reshape(-1, 2)[:len(verts), 0]
            if u_x[left].mean() * u_x[right].mean() < 0:
                return i
        return 0

    # -- 1. The modes: the shapes the fork rings in, and their pitches ------------------
    mesh, solution = solve_fork(tine_length, n_modes)
    freqs = solution.frequencies
    voice = voice_index(mesh, solution)
    n_v = len(mesh.vertices)

    def mode_shape(i):
        """Mode `i` as a deformed mesh, and the signed transverse motion colouring it."""
        transverse = solution.modes[i].reshape(-1, 2)[:n_v, 0]
        scale = 0.12 * tine_length / np.abs(transverse).max()
        return solution.mode_mesh(i, scale), scale * transverse

    def hide_x_ticks(plotter, idx):
        """Drop the x-axis ticks on a tall, thin fork panel, where the millimetre-scale
        labels only collide; the y-axis carries the scale."""
        ax = plotter.get_ax(idx)
        ax.tick_params(axis='x', labelbottom=False, bottom=False)

    modes = Plotter(1, n_shown, figsize=(2.9*n_shown, 6.0), axis_labels=False,
                    title="A tuning fork's natural modes and their pitches")
    for i in range(n_shown):
        shape, colour = mode_shape(i)
        lim = float(np.abs(colour).max())
        tag = '  (the voice)' if i == voice else ''
        # One shared caption names the colour below; a per-bar label would repeat it.
        modes.plot(shape, colour, mode='colored', idx=(0, i), cmap='coolwarm',
                   clim=(-lim, lim), title=f'Mode {i+1}: {freqs[i]:.0f} Hz{tag}')
        modes.overlay_supports(mesh, clamp(), idx=(0, i), coords=shape.vertices)
        hide_x_ticks(modes, (0, i))
    modes.fig.supxlabel(
        'Colour: sideways (transverse) displacement of the mode. Its sign and amplitude '
        'are arbitrary; the pattern of motion is what is physical.', fontsize='small')

    # -- 2. The voice, flexing: the mode as motion rather than a frozen shape -----------
    transverse = solution.modes[voice].reshape(-1, 2)[:n_v, 0]
    amp = 0.12 * tine_length / np.abs(transverse).max()
    phases = np.cos(np.linspace(0, 2*np.pi, n_frames, endpoint=False))
    frames = [solution.mode_mesh(voice, amp*c) for c in phases]
    colour = amp * transverse                    # fixed colour; only the geometry moves
    lim = float(np.abs(colour).max())
    swing = Plotter(1, 1, figsize=(4.6, 6.2),
                    title=f'The voice mode swinging: {freqs[voice]:.0f} Hz')
    swing.plot_animation(mesh, [colour]*n_frames, mode='colored', meshes=frames,
                         cmap='coolwarm', cbar_lims=(-lim, lim), label='sideways motion',
                         titles=['']*n_frames)
    # Say plainly what the animation is: not a dynamics simulation, but the mode's own
    # exact solution evaluated frame by frame. A standing-wave mode separates into a fixed
    # shape times cos(omega t), so no time-stepping is needed, and none is done here.
    swing.fig.supxlabel(
        "Not a time-stepped simulation: this is the mode's exact\n"
        'motion phi cos(omega t), one undamped, idealized mode\n'
        'at exaggerated amplitude. Only the shape and frequency\n'
        'are physical, not the size; a real fork mixes modes and\n'
        'rings down.', fontsize='small')

    # -- 3. Euler-Bernoulli's tuning law: pitch falls as 1/L^2 --------------------------
    sweep_L = np.array(sweep_lengths)
    sweep_f = []
    for length in sweep_lengths:
        swept_mesh, swept = solve_fork(length, max(voice + 2, 3), across=max(3, n_across_tine - 1))
        sweep_f.append(swept.frequencies[voice_index(swept_mesh, swept)])
    sweep_f = np.array(sweep_f)
    slope = np.polyfit(np.log(sweep_L), np.log(sweep_f), 1)[0]

    law = Plotter(1, 2, title='Against Euler-Bernoulli beam theory')
    curve = law.chart_ax(idx=(0, 0), xlabel='tine length L (m)', ylabel='voice frequency (Hz)')
    curve.loglog(sweep_L, sweep_f, 'o', color='tab:blue', label=f'computed fork (slope {slope:.2f})')
    dense = np.linspace(sweep_L.min(), sweep_L.max(), 100)
    curve.loglog(dense, cantilever_hz(dense), '-', color='tab:red', alpha=0.6,
                 label='ideal tine  f ~ 1/L^2')
    curve.axvline(tine_length, color='0.6', ls=':', label=f'this fork ({tine_length*1000:.0f} mm)')
    curve.set_title('Pitch falls as 1/L^2')
    curve.grid(True, which='both', alpha=0.3)
    curve.legend(fontsize='small')

    bars = law.chart_ax(idx=(0, 1), ylabel='frequency (Hz)')
    x = np.arange(n_shown)
    bars.bar(x, freqs[:n_shown],
             color=['tab:red' if i == voice else 'tab:blue' for i in range(n_shown)])
    bars.axhline(440.0, color='0.4', ls='--', label='concert A (440 Hz)')
    bars.axhline(cantilever_hz(tine_length), color='tab:red', ls=':', alpha=0.6,
                 label=f'ideal tine ({cantilever_hz(tine_length):.0f} Hz)')
    bars.set_xticks(x, [str(i + 1) for i in range(n_shown)])
    bars.set_xlabel('mode')
    bars.set_title('First modes (voice in red)')
    bars.grid(True, axis='y', alpha=0.3)
    bars.legend(fontsize='small')

    # -- 4. How the fork is posed: an outline, meshed, held at the stem -----------------
    built = Plotter(1, 2, figsize=(6.0, 7.0), title='From an outline to a meshed fork')
    built.plot(mesh, mode='mesh', idx=(0, 0), title=f'{len(mesh.elements)} triangles')
    hide_x_ticks(built, (0, 0))
    built.plot(mesh, mode='bc', bc=clamp(), idx=(0, 1), title='Clamped at the stem base')

    ideal = cantilever_hz(tine_length)
    text = (
        f'A steel tuning fork (E={E:.0e} Pa, rho={RHO:.0f} kg/m^3), meshed from its outline.\n'
        f'tine length x thickness   {tine_length*1000:.0f} x {tine_thickness*1000:.1f} mm\n'
        f'mesh                      {len(mesh.elements)} P2 triangles\n\n'
        f'ideal clamped tine (beam theory)   {ideal:.0f} Hz\n'
        f'fork voice (mode {voice+1}, computed)      {freqs[voice]:.0f} Hz   '
        f'({100*(freqs[voice]/ideal - 1):+.0f}%: the base is not a rigid clamp)\n'
        f'first {n_shown} modes (Hz)             ' + '  '.join(f'{f:.0f}' for f in freqs[:n_shown]) + '\n'
        f'tuning law   f ~ L^{slope:.2f}         (beam-theory exponent -2)'
    )

    return DemoResult([
        Figure(modes,
               'The fork rings in these shapes, each at its own pitch. The low modes come '
               'in pairs: the tips swing together (a rocking that shakes the stem, damped '
               'the moment the fork is held there) or oppositely, and the oppositely '
               'moving one, which leaves the stem still, is "the voice" the fork is made '
               'for. A mode is an eigenvector, so its sign is free (the shading is that '
               'free sign, red one way and blue the other) and its amplitude unset, scaled '
               'here only to be visible: read the shape and the frequency, not the colour '
               'direction or the size.',
               'modes', thumbnail=True),
        Figure(swing,
               'The voice mode as motion rather than a frozen shape: phi cos(omega t), the '
               'tines flexing apart and together at the natural frequency. Any free '
               'vibration is a sum of the modes, each ringing at its own rate; struck, a '
               'fork sheds the others and settles onto this one, which is why it sounds a '
               'single clean tone.',
               'swing'),
        Figure(law,
               'Left: the fork is a pair of clamped-free tines, so beam theory sets its '
               'voice at f = (1.875)^2 / (2 pi) . (t / L^2) . sqrt(E* / 12 rho), the pitch '
               'falling as 1/L^2. Sweeping the tine length, the computed fork tracks that '
               'slope and sits a little below the ideal-tine line, because a real fork\'s '
               'base yields where beam theory assumes a rigid clamp. Right: this fork\'s '
               'first modes: the voice (red) lands near concert A, a few percent under '
               'the ideal tine for the same base-compliance reason.',
               'law'),
        Figure(built,
               'The fork is one non-convex outline (stem, base, two tines with a slot) '
               'meshed by Ruppert\'s algorithm, with no structured grid. It is held only at '
               'the stem base: that Dirichlet clamp grounds the structure (a free body has '
               'rigid-body modes the shift-invert eigensolve cannot factor through) and is '
               'exactly where a fork is held: the one place that does not damp the voice, '
               'since the stem barely moves in it.',
               'built', setup=True),
    ], text=text)


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
        'Heat diffusing from a hot corner through a tetrahedral box: the same solve '
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
    # it shows: the pipeline demo, from an outline through to a stress. The smoke run
    # loosens the size cap and shortens the adaptive-refinement loop, which together are
    # where all of its cost is; refinement_iters/_budget aren't reachable through
    # max_area_fraction alone, since they're independent knobs on top of it.
    Demo('stress_concentration', demo_stress_concentration, section=SOLIDS,
         smoke_kwargs={'max_area_fraction': 0.05, 'refinement_iters': 3, 'refinement_budget': 200}),
    # Builds its own box: the only 3D domain, and the tet count is what sets the
    # cost, so the smoke run takes a coarser one.
    Demo('elastic_3d', demo_elastic_3d, section=SOLIDS, smoke_kwargs={'n': 5}),
    # Builds its own columns (several lengths for the slenderness sweep, plus the four
    # end conditions) so it takes no domain. The smoke run shrinks the mesh and the
    # sweep, which together are all of its cost (each case is a small eigensolve).
    Demo('buckling', demo_buckling, section=SOLIDS,
         smoke_kwargs={'n_length': 12, 'n_across': 4, 'n_modes': 2,
                       'sweep_lengths': (12.0, 18.0)}),
    # Builds its own fork from an outline, so it takes no domain. Its cost is the eigen-
    # solves (the main one plus the tuning-law sweep) and the animation frames, so the
    # smoke run coarsens the mesh, shortens the sweep, and takes only a few frames.
    Demo('modal', demo_modal, section=SOLIDS,
         smoke_kwargs={'n_across_tine': 3, 'min_angle': 25, 'n_modes': 4, 'n_shown': 3,
                       'sweep_lengths': (0.088, 0.125), 'n_frames': 6}),
    # 2:1, because the aspect ratio makes SIMP produce the truss it is known
    # for. The resolution is now set by what the filter needs rather than by what 40
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
    # sequence is the demo. The smoke run keeps the two coarsest: an order needs
    # two points, and the 81x81 solve is most of the cost.
    Demo('convergence', demo_convergence, section=ACCURACY,
         smoke_kwargs={'resolutions': (11, 21), 'elastic_resolutions': (9, 17),
              'step_counts': (16, 32)}),
    # Both build their own refinement sequences rather than taking a domain (the
    # sequence is the demo) so the smoke run keeps only the two coarsest.
    Demo('higher_order', demo_higher_order, section=ACCURACY,
         smoke_kwargs={'resolutions': (11, 21)}),
    Demo('quadrature_load', demo_quadrature_load, section=ACCURACY,
         smoke_kwargs={'resolutions': (11, 21)}),
]
