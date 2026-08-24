"""Solver demos. Run via the shared CLI:

    uv run python examples/cli.py list
    uv run python examples/cli.py run poisson
"""
import numpy as np
from functools import partial

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from fem.adaptivity import AdaptiveRefinement
from fem.backends import IterativeBackend
from fem.geometry import calculate_triangle_min_angle
from fem.numerics import bump_function
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import (
    ANNULUS_INNER, ANNULUS_OUTER, LOAD_MMS_FREQUENCY, ConvergenceStudy, create_annulus_mesh,
    elastic_convergence, load_comparison_convergence, oscillatory_exact, poisson_convergence,
    poisson_p2_convergence, solve_annulus_mms, solve_load_comparison, theta_convergence,
)
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.estimators import goal_oriented_estimator, recovery_estimator
from fem.forms import MaskedMassForm
from fem.space import FunctionSpace
from fem.regions import everywhere, on_plane, in_box, intersect, union
from fem.plot.plotter import Plotter
from fem.plot.helpers import plot_mesh
from fem.equations import Projection, Poisson, LinearElastic, StrainMeasure
from fem.solver import Solver
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.structured import create_box_mesh, create_rect_mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.problem import heat, wave
from fem.integrators import NewmarkMethod, ThetaMethod
from fem.topology import TopologyOptimizer
from fem.sensitivity import Compliance, PointValue, SensitivityAnalysis
from fem.design import DesignOptimizer, SIMPModel
from fem.topology import calculate_smoothing_matrix
from fem.energy_solver import EnergySolver
from fem.buckling import BucklingSolver
from fem.modal import ModalSolver

from demo_registry import Demo, DemoResult, Figure
from domains import (
    airfoil_channel_pslg, beam, column, heatsink_pslg, l_bracket_pslg, plate_with_hole_pslg,
    square, tuning_fork_pslg,
)

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

def demo_l2_projection(mesh, reference_resolution=120):
    """Project an oscillatory function onto a mesh's P1 space to show representation error:
    the resolution limit of the space itself, before any PDE is solved."""
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return [np.sin(40 * (x**2 + y**2))]

    solution = Solver(mesh, Projection(source=cool_f)).solve()

    # The target sampled on a fine mesh, so the left panel is the function itself rather
    # than another coarse approximation of it. The projection (right) is on the demo's own
    # mesh, coarse enough that the fast outer rings outrun what P1 can represent.
    fine = square(reference_resolution)
    xy = fine.vertices - np.array([0.5, 0.5])
    exact = np.sin(40 * (xy[:, 0]**2 + xy[:, 1]**2))

    plotter = Plotter(1, 2, title='L2 projection onto a P1 space')
    plotter.plot(fine, exact, mode='colored', idx=(0, 0), label='', clim=(-1, 1),
                 title='The target: sin(40 r^2)')
    plotter.plot(mesh, solution.u, mode='colored', idx=(0, 1), label='', clim=(-1, 1),
                 title=f'Projected onto P1 ({len(mesh.elements)} triangles)')
    return DemoResult([Figure(
        plotter,
        'Representation error, before any PDE: how well the P1 space can represent a '
        'function at all. Left: the target sin(40 r^2), whose rings tighten with radius. '
        'Right: its L2 projection onto a deliberately coarse mesh. The slow inner rings '
        'come through; past the radius where one ring spans only a couple of triangles the '
        'space can no longer follow it, and the outer rings break up into the mesh. This '
        'is the error floor every solver on this mesh starts from, and the rest of this '
        'section measures how fast it falls as the mesh is refined.')])

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

def demo_potential_flow(length=7.0, height=4.0, chord=3.0, angle_of_attack=12.0,
                        n_points=80, min_angle=20, max_area_fraction=0.0015):
    """Potential flow over a NACA airfoil: Laplace's equation for the velocity potential,
    with the wing a no-flux streamline the flow accelerates over. Solved with quadratic
    (P2) elements, so the recovered flow speed is smooth without a very fine mesh."""
    # An ideal (incompressible, irrotational) flow has a velocity potential phi with
    # v = grad(phi) and div(v) = 0, so phi solves Laplace's equation. The wing carries no
    # flow through it, which is exactly the natural (zero-flux) condition of the weak
    # form: say nothing on its surface and it becomes a streamline the flow parts around.
    # A potential difference across the channel drives the flow; the walls are no-flux too.
    pslg = airfoil_channel_pslg(length, height, chord, angle_of_attack, n_points=n_points)
    pslg.validate()
    mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                             max_area=max_area_fraction * pslg.area()).refine()

    equation = Poisson(source=0)   # Laplace: no sources in the flow
    bc = BoundaryConditions()
    # phi rises from inlet to outlet, so v = grad(phi) runs left to right. The wing and
    # the walls take no condition, which is the no-flux streamline that makes this a flow
    # *over* the wing rather than through it.
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)      # inlet (left)
    bc.add(BCType.DIRICHLET, on_plane(0, length), 1.0)   # outlet (right)

    solver = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement)
    solution = solver.solve()
    # v = grad(phi). The per-element gradient is constant on each triangle; nodal_flux
    # recovers a continuous per-node field from it, which the P2 tessellation then draws
    # smoothly, so the speed reads as a field rather than a mosaic of flat triangles.
    speed = np.linalg.norm(solution.nodal_flux(), axis=1)   # (n_nodes,)
    # Ideal flow with no Kutta condition predicts a near-singular velocity where the
    # airfoil edges are sharp, which would swamp the colour scale. Clip it to a high
    # percentile so the flow over the wing, the point of the figure, stays legible.
    cap = float(np.percentile(speed, 96))

    conditions = Plotter(panel_aspect=1.8)
    conditions.plot(mesh, mode='bc', bc=bc)

    # `space=solution.space` opts both panels onto the P2 tessellation: the potential
    # shows its within-element curvature and the recovered speed draws smoothly.
    plotter = Plotter(1, 2, title='Potential flow over an airfoil', panel_aspect=1.8)
    plotter.plot(mesh, solution.u, mode='colored', idx=(0, 0), label='velocity potential',
                 title='Potential and its equipotentials', contour=22, space=solution.space)
    plotter.plot(mesh, speed, mode='colored', idx=(0, 1), label='flow speed', clim=(0.0, cap),
                 title='Flow speed (clipped near the edges)', space=solution.space)
    return DemoResult([
        Figure(plotter,
               'Ideal (irrotational, incompressible) flow over a NACA 2412 airfoil at a '
               f'{angle_of_attack:g}-degree angle of attack, generated from the standard '
               'formula rather than a data file. Left: the velocity potential phi (Laplace) '
               'with its equipotentials, which crowd over the upper surface where the flow '
               'speeds up. Right: the flow speed, faster over the top than the bottom, with '
               'stagnation near the leading and trailing edges. The wing takes no condition '
               'at all: a free surface is the no-flux streamline of the weak form, so the '
               'flow parts around it. The speed is clipped near the sharp edges, where ideal '
               'flow with no Kutta condition predicts an unphysical velocity spike.',
               'flow'),
        Figure(conditions,
               'A potential difference across the channel (phi = 0 at the inlet, 1 at the '
               'outlet) drives the flow left to right; the walls and the wing surface take '
               'nothing, which is the no-flux condition that makes them streamlines.',
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


def _share_panel_limits(plotter, n_panels):
    """Give the panels in a row one shared view: the union of the x and y limits each set
    for its own shape, so they share a scale and their baselines and titles line up."""
    axes = [plotter.get_ax((0, c)) for c in range(n_panels)]
    xlo = min(a.get_xlim()[0] for a in axes)
    xhi = max(a.get_xlim()[1] for a in axes)
    ylo = min(a.get_ylim()[0] for a in axes)
    yhi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_xlim(xlo, xhi)
        a.set_ylim(ylo, yhi)


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
    """Compare P1 and P2 elements on the same manufactured Poisson problem: P2 is third
    order in L2 where P1 is second."""
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

def _annulus_area_study(element_type, resolutions):
    """Domain-area error vs h for the annulus, a pure measure of boundary fidelity.

    The area the elements integrate over is the polygon for straight facets and the true
    curved annulus for isoparametric ones, so `space.geometry.total_volume` minus the
    exact area isolates the geometry error with no solve involved.
    """
    true_area = np.pi * (ANNULUS_OUTER**2 - ANNULUS_INNER**2)
    steps, errors = [], []
    for n in sorted(resolutions):
        mesh = create_annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, n, 4 * n)
        space = FunctionSpace(mesh, element_type, n_components=1)
        errors.append(abs(space.geometry.total_volume - true_area))
        steps.append(1.0 / (n - 1))
    return ConvergenceStudy(np.array(steps), np.array(errors))


def demo_curved_elements(coarse_n=4, resolutions=(3, 5, 9, 17)):
    """Show what curved (isoparametric) elements buy on a curved domain: the boundary
    follows the true circle instead of a polygon, so the domain area (pure geometry)
    converges at the element's own order instead of the polygonal O(h^2)."""
    # A deliberately coarse annulus for the two picture panels, so the straight facets are
    # obvious. Both solve the same manufactured Poisson problem (u = sin(x) sin(y)); only
    # the element's geometry differs, straight P2 vs isoparametric P2.
    coarse = create_annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, coarse_n, 4 * coarse_n)
    straight = solve_annulus_mms(coarse_n, QuadraticTriangleElement)
    curved = solve_annulus_mms(coarse_n, IsoparametricTriangleElement)
    sp_straight = FunctionSpace(coarse, QuadraticTriangleElement, n_components=1)
    sp_curved = FunctionSpace(coarse, IsoparametricTriangleElement, n_components=1)

    # One colour scale across both panels so the fields are read side by side; the
    # difference is meant to be the boundary, not the normalisation.
    clim = (float(min(straight.u.min(), curved.u.min())),
            float(max(straight.u.max(), curved.u.max())))

    figure = Plotter(1, 3, figsize=(14.0, 4.2),
                     title='Curved elements follow the true boundary')
    # Both fields are drawn on a sub-triangulation of each P2 element (a display
    # tessellation, added below in `space=`), so the quadratic field shows faithfully and
    # the only visible difference is the rim: a polygon for straight facets, the true
    # circle for the curved map. A light wireframe over each makes the elements explicit.
    figure.plot(coarse, straight.u, mode='colored', idx=(0, 0), space=sp_straight,
                clim=clim, colorbar=False, title='Straight P2: the rim is a polygon')
    plot_mesh(figure.get_ax((0, 0)), coarse, color='0.6', linewidth=0.4)

    figure.plot(coarse, curved.u, mode='colored', idx=(0, 1), space=sp_curved,
                clim=clim, label='u', title='Isoparametric P2: the rim is the true circle')
    plot_mesh(figure.get_ax((0, 1)), coarse, color='0.6', linewidth=0.4, space=sp_curved)

    # The geometry behind the picture: the polygon's area is wrong by O(h^2), the curved
    # rim's by the element's own O(h^3). This is where straight and curved separate
    # cleanly and honestly. On this smooth Dirichlet solve the solution's own L2 error
    # does not floor visibly (the domain-perturbation error stays subdominant here), so
    # the panel measures the geometry directly rather than claiming a rate gap that this
    # problem does not show.
    straight_area = _annulus_area_study(QuadraticTriangleElement, resolutions)
    curved_area = _annulus_area_study(IsoparametricTriangleElement, resolutions)
    rate = figure.chart_ax(idx=(0, 2), xlabel='h', ylabel='domain area error')
    _plot_study(rate, straight_area, 'straight P2', 'tab:blue', 2, 'h')
    _plot_study(rate, curved_area, 'isoparametric P2', 'tab:orange', 3, 'h')
    rate.set_title('Area error: O(h^2) vs O(h^3)')
    _tidy_log_axis(rate, straight_area.step)

    rows = ['             area-error order   expected']
    for name, study, expected in (('straight P2', straight_area, 2),
                                  ('isoparametric', curved_area, 3)):
        rows.append(f'{name:<18}{study.fitted_order:>9.2f}{expected:>11}')

    return DemoResult(
        [Figure(figure,
                'The same manufactured Poisson solve on an annulus, straight versus curved '
                'elements. Left: straight P2 approximates each rim by a chord, so the domain '
                'is a polygon. Middle: isoparametric P2 places its boundary nodes on the true '
                'circle and integrates over the curved element, so the rim is round. Both '
                'fields are drawn on a sub-triangulation of each P2 element, a display '
                'tessellation that shows the quadratic field faithfully and adds nothing to '
                'the solve. Right: the geometry behind the picture. The area the straight '
                'elements integrate over is a polygon, wrong by O(h^2); the curved elements '
                'integrate the true annulus, area right to O(h^3) and orders of magnitude '
                'closer at every mesh.')],
        text='\n'.join(rows),
    )


def demo_quadrature_load(resolutions=(11, 21, 41, 81)):
    """Show what sampling the load at the quadrature points buys, against reading the
    source only at the vertices."""
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

def _finite_plate_kt(hole_over_width: float) -> float:
    """Stress concentration at a circular hole in a finite-width plate under tension,
    on the gross (applied) stress: Howland's result, in Peterson's polynomial fit for
    the net-section factor divided by the net fraction. Reads 3 for a vanishing hole."""
    r = hole_over_width
    net = 3.000 - 3.140 * r + 3.667 * r**2 - 1.527 * r**3
    return net / (1.0 - r)


def demo_stress_concentration(traction=1.0, length=6.0, height=3.0, radius=0.15,
                              min_angle=25, max_area_fraction=0.01, circle_segments=16,
                              refinement_iters=20, refinement_budget=8000):
    """Take a plate with a hole from an outline through meshing, boundary conditions and
    adaptive refinement to the stress concentration at its rim, measured against
    Kirsch's factor of 3 and the finite-width value it approaches."""
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
    # The hole is only a coarse 16-gon here, and that is enough: `plate_with_hole_pslg`
    # tags the hole loop with a `Circle`, so Ruppert's split points and the adaptive
    # red-green refinement below project onto the true rim rather than subdividing chords,
    # and the isoparametric element's edge nodes sit on the circle too. The hole gets
    # rounder as the mesh gets finer, instead of freezing into whatever polygon the
    # initial sampling drew.
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

    # Solved on the curved quadratic element, and adaptively refined by the recovery
    # estimator (recovery_estimator, from fem.estimators), which reads the curved rim's
    # stress correctly. The loop replaces the uniform mesh above with one built by
    # repeatedly re-solving and splitting wherever the estimator finds the most error:
    # everything the rest of this demo plots and measures is read off the result of
    # this loop, not off the coarse mesh it started from. Because the rim splits
    # project onto the true circle, more refinement keeps rounding the hole rather
    # than subdividing a fixed polygon, so the budget is the only ceiling.
    equation = LinearElastic(E=200, nu=0.3)
    solver = Solver(mesh, equation, bc, element_type=IsoparametricTriangleElement)
    solution = AdaptiveRefinement(
        solver, recovery_estimator(equation),
        max_triangles=n_initial + refinement_budget, max_iters=refinement_iters,
    ).run()
    mesh = solver.mesh
    # The stress at the nodes, each element evaluated at its own nodes and averaged
    # where they meet. A P2 stress varies within the element, and this reads it on the
    # rim itself rather than at an interior sample point near it.
    nodes = solution.space.node_coords
    sigma_xx = solution.nodal_stress()[:, 0, 0]

    # A vertical strip through the hole's centre: the line the concentration decays
    # along, from the rim out to the far field. The geometry is known here rather than
    # measured back off the mesh, which building the domain in the demo buys. The rim
    # crossings are nodes of the mesh (the 16-gon has a vertex at the top and bottom of
    # the hole, and refinement keeps it), so the peak is the value at those two nodes.
    strip = np.abs(nodes[:, 0] - length/2) < 0.25*radius
    order = np.argsort(nodes[strip, 1])
    y_strip, ratio_strip = nodes[strip, 1][order], (sigma_xx[strip] / traction)[order]
    on_rim = np.isclose(nodes[:, 0], length/2) & np.isclose(np.abs(nodes[:, 1] - height/2), radius)
    peak = float(sigma_xx[on_rim].max() / traction)

    # Kirsch's factor of 3 is the infinite-plate limit, and this plate is finite: the
    # hole removes section, which raises the stress the remaining material carries, so
    # the exact answer sits a little above 3 (Howland's finite-width value, below).
    # The rim reading lands within about a percent of it, and further refinement moves
    # it by less than the last digit shown.
    hole_over_width = 2*radius / height
    finite_kt = _finite_plate_kt(hole_over_width)

    # One figure, three plots: the whole pipeline in a row. The first plot folds the mesh,
    # its input outline, and the conditions together, since a 2:1 plate at this zoom has
    # room for all three; the second is the stress it drives; the third is the chart. The
    # plate panels sit at 2:1 and the chart takes the third cell at whatever shape is left.
    figure = Plotter(1, 3, figsize=(14.0, 3.6),
                     title='From an outline to a stress concentration')
    figure.plot(mesh, mode='bc', bc=bc, idx=(0, 0),
                title=f'{len(mesh.elements)} triangles (refined from {n_initial}), '
                      'with conditions')
    ax0 = figure.get_ax((0, 0))
    # The triangulation under the conditions, so the panel shows the mesh the solve ran
    # on (and how finely refinement graded the rim) rather than only the outline. Thin
    # and grey, below the glyphs, which keep their own zorder and still read over it.
    ax0.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements,
                color='0.55', linewidth=0.2, zorder=1.5)
    # The input segments over the triangulation: which of the outline the mesher kept and
    # which it split. Fixed PSLG data, so it overlays the refined mesh as validly as the
    # coarse one it started from.
    ax0.add_collection(LineCollection(
        rupperts.vertices[rupperts.segments], colors='blue', linewidths=1.0, zorder=2.0))
    # Passing the solution draws the P2 field on its own tessellation, with the rim
    # following the true circle, rather than flattening it to one value per triangle.
    figure.plot(solution, sigma_xx, mode='colored', idx=(0, 1), label='sigma_xx',
                title='Stress concentration (sigma_xx)')
    ax = figure.chart_ax(idx=(0, 2), xlabel='y', ylabel='sigma_xx / applied')
    # Drawn as two runs, below the hole and above it. One run joins them straight
    # across the gap, which reads as a stress the hole does not have.
    below = y_strip < height/2
    ax.plot(y_strip[below], ratio_strip[below], 'o-', color='tab:blue', markersize=2,
            label='through the hole centre')
    ax.plot(y_strip[~below], ratio_strip[~below], 'o-', color='tab:blue', markersize=2)
    ax.axhline(finite_kt, color='tab:red', linestyle='--',
               label=f'finite plate (Howland): {finite_kt:.2f}x')
    ax.axhline(3.0, color='tab:red', linestyle=':', label='infinite plate (Kirsch): 3x')
    ax.axhline(1.0, color='gray', linestyle=':', label='far field')
    ax.set_title(f'Peak {peak:.2f}x the applied stress')
    ax.grid(alpha=0.3)
    # Beside the peak, over the flat far field, where it hides nothing.
    ax.legend(loc='center left', fontsize='small')

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
        [Figure(figure,
                f'The whole pipeline in one row. Left: the mesh after adaptive refinement, '
                f'grown from {n_initial} triangles to {len(mesh.elements)} by putting '
                f'elements where the recovery estimator found the most error, with the '
                f'input outline in blue and the conditions drawn on it; the rim and long '
                f'edges carry none, which is not an omission but the natural (traction-'
                f'free) condition of the weak form. Middle: the stress sigma_xx on curved '
                f'quadratic elements, crowding into the material either side of the hole '
                f'and relaxing to the applied value within about a diameter. Right: that '
                f'stress along a strip through the hole centre, read at the nodes, peaking '
                f'at {peak:.2f}x the applied value at the rim. The classic Kirsch factor of '
                f'3 is for a hole in an infinite plate; this plate is finite, and '
                f"Howland's value for a hole {hole_over_width:.2f} of its width is "
                f'{finite_kt:.2f}.')],
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
              f'hole diameter / height   {hole_over_width:.2f}\n'
              f'peak sigma_xx / applied  {peak:.2f}   '
              f'(Howland, finite plate: {finite_kt:.2f}; Kirsch, infinite plate: 3)'),
    )

def demo_bracket(arm=4.0, width=1.2, fillet_radius=0.25, traction=0.4, E=300.0, nu=0.3,
                 min_angle=28, max_area_fraction=0.0015, n_rounds=18, refine_fraction=0.9):
    """Load an L-bracket and read the stress at its inner corner: a sharp re-entrant
    corner is a stress singularity whose peak climbs without bound as the mesh refines,
    while a fillet gives a finite, converged value. This is why real parts round their
    inner corners.

    Solved on quadratic (P2) elements: the sharp bracket on straight `QuadraticTriangleElement`,
    the filleted one on `IsoparametricTriangleElement` so the arc is a true circle rather than
    a polygon. The recovery estimator drives refinement (it reads the curved fillet's flux
    correctly), and the peak is read from the recovered nodal von Mises."""
    # The re-entrant corner is where the two limbs meet. There the exact elastic stress
    # is genuinely infinite (it grows like r^(-0.46) into the corner), so no mesh
    # resolves it: refine and the computed peak just keeps climbing. Rounding the corner
    # with a fillet removes the singularity, and the peak settles on a real number. The
    # demo shows both halves: the fields side by side, and the corner peak against mesh
    # size for each, one curve climbing and one levelling off.
    equation = LinearElastic(E, nu)
    corner = np.array([width, width])

    def make_bc():
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, arm), [0, 0])        # clamp the top of the upright limb
        bc.add(BCType.NEUMANN, on_plane(0, arm), [0, -traction])  # pull the horizontal tip down
        return bc

    def corner_peak(solution):
        # The von Mises peak near the inner corner alone, kept clear of the clamp's own
        # concentration at the far top so the comparison is about the corner. Read from the
        # L2-recovered nodal field (recover-then-reduce), the same smooth field the panels
        # draw, so the tracked peak and the plotted colour agree.
        space = solution.space
        nodal_vm = solution.nodal_von_mises(method='l2')
        near = np.linalg.norm(space.node_coords - corner, axis=1) < 0.8 * width
        return float(nodal_vm[near].max())

    def refine_and_track(fillet, element_type):
        """Adaptively refine one bracket, recording the corner peak each round.

        The refinement loop is `AdaptiveRefinement`'s, unrolled here so the corner peak
        can be read off every intermediate mesh rather than only the last: the sequence
        of peaks is the point, not just the final field. `element_type` is the straight
        quadratic triangle for the sharp corner and the isoparametric one for the fillet,
        so the arc stays a true circle through refinement.
        """
        pslg = l_bracket_pslg(arm, width, fillet_radius=fillet, n_fillet=20)
        pslg.validate()
        mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area()).refine()
        solver = Solver(mesh, equation, make_bc(), element_type=element_type)
        refiner = RedGreenRefiner(solver.mesh)
        estimator = recovery_estimator(equation)
        solution = solver.solve()
        sizes, peaks = [], []
        for _ in range(n_rounds):
            sizes.append(len(solver.mesh.elements))
            peaks.append(corner_peak(solution))
            residuals = estimator.estimate(solver)
            refine_idxs = np.flatnonzero(residuals >= refine_fraction * residuals.max())
            solver.remesh(refiner.refine([int(i) for i in refine_idxs]))
            solution = solver.solve()
        sizes.append(len(solver.mesh.elements))
        peaks.append(corner_peak(solution))
        return solver.mesh, solution, np.array(sizes), np.array(peaks)

    sharp_mesh, sharp, sharp_sizes, sharp_peaks = refine_and_track(0.0, QuadraticTriangleElement)
    round_mesh, rounded, round_sizes, round_peaks = refine_and_track(
        fillet_radius, IsoparametricTriangleElement)

    conditions = Plotter(panel_aspect=1.0)
    conditions.plot(sharp_mesh, mode='bc', bc=make_bc())

    # Independent colour scales: the sharp peak dwarfs the filleted one, so a shared scale
    # would wash the fillet's own concentration to a single flat colour. The titles carry
    # the numbers the comparison rests on.
    fields = Plotter(1, 2, title='An L-bracket under a tip load')
    for i, (name, mesh, solution, peaks) in enumerate((
            ('Sharp corner', sharp_mesh, sharp, sharp_peaks),
            (f'Fillet r = {fillet_radius:g}', round_mesh, rounded, round_peaks))):
        # P2-aware render: passing the solution pulls its space (the tessellation), the
        # L2-recovered nodal von Mises is the smooth field, and warp=True draws it on the
        # deformed shape.
        fields.plot(solution, solution.nodal_von_mises(method='l2'), mode='colored',
                    idx=(0, i), warp=True, label='von Mises stress',
                    title=f'{name}\n{len(mesh.elements)} elements, corner peak {peaks[-1]:.0f}')
        # Clamp and tip-load glyphs read off the undeformed mesh, drawn at the deformed
        # vertex positions so the load follows the tip it pulls on.
        deformed_vertices = mesh.vertices + solution.u.reshape(-1, 2)[:len(mesh.vertices)]
        fields.overlay_supports(mesh, make_bc(), idx=(0, i), coords=deformed_vertices)

    sweep = Plotter(1, 1, title='The corner peak against mesh refinement')
    ax = sweep.chart_ax(xlabel='elements', ylabel='von Mises at the inner corner')
    ax.semilogx(sharp_sizes, sharp_peaks, 'o-', color='tab:red', label='sharp (singular)')
    ax.semilogx(round_sizes, round_peaks, 'o-', color='tab:blue',
                label=f'fillet r = {fillet_radius:g} (converges)')
    ax.set_title('Sharp corner keeps climbing; the fillet settles')
    # The element counts span well under a decade, where a log axis crowds itself with
    # minor labels (1.1x10^3, 1.2x10^3, ...); label a few round values across the range.
    ax.grid(True, which='both', alpha=0.3)
    sizes_all = np.concatenate([sharp_sizes, round_sizes])
    mantissas = np.array([1, 1.2, 1.5, 1.8, 2, 2.5, 3, 4, 5, 7])
    nice = np.concatenate([mantissas * 10**k for k in (2, 3, 4)])
    ticks = nice[(nice >= sizes_all.min()) & (nice <= sizes_all.max())]
    if len(ticks) >= 2:
        _tidy_log_axis(ax, ticks)
    ax.legend()

    reduction = 100 * (1 - round_peaks[-1] / sharp_peaks[-1])
    return DemoResult([
        Figure(fields,
               'The same bracket, clamped at the top and pulled down at the horizontal '
               'tip, with a sharp inner corner (left) and a filleted one (right). Stress '
               'crowds into the re-entrant corner; the fillet spreads it over a radius and '
               f'cuts the peak by about {reduction:.0f}% at this resolution. The colour '
               'scales are independent, since the sharp peak would otherwise flatten the '
               "fillet's own concentration to nothing.",
               'fields', thumbnail=True),
        Figure(sweep,
               'The corner von Mises peak as the mesh is adaptively refined into the '
               'corner. The sharp corner is a true stress singularity, so its peak climbs '
               'without bound and never converges: the "stress" there is a property of the '
               'mesh, not of the part. The fillet removes the singularity, and its peak '
               'settles on a finite value. This is the whole reason real parts round their '
               'inner corners.',
               'singularity'),
        Figure(conditions,
               'The upright limb is clamped at the top; a downward traction pulls the '
               'horizontal tip. Everything else, the inner corner included, is '
               'traction-free.',
               'conditions', setup=True),
    ], text=(f'corner von Mises peak (sharp)   {sharp_peaks[-1]:.1f}  '
             f'over {sharp_sizes[-1]} elements, still climbing\n'
             f'corner von Mises peak (fillet)  {round_peaks[-1]:.1f}  '
             f'over {round_sizes[-1]} elements, converged\n'
             f'reduction from the fillet       {reduction:.0f}%'))


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

def _heatsink_film(mesh):
    """The convective film: every boundary but the heated bottom edge (the surfaces above
    the base, plus the base's two sides down to the corners)."""
    w = float(np.max(mesh.vertices[:, 0]))
    return union(in_box([None, 1e-6], [None, None]), on_plane(0, 0.0), on_plane(0, w))


def _heatsink_bc(mesh, add_base, kappa, u_ambient):
    """The boundary spec: `add_base(bc)` on the bottom edge, a Robin film everywhere else."""
    bc = BoundaryConditions()
    add_base(bc)
    bc.add_robin(_heatsink_film(mesh), kappa=kappa, g=kappa * u_ambient)
    return bc


def _steady_heatsink(mesh, bc, kappa, u_ambient):
    """Steady heat field for `bc` (a base condition plus a Robin film).

    Returns (u, heat_shed), where heat_shed is the convective loss through the film,
    kappa * integral_film (u - u_ambient). At steady state that equals the heat entering
    the base, so it is the sink's dissipation.
    """
    solver = Solver(mesh, Poisson(source=0), bc)
    u = solver.solve().u
    # The convective loss, read off the same region-restricted boundary mass a Robin
    # condition assembles, so it is the exact discrete integral of (u - u_ambient).
    resolved = bc.resolve(solver.space.nodes, 1)
    film_mass = solver.space.assemble(
        MaskedMassForm(1, resolved.robin[0].facet_mask), boundary=True)
    heat_shed = kappa * float(np.asarray(film_mass @ (u - u_ambient)).sum())
    return u, heat_shed


def _solid_block(width, height, target_area):
    """A structured mesh of a solid `width` x `height` block, at roughly `target_area` per
    element so it matches a Ruppert's mesh built to the same cap."""
    nx = max(2, round(width / np.sqrt(target_area)))
    ny = max(2, round(height / np.sqrt(target_area)))
    return create_rect_mesh(corners=[[0.0, 0.0], [width, height]], resolution=(nx, ny))


def _mesh_area(mesh):
    """Total area of a triangle mesh: its material, per unit depth."""
    tri = np.asarray(mesh.vertices)[np.asarray(mesh.elements)]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    return float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() / 2)


def _fin_efficiency(kappa, u_ambient, u_hot, thickness, lengths):
    """Fin efficiency for a single straight fin at each length, computed and from theory.

    Efficiency is the heat a fin sheds over what it would shed with all of it at the base
    temperature: eta = shed / (kappa * A_fin * (u_hot - u_ambient)), A_fin = 2L + t the
    convecting surface (two sides and the tip, per unit depth). Beam theory gives
    eta = tanh(m*Lc)/(m*Lc), with m = sqrt(2*kappa/(k*t)) and the corrected length
    Lc = L + t/2 standing in for the convecting tip.
    """
    def add_hot(bc):
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), u_hot)

    m = np.sqrt(2 * kappa / thickness)      # conductivity k = 1
    eta_fem, eta_theory = [], []
    for length in lengths:
        ny = max(10, round(10 * length / thickness))    # ~10 elements across the thickness
        fin = create_rect_mesh(corners=[[0.0, 0.0], [thickness, length]], resolution=(10, ny))
        _, shed = _steady_heatsink(fin, _heatsink_bc(fin, add_hot, kappa, u_ambient),
                                   kappa, u_ambient)
        area = 2 * length + thickness
        eta_fem.append(shed / (kappa * area * (u_hot - u_ambient)))
        lc = length + thickness / 2
        eta_theory.append(float(np.tanh(m * lc) / (m * lc)))
    return np.array(lengths), np.array(eta_fem), np.array(eta_theory)


def _mark_base(ax, width, kind):
    """Draw the base condition just below the domain, off the coloured field so it does not
    clash with the warm colormap: upward arrows for a Neumann heat flux, a bar for a held
    Dirichlet temperature. The Robin film (every other surface) is left to the legend."""
    y0 = -0.22
    if kind == 'flux':
        xs = np.linspace(0.1 * width, 0.9 * width, 8)
        ax.quiver(xs, np.full_like(xs, y0), np.zeros_like(xs), np.ones_like(xs),
                  color='red', angles='xy', scale_units='xy', scale=1 / 0.17, width=0.010,
                  headwidth=4, headlength=5, clip_on=False, zorder=6)
    else:
        ax.plot([0.05 * width, 0.95 * width], [y0, y0], color='tab:blue', lw=4,
                solid_capstyle='round', clip_on=False, zorder=6)
    ax.set_ylim(bottom=y0 - 0.12)


def demo_heat_equation(dt=0.06, steps=30, kappa=0.3, u_ambient=300.0, u_hot=400.0,
                       flux=40.0, fin_lengths=(0.4, 0.8, 1.4, 2.0, 2.8),
                       min_angle=28, max_area_fraction=0.0004):
    """A finned heatsink: warm it from a cold start (the transient heat equation), then
    measure how much better it dissipates than a solid block, and check the fins against
    beam theory."""
    # The heat equation is Poisson's operator integrated in time (see fem.problem.heat).
    # The shape earns its keep: a square plate has nowhere for the heat to go, so only its
    # contrast fades; a heatsink conducts heat up its fins and sheds it, which is a shape
    # worth measuring. The mesh is built here because it is part of what the demo says.
    pslg = heatsink_pslg()
    pslg.validate()
    target_area = max_area_fraction * pslg.area()
    mesh = RuppertsAlgorithm(pslg, min_angle=min_angle, max_area=target_area).refine()
    width = float(np.max(mesh.vertices[:, 0]))
    height = float(np.max(mesh.vertices[:, 1]))
    # The naive baseline: a solid block of the same bounding box. The fins carve channels
    # out of it, trading metal for surface area.
    block = _solid_block(width, height, target_area)
    metal_ratio = _mesh_area(mesh) / _mesh_area(block)

    # -- the transient: warm the sink from a cold start --------------------------------
    # The bottom face is held hot (a chip beneath the base); every other surface is a
    # convective film, du/dn + kappa*(u - u_ambient) = 0, so a fin sheds heat and cools
    # toward its tip. A cold start at ambient makes the run a warm-up: the base energizes
    # at the first step and the front climbs the fins to a steady gradient.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), u_hot)
    bc.add_robin(_heatsink_film(mesh), kappa=kappa, g=kappa * u_ambient)
    u_initial = np.full(len(mesh.vertices), u_ambient)
    solution = ThetaMethod(dt=dt, steps=steps).run(heat(mesh, bc=bc), u_initial)
    u_values, t_values = solution.u, solution.t

    # -- effectiveness: block vs finned, posed two ways --------------------------------
    # Fixed power: the same heat flux into each base (a chip of fixed wattage); compare the
    # base temperature. Fixed temperature: each base held hot; compare the heat shed. The
    # thermal resistance R = (base rise)/power is the shape's property either way.
    def add_flux(bc):
        bc.add(BCType.NEUMANN, on_plane(1, 0.0), [flux])

    def add_hot(bc):
        bc.add(BCType.DIRICHLET, on_plane(1, 0.0), u_hot)

    bc_block_p = _heatsink_bc(block, add_flux, kappa, u_ambient)
    bc_fin_p = _heatsink_bc(mesh, add_flux, kappa, u_ambient)
    bc_block_t = _heatsink_bc(block, add_hot, kappa, u_ambient)
    bc_fin_t = _heatsink_bc(mesh, add_hot, kappa, u_ambient)

    power = flux * width
    u_block_p, _ = _steady_heatsink(block, bc_block_p, kappa, u_ambient)
    u_fin_p, _ = _steady_heatsink(mesh, bc_fin_p, kappa, u_ambient)
    r_block = (float(u_block_p.max()) - u_ambient) / power
    r_fin = (float(u_fin_p.max()) - u_ambient) / power

    u_block_t, q_block = _steady_heatsink(block, bc_block_t, kappa, u_ambient)
    u_fin_t, q_fin = _steady_heatsink(mesh, bc_fin_t, kappa, u_ambient)
    effectiveness = q_fin / q_block

    # One colour scale across all four (ambient to the block's fixed-power peak), so the
    # panels compare directly, one shared bar per row on the right.
    clim = (u_ambient, max(float(u_block_p.max()), u_hot))
    comparison = Plotter(2, 2, panel_aspect=1.6, axis_labels=False, figsize=(10.5, 7.2),
                         title='Heatsink vs a solid block of the same size')
    comparison.plot(block, u_block_p, mode='colored', idx=(0, 0), cmap='inferno', clim=clim,
                    colorbar=False,
                    title=f'Same power in: solid block\nbase +{u_block_p.max()-u_ambient:.0f} C  '
                          f'(R = {r_block:.2f})')
    comparison.plot(mesh, u_fin_p, mode='colored', idx=(0, 1), cmap='inferno', clim=clim,
                    label='temperature',
                    title=f'Same power in: finned\nbase +{u_fin_p.max()-u_ambient:.0f} C  '
                          f'(R = {r_fin:.2f})')
    comparison.plot(block, u_block_t, mode='colored', idx=(1, 0), cmap='inferno', clim=clim,
                    colorbar=False,
                    title=f'Base held at {u_hot:.0f}: solid block\nsheds Q = {q_block:.0f}')
    comparison.plot(mesh, u_fin_t, mode='colored', idx=(1, 1), cmap='inferno', clim=clim,
                    label='temperature',
                    title=f'Base held at {u_hot:.0f}: finned\nsheds Q = {q_fin:.0f}  '
                          f'({effectiveness:.1f}x on {metal_ratio:.2f}x the metal)')
    # Mark only the base, and off the coloured field so nothing clashes with the warm
    # colormap: upward arrows below the base for the Neumann heat flux (fixed-power row), a
    # bar for the held Dirichlet base (fixed-temperature row). The Robin film is every
    # other surface, named in the legend rather than traced over the fins.
    for idx, kind in (((0, 0), 'flux'), ((0, 1), 'flux'), ((1, 0), 'held'), ((1, 1), 'held')):
        _mark_base(comparison.get_ax(idx), width, kind)
        comparison.get_ax(idx).tick_params(left=False, bottom=False,
                                           labelleft=False, labelbottom=False)
    comparison.fig.legend(handles=[
        Line2D([], [], color='red', marker='^', linestyle='', markersize=9,
               label='Neumann: heat flux into the base'),
        Line2D([], [], color='tab:blue', lw=4, label='Dirichlet: base held hot'),
        Line2D([], [], color='tab:orange', lw=3, label='Robin: film on all other surfaces'),
    ], loc='outside lower center', ncol=3, frameon=False, fontsize='small')

    # -- validation: fin efficiency against beam theory --------------------------------
    # thickness matches the sink's own fins (heatsink_pslg's fin_width default).
    lengths, eta_fem, eta_theory = _fin_efficiency(
        kappa, u_ambient, u_hot, thickness=0.22, lengths=fin_lengths)
    m_fin = np.sqrt(2 * kappa / 0.22)
    efficiency = Plotter(1, 1, title='Fin efficiency against beam theory')
    ax = efficiency.chart_ax(xlabel='fin length L', ylabel='fin efficiency (heat shed / ideal)')
    dense = np.linspace(min(lengths), max(lengths), 100)
    dense_lc = dense + 0.22 / 2
    ax.plot(dense, np.tanh(m_fin * dense_lc) / (m_fin * dense_lc), '-', color='tab:red',
            alpha=0.6, label='theory  tanh(mL)/mL')
    ax.plot(lengths, eta_fem, 'o', color='tab:blue', label='computed')
    ax.axvline(1.4, color='0.6', ls=':', label="this sink's fins (L = 1.4)")
    ax.set_title('Longer fins shed more, but run less efficiently')
    ax.grid(alpha=0.3)
    ax.legend()

    # -- the transient figures ---------------------------------------------------------
    # One scale across the row (ambient to the heated base), a warm colormap for a warming
    # shape, and one shared bar on the last panel.
    n_shown = 4
    snapshots = Plotter(1, n_shown, panel_aspect=1.6,
                        title='Heatsink warming: heat climbing the fins')
    for panel, i in enumerate(np.linspace(0, len(u_values) - 1, n_shown).astype(int)):
        snapshots.plot(mesh, u_values[i], mode='colored', idx=(0, panel),
                       label='temperature', cmap='inferno', clim=(u_ambient, u_hot),
                       colorbar=(panel == n_shown - 1), title=f't={t_values[i]:.2f}')

    animation = Plotter(1, 1, title='Heatsink warming up')
    animation.plot_animation(mesh, u_values, mode='colored', label='temperature',
                             cmap='inferno', cbar_lims=(u_ambient, u_hot),
                             titles=[f't={t:.2f}' for t in t_values], idx=(0, 0))

    setup = Plotter(1, 2, title='How the heatsink is posed')
    setup.plot(mesh, mode='bc', bc=bc, title='Boundary conditions', idx=(0, 0))
    setup.plot(mesh, u_initial, mode='colored', idx=(0, 1), label='temperature',
               cmap='inferno', clim=(u_ambient, u_hot),
               title=f'Initial condition u(x, 0) = {u_ambient:.0f}')

    tip = float(u_values[-1].min())
    eta_here = float(eta_fem[np.argmin(np.abs(lengths - 1.4))])
    return DemoResult([
        Figure(comparison,
               'The finned sink against a solid block of the same bounding box, posed two '
               'ways. Top, the same heat flux into each base (a chip of fixed power): the '
               f'block runs {u_block_p.max()-u_ambient:.0f} C above ambient, the finned sink '
               f'only {u_fin_p.max()-u_ambient:.0f} C, roughly halving the thermal resistance '
               f'(R {r_block:.2f} -> {r_fin:.2f}). Bottom, each base held at {u_hot:.0f}: the '
               f'finned sink sheds {effectiveness:.1f}x the heat, on {metal_ratio:.2f}x the '
               'metal, because the fins trade material for surface area. This is why '
               'heatsinks have fins instead of being solid.',
               'comparison', thumbnail=True),
        Figure(snapshots,
               'The same finned sink warming from a cold start, the transient heat equation '
               'stepped in time. The base is held hot underneath and the fins shed to '
               'ambient through a Robin film, so the warming front climbs each fin and '
               f'settles into the fin gradient, hot at the root and about {tip:.0f} at the tips.',
               'snapshots'),
        Figure(efficiency,
               'Fin efficiency, the heat a fin sheds over what it would shed with all of it '
               'at the base temperature, against the beam-theory law tanh(mL)/(mL). The '
               'computed fins track it closely. Efficiency falls as fins lengthen, because a '
               "long fin runs cold toward the tip and carries less of its share: this sink's "
               f'fins (L = 1.4) sit near {eta_here:.0%}, trading efficiency for the extra '
               'surface area that does the cooling.',
               'efficiency'),
        Figure(animation, 'Crank-Nicolson warming of the heatsink, base to fin tips.',
               'animation'),
        Figure(setup,
               'The bottom face is held at a fixed hot temperature (a chip beneath the '
               'base); every other surface carries a Robin film, du/dn + kappa*(u - '
               'u_ambient) = 0, shedding heat to ambient. The sink starts cold at ambient, '
               'so the transient is a warm-up to the steady dissipating state.',
               'conditions', setup=True),
    ], text=(f'thermal resistance R (base rise per unit power):\n'
             f'  solid block   {r_block:.3f}\n'
             f'  finned sink   {r_fin:.3f}   ({r_block/r_fin:.1f}x lower)\n'
             f'heat shed with the base held {u_hot:.0f} (ambient {u_ambient:.0f}):\n'
             f'  solid block   {q_block:.1f}\n'
             f'  finned sink   {q_fin:.1f}   ({effectiveness:.1f}x, on {metal_ratio:.2f}x the metal)\n'
             f'fin efficiency at L = 1.4:  {eta_here:.2f}  (beam theory close)'))

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

def demo_linear_elastic(mesh, n_3d=14):
    """Solve linear elasticity for a cantilever in 2D and again in 3D, then read four
    rotation-invariant stress measures off the 2D solve."""
    E, nu = 200.0, 0.4

    # -- 2D: clamped on the left, pulled down over the middle of the right edge ---------
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    # Transverse, so the beam bends: an axial pull is much the same solve on any domain,
    # where a tip load is what makes a cantilever one. Sized for a tip deflection near 9%
    # of the span, well inside the small-strain regime the solver assumes.
    bc.add(BCType.NEUMANN,
           intersect(on_plane(0, w), in_box([None, 0.2], [None, 0.8])),
           [0, -0.5])
    solution = Solver(mesh, LinearElastic(E, nu), bc).solve()
    deformed = solution.deformed_mesh()

    # -- 3D: the same clamp-and-load, one dimension up ---------------------------------
    # The same assembly and element hierarchy, Solver reading the tetrahedron off the
    # connectivity. An AMG-preconditioned CG solve rather than a direct factorization,
    # where a 3D elastic system's fill-in starts to hurt. Only the boundary surface is
    # drawn: there are several times more tets than surface triangles.
    box = create_box_mesh(corners=[[0, 0, 0], [4, 1, 1]],
                          resolution=(4 * n_3d // 2, n_3d // 2, n_3d // 2))
    bc_3d = BoundaryConditions()
    bc_3d.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0, 0])
    bc_3d.add(BCType.NEUMANN, on_plane(0, 4.0), [0, 0, -0.5])
    solution_3d = Solver(box, LinearElastic(E, nu), bc_3d, backend=IterativeBackend()).solve()
    tip_3d = float(np.abs(solution_3d.u.reshape(-1, 3)[:, 2]).max())

    fields = Plotter(1, 2, figsize=(10.5, 4.2), title='Linear elasticity in 2D and 3D')
    fields.plot(deformed, solution.von_mises, mode='colored', idx=(0, 0),
                label='von Mises stress', title=f'2D: {len(mesh.elements)} triangles')
    fields.plot(solution_3d.deformed_mesh(), solution_3d.von_mises, mode='solid', idx=(0, 1),
                label='von Mises stress', title=f'3D: {len(box.elements)} tetrahedra')

    # The same stress tensor admits other rotation-invariant reductions besides von Mises:
    # mean normal stress, the Tresca measure, and the largest tensile principal value.
    # Each is its own question asked of one solve, not a different problem.
    invariant_fields = [
        ('Von Mises', solution.von_mises),
        ('Pressure', solution.pressure),
        ('Max shear', solution.max_shear),
        ('Max principal', solution.principal_stress[:, -1]),
    ]
    invariants = Plotter(2, 2, title='Stress invariants of the same solve', panel_aspect=4.0)
    for i, (name, values) in enumerate(invariant_fields):
        invariants.plot(deformed, values, mode='colored', idx=divmod(i, 2), title=name)

    conditions = Plotter(panel_aspect=4.0)
    conditions.plot(mesh, mode='bc', bc=bc)

    return DemoResult([
        Figure(fields,
               'The same clamp-and-load solved in 2D and 3D. The bending stress is largest '
               'at the clamp and splits top from bottom: tension over the neutral axis, '
               'compression under it. The 3D solve carries a three-component displacement '
               'and recovers stress the same way, drawn on its boundary surface.',
               'fields'),
        Figure(invariants,
               'Four rotation-invariant reductions of that same 2D stress tensor: '
               'distortion, mean normal stress, the Tresca measure, and the largest '
               'tensile principal value.',
               'invariants'),
        Figure(conditions,
               'Clamped along the left edge, pulled down over the middle of the right one. '
               'Everything between is traction-free, which is what makes this a cantilever '
               'rather than a beam being squeezed. The 3D solve imposes the same clamp and '
               'tip load, one dimension up.',
               'conditions', setup=True),
    ], text=(f'2D triangles           {len(mesh.elements)}\n'
             f'3D tetrahedra          {len(box.elements)}\n'
             f'3D degrees of freedom  {3 * len(box.vertices)}\n'
             f'3D peak deflection     {tip_3d:.4f}'))

def demo_topology_optimization(mesh, iters=60):
    """Optimize where to put half a beam's material with SIMP, and measure how much
    stiffness that buys against the fully solid block it came from."""
    E, nu = 200.0, 0.4
    w = np.max(mesh.vertices[:, 0])
    h = np.max(mesh.vertices[:, 1])
    aspect = float(w / h)

    # A simply supported (MBB) beam, the classic topology-optimization test: pinned at one
    # bottom corner, a vertical roller at the other, a downward load at the top centre.
    bc = BoundaryConditions()
    bottom, top = on_plane(1, 0.0), on_plane(1, h)
    bc.add(BCType.DIRICHLET, intersect(bottom, in_box([None, None], [0.04 * w, None])), [0, 0])
    bc.add(BCType.DIRICHLET, intersect(bottom, in_box([0.96 * w, None], [None, None])), [None, 0])
    # A load over the central fifth of the top rather than a single point: wide enough to
    # land on a boundary edge on any mesh (a point between two nodes carries no traction),
    # so the demo also runs on the tiny mesh the gallery smoke-tests it with.
    bc.add(BCType.NEUMANN, intersect(top, in_box([0.4 * w, None], [0.6 * w, None])), [0, -0.5])

    equation = LinearElastic(E, nu)

    # The solid block first: 100% material, the stiffest this domain and load admit, and
    # the baseline the optimized structure is measured against.
    solid = Solver(mesh, equation, bc).solve()
    compliance_solid = float(solid.compliance.sum())
    solid_disp = np.linalg.norm(solid.u.reshape(-1, 2), axis=1)

    # Then optimize where to put half of it. Compliance is u.f, the work the load does, so
    # a lower value is a stiffer structure; SIMP minimizes it under the volume constraint.
    topopt = TopologyOptimizer(mesh, equation, bc, iters=iters, volume_frac=0.5,
                               smoothing_radius=0.05)
    history = topopt.solve()
    compliance_opt = float(history.compliance[-1].sum())
    ratio = compliance_opt / compliance_solid

    # Explicit figsize rather than panel_aspect: two 4:1 panels stacked, sized so each
    # fills its row instead of floating above the aspect helper's minimum panel height.
    comparison = Plotter(2, 1, figsize=(6.5, 4.6),
                         title='Half the material, comparable stiffness')
    comparison.plot(solid.deformed_mesh(), solid_disp, mode='colored', idx=(0, 0), label='|u|',
                    title=f'Solid: 100% material, compliance {compliance_solid:.3f}')
    comparison.plot(topopt.deformed_mesh(), history.rho[-1], mode='colored', idx=(1, 0),
                    label='density',
                    title=f'Optimized: 50% material, compliance {compliance_opt:.3f} '
                          f'({ratio:.2f}x)')

    animation = Plotter(title='Topology optimization', panel_aspect=aspect)
    animation.plot_animation(mesh, history.rho, mode='colored', label='density')

    conditions = Plotter(panel_aspect=aspect)
    conditions.plot(mesh, mode='bc', bc=bc)

    return DemoResult([
        Figure(comparison,
               'The same simply supported beam under the same central load, solid and then '
               'with half its material removed by optimization, both drawn deformed. '
               'Compliance is the work the load does, so it measures deflection under load: '
               f'the optimized truss is only {ratio:.2f}x as compliant as the fully solid '
               'block on half the material, because what it removed was near the neutral '
               'axis, where it was barely resisting the bending.',
               'comparison'),
        Figure(animation,
               'Density evolving over the SIMP iterations, from an even grey to the '
               'black-and-white truss.',
               'animation'),
        Figure(conditions,
               'Simply supported: pinned at one bottom corner (both directions held), a '
               'vertical roller at the other (free to slide horizontally), and a downward '
               'load at the top centre.',
               'conditions', setup=True),
    ], text=(f'compliance, solid (100% material)     {compliance_solid:.4f}\n'
             f'compliance, optimized (50% material)  {compliance_opt:.4f}\n'
             f'ratio                                 {ratio:.2f}x'))


def demo_design_sensitivity(mesh, iters=40):
    """Optimize a cantilever with the general design driver, and show the adjoint
    sensitivity field it runs on: which elements matter most, for two different goals."""
    E, nu = 200.0, 0.4
    w = float(np.max(mesh.vertices[:, 0]))
    h = float(np.max(mesh.vertices[:, 1]))
    aspect = w / h

    # A cantilever: clamp the left edge, pull the free right tip down. Homogeneous
    # supports, so the compliance adjoint is the forward solve itself (lambda = u).
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    # A load band over the central third of the free edge rather than a point: wide enough
    # to land on a boundary edge on any mesh, so the demo also runs on the coarse smoke one.
    bc.add(BCType.NEUMANN, intersect(on_plane(0, w), in_box([None, 0.33 * h], [None, 0.67 * h])),
           [0.0, -1.0])

    space = FunctionSpace(mesh, n_components=2)
    radius = 0.06 * h
    model = SIMPModel(space, base_E=E, nu=nu, bc=bc, penalty=3.0,
                      sensitivity_filter=calculate_smoothing_matrix(mesh, radius))

    # The two adjoint sensitivity fields, computed on a uniform half-dense structure: one
    # per objective, before any optimization. Each says d(objective)/d(density) per
    # element, so it maps where material most changes that particular output.
    rho0 = np.full(len(space.element_nodes), 0.5)
    problem = model.problem(rho0)
    analysis = SensitivityAnalysis(problem)
    u0 = analysis.solve_forward()
    density = model.parameterization(rho0)

    tip_dof = _tip_vertical_dof(space, w)
    # Magnitude, since the tip sensitivity is signed (adding material can move the tip
    # either way locally); the magnitude is "how much this element steers the tip".
    compliance_field = -analysis.gradient(Compliance(), density, u0)
    tip_field = np.abs(analysis.gradient(PointValue(tip_dof), density, u0))

    sensitivity = Plotter(2, 1, figsize=(6.5, 4.6),
                          title='Adjoint sensitivity: which elements matter')
    sensitivity.plot(mesh, compliance_field, mode='colored', idx=(0, 0), label='dC/drho',
                     title='For total stiffness (compliance)')
    sensitivity.plot(mesh, tip_field, mode='colored', idx=(1, 0), label='|du_tip/drho|',
                     title='For the tip deflection alone')

    # Then optimize: the general DesignOptimizer minimizes compliance over the density,
    # its gradient supplied by the same adjoint core the fields above visualize.
    design = DesignOptimizer(model, Compliance(), volume_frac=0.5, iters=iters,
                             move=0.2).solve()
    solid = Solver(mesh, LinearElastic(E, nu), bc).solve()
    compliance_solid = float(solid.compliance.sum())
    compliance_opt = float(design.objective[-1])
    ratio = compliance_opt / compliance_solid

    result = Plotter(panel_aspect=aspect, title='Optimized cantilever')
    result.plot(mesh, design.rho[-1], mode='colored', label='density',
                title=f'50% material, compliance {compliance_opt:.3g} ({ratio:.2f}x solid)')

    conditions = Plotter(panel_aspect=aspect)
    conditions.plot(mesh, mode='bc', bc=bc)

    return DemoResult([
        Figure(sensitivity,
               'The adjoint gradient as a field, computed on the uniform half-dense beam '
               'before any optimization. Top: how much each element affects the total '
               'compliance, the stiffness of the whole structure. Bottom: how much each '
               'element affects the tip deflection specifically. The two light up '
               'different regions, which is the point of the adjoint: one solve answers '
               '"which inputs matter for this output", and the output can be anything.',
               'sensitivity'),
        Figure(result,
               'The cantilever optimized to minimum compliance under a 50% volume budget '
               'by the general DesignOptimizer, whose descent direction is the compliance '
               f'sensitivity field above. The result is {ratio:.2f}x as compliant as the '
               'fully solid block on half the material.',
               'result', thumbnail=True),
        Figure(conditions,
               'A cantilever: the left edge clamped, a downward load on the middle of the '
               'free right edge.',
               'conditions', setup=True),
    ], text=(f'compliance, solid (100% material)     {compliance_solid:.4f}\n'
             f'compliance, optimized (50% material)  {compliance_opt:.4f}\n'
             f'ratio                                 {ratio:.2f}x'))


def _tip_vertical_dof(space, width):
    """The vertical DOF of the loaded free-end node nearest mid-height, for the point QoI."""
    coords = space.node_coords
    on_tip = np.isclose(coords[:, 0], width)
    mid = np.median(coords[:, 1])
    candidates = np.where(on_tip)[0]
    node = int(candidates[np.argmin(np.abs(coords[candidates, 1] - mid))])
    return node * space.n_components + 1


def demo_buckling(length=24.0, height=1.0, n_length=48, n_across=6, n_modes=3,
                  sweep_lengths=(16.0, 20.0, 28.0, 40.0)):
    """Find the loads at which a slender column buckles and the shapes it buckles into,
    checked against Euler three ways: mode shapes, end conditions, and the 1/L^2 law."""
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
    modes = Plotter(1, n_modes, figsize=(3.2 * n_modes, 6.0), axis_labels=False,
                    title='Buckling modes of a pinned-pinned column')
    for i in range(n_modes):
        shape, colour = buckled(pinned_solution, i, length)
        modes.plot(shape, colour, mode='colored', idx=(0, i), cmap='coolwarm', colorbar=False,
                   title=f'Mode {i+1}: P_cr = {pinned_loads[i]:.3g}\n'
                         f'({i+1} half-wave{"s" if i else ""})')
        # The pin/load glyphs, on the deformed shape so the load rides the moving end.
        modes.overlay_supports(mesh, pinned_bc, idx=(0, i), coords=shape.vertices)
        # Drop the x ticks: on these tall, thin columns the labels only collide, and the
        # y axis already carries the scale (as on the modal modes plot).
        modes.get_ax((0, i)).tick_params(axis='x', labelbottom=False, bottom=False)
    _share_panel_limits(modes, n_modes)
    modes.fig.supxlabel(
        'Blue triangles: the pinned ends, held sideways but free to rotate.\n'
        'Red arrow: the compressive load.\n'
        'Colour: sideways deflection; its sign and amplitude are arbitrary.',
        fontsize='medium')

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
    _share_panel_limits(factor_plots, len(ends))

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
               'The same slender column held four ways, buckling at loads spanning 16x. '
               'Clamping an end against rotation shortens the effective length K*L the '
               'column buckles over, from 2L free-standing down to L/2 with both ends fixed, '
               'and the load goes as 1/K^2. The measured K sits within a few percent of '
               'Euler\'s 2, 1, 1/2 and ~0.7; the small excess is a real continuum effect, a '
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
    its own outline and checked against beam theory."""
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
        # No colorbar: the colour is qualitative (its amplitude is arbitrary), and one
        # shared caption below names it. The symmetric clim keeps the diverging map
        # centred on zero so the still tine reads white in every panel.
        modes.plot(shape, colour, mode='colored', idx=(0, i), cmap='coolwarm',
                   clim=(-lim, lim), colorbar=False, title=f'Mode {i+1}: {freqs[i]:.0f} Hz{tag}')
        modes.overlay_supports(mesh, clamp(), idx=(0, i), coords=shape.vertices)
        hide_x_ticks(modes, (0, i))
    # One shared vertical scale, so the tines line up across panels like the buckling modes.
    _share_panel_limits(modes, n_shown)
    modes.fig.supxlabel(
        'Colour: sideways (transverse) displacement of the mode. Its sign and amplitude '
        'are arbitrary; the pattern of motion is what is physical.', fontsize='medium')

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
               'for.',
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
    # Same box and resolution convention as the 3D cantilever in `linear_elastic`.
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


def demo_goal_oriented_refinement(resolution=14, target=(0.72, 0.72), max_triangles=900):
    """Refine a mesh for one quantity of interest, not the global error: a point value,
    and how the goal-oriented mesh concentrates where the global one does not."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=lambda p: 1.0)

    def refined(estimator_for):
        mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(resolution, resolution))
        solver = Solver(mesh, equation, bc)
        solver.solve()
        # The point value to resolve: the solution at the node nearest `target`.
        node = int(np.argmin(np.linalg.norm(solver.space.node_coords - np.asarray(target), axis=1)))
        AdaptiveRefinement(
            solver, estimator_for(solver, PointValue(node)),
            max_triangles=max_triangles, max_iters=8,
        ).run()
        return solver.mesh

    goal_mesh = refined(lambda solver, qoi: goal_oriented_estimator(equation, qoi))
    # The global recovery estimator ignores the goal, so it takes the QoI only to share
    # the closure signature; refinement follows the whole-domain error instead.
    global_mesh = refined(lambda solver, qoi: recovery_estimator(equation))

    comparison = Plotter(1, 2, figsize=(7.4, 3.9),
                         title='Refining for a point value, versus for the whole field')
    comparison.plot(global_mesh, mode='mesh', idx=(0, 0),
                    title=f'Global estimator: {len(global_mesh.elements)} triangles')
    comparison.plot(goal_mesh, mode='mesh', idx=(0, 1),
                    title=f'Goal-oriented: {len(goal_mesh.elements)} triangles')
    for idx in ((0, 0), (0, 1)):
        comparison.get_ax(idx).plot(*target, 'o', color='crimson', markersize=7,
                                    markeredgecolor='white', markeredgewidth=1.0, zorder=5)

    return DemoResult([
        Figure(comparison,
               'The same Poisson problem refined two ways to a similar triangle budget. '
               'Left: the global recovery estimator spreads refinement across the domain, '
               'blind to what the answer is for. Right: the goal-oriented estimator refines '
               'for the solution value at the marked point (crimson), packing triangles '
               'around it and the region that most influences it, and leaving the rest '
               'coarse. The dual (adjoint) solution is the influence function of that '
               'point value, and weighting the residual by it is what steers the mesh.',
               'comparison'),
    ], text=(f'global estimator refined to        {len(global_mesh.elements)} triangles\n'
             f'goal-oriented estimator refined to {len(goal_mesh.elements)} triangles'))


SOLVING = 'Solving PDEs'
SOLIDS = 'Solids & structures'
ACCURACY = 'Accuracy & performance'

DEMOS = [
    Demo('poisson', demo_poisson_equation, section=SOLVING, domain=partial(square, 80)),
    # Builds its own heatsink and a solid-block baseline (the shape is part of what it
    # shows), so it takes no domain. The smoke run loosens the size cap, takes a few steps,
    # and sweeps only two fin lengths for the efficiency check.
    Demo('heat', demo_heat_equation, section=SOLVING,
         smoke_kwargs={'max_area_fraction': 0.03, 'steps': 4, 'fin_lengths': (0.8, 2.0)}),
    Demo('heat_3d', demo_heat_3d, section=SOLVING, smoke_kwargs={'steps': 3, 'n': 5}),
    Demo('wave', demo_wave_equation, section=SOLVING, domain=square),
    # Builds its own airfoil-in-a-channel from the NACA formula (the meshing is part of
    # what it shows), so it takes no domain. The smoke run loosens the size cap and
    # coarsens the airfoil, which together are all of its cost.
    Demo('potential_flow', demo_potential_flow, section=SOLVING,
         smoke_kwargs={'n_points': 40, 'max_area_fraction': 0.02}),

    # The 2D cantilever whose domain this is, plus a 3D one the demo builds for itself.
    # The smoke run coarsens only that box, where the tet count sets the cost.
    Demo('linear_elastic', demo_linear_elastic, section=SOLIDS,
         domain=partial(beam, 4.0, 1.0, 140), smoke_kwargs={'n_3d': 6}),
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
    # Builds its own L-brackets (sharp and filleted) from outlines, so it takes no
    # domain. Its cost is the two adaptive-refinement chains, so the smoke run coarsens
    # the mesh and takes only a couple of rounds.
    Demo('bracket', demo_bracket, section=SOLIDS,
         smoke_kwargs={'max_area_fraction': 0.08, 'n_rounds': 2}),
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
    # A 4:1 simply supported (MBB) beam, the aspect that optimizes into the classic arch.
    # `smoothing_radius` is a fixed physical length setting the feature size, so the
    # resolution resolves the same structure more finely rather than growing thinner
    # members. The smoke run keeps the mesh but takes only a few iterations.
    Demo('topology_optimization', demo_topology_optimization, section=SOLIDS,
         domain=partial(beam, 4.0, 1.0, 160), smoke_kwargs={'iters': 3}),
    # The general design driver over the adjoint core, on a 3:1 cantilever. Leads with the
    # sensitivity field (the adjoint's own output) before the optimized result. Smoke runs
    # a coarse beam for a few iterations.
    Demo('design_sensitivity', demo_design_sensitivity, section=SOLIDS,
         domain=partial(beam, 3.0, 1.0, 120), smoke_kwargs={'iters': 3}),

    # Meshed deliberately coarse: the point is the resolution limit, so it runs where
    # sin(40 r^2)'s slow inner rings still resolve but the fast outer ones alias into the
    # triangulation. It leads the accuracy section because representation error is the
    # floor the rest of the section measures the shrinking of.
    Demo('l2_projection', demo_l2_projection, section=ACCURACY, domain=partial(square, 28),
         smoke_kwargs={'reference_resolution': 60}),
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
    # Builds its own coarse annulus and a convergence sequence (both are the demo), so it
    # takes no domain; the smoke run keeps the two coarsest resolutions.
    Demo('curved_elements', demo_curved_elements, section=ACCURACY,
         smoke_kwargs={'resolutions': (3, 5)}),
    Demo('quadrature_load', demo_quadrature_load, section=ACCURACY,
         smoke_kwargs={'resolutions': (11, 21)}),
    # Two adaptive-refinement chains (goal-oriented and global) from a coarse mesh; the
    # smoke run keeps the mesh small and caps the triangle budget so both chains are short.
    Demo('goal_oriented_refinement', demo_goal_oriented_refinement, section=ACCURACY,
         smoke_kwargs={'resolution': 8, 'max_triangles': 200}),
]
