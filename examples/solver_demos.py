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
from fem.boundary import BoundaryConditions, BCType
from fem.convergence import (
    ConvergenceStudy, elastic_convergence, load_comparison_convergence, poisson_convergence,
    poisson_p2_convergence, theta_convergence,
)
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.estimators import RecoveryEstimator
from fem.forms import EnergyForm, MaskedMassForm
from fem.problem import Problem
from fem.space import FunctionSpace
from fem.regions import on_plane, in_box, intersect, union
from fem.plot.plotter import Plotter
from fem.equations import Projection, Poisson, LinearElastic, StrainMeasure, Wave
from fem.solve import BacktrackingLineSearch, NewtonSolve
from fem.solver import Solver
from fem.mesh.ruppert import RuppertsAlgorithm
from fem.mesh.structured import create_box_mesh, create_rect_mesh
from fem.mesh.refinement import RedGreenRefiner
from fem.integrators import NewmarkMethod, ThetaMethod
from fem.design import DesignOptimizer, SIMPModel, calculate_smoothing_matrix
from fem.buckling import BucklingAnalysis
from fem.modal import ModalAnalysis

from demo_registry import Demo, DemoResult, Figure
from domains import (
    airfoil_channel_pslg, beam, column, harbor_pslg, heatsink_pslg, l_bracket_pslg,
    plate_with_hole_pslg, square, tuning_fork_pslg,
)

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

def demo_l2_projection(mesh, reference_resolution=120):
    """An oscillatory function projected onto a coarse mesh's P1 and P2 spaces, showing
    what each can represent."""
    def cool_f(point):
        x, y = point - np.array([0.5, 0.5])
        return [np.sin(40 * (x**2 + y**2))]

    p1 = Solver(mesh, Projection(source=cool_f)).solve()
    p2 = Solver(mesh, Projection(source=cool_f), element_type=QuadraticTriangleElement).solve()

    # The target sampled on a fine mesh, so the left panel is the function itself rather
    # than another coarse approximation of it. The projection (right) is on the demo's own
    # mesh, coarse enough that the fast outer rings outrun what P1 can represent.
    fine = square(reference_resolution)
    xy = fine.vertices - np.array([0.5, 0.5])
    exact = np.sin(40 * (xy[:, 0]**2 + xy[:, 1]**2))

    plotter = Plotter(1, 3, title='L2 projection onto one coarse mesh')
    plotter.plot(fine, exact, mode='colored', idx=(0, 0), label='', clim=(-1, 1),
                 title='The target: sin(40 r^2)')
    plotter.plot(mesh, p1.u, mode='colored', idx=(0, 1), label='', clim=(-1, 1),
                 title=f'P1 ({len(mesh.elements)} triangles)')
    plotter.plot(mesh, p2.u, mode='colored', idx=(0, 2), label='', clim=(-1, 1),
                 title='P2, same mesh', space=p2.space)
    return DemoResult([Figure(
        plotter,
        'How well a space can represent a function at all, before any PDE. Left: the '
        'target sin(40 r^2), whose rings tighten with radius. Middle: its L2 projection '
        'onto the P1 space of a coarse mesh. The slow inner rings come through; where one '
        'ring spans only a couple of triangles the space can no longer follow it, and the '
        'outer rings break up into the mesh. Right: the P2 space of the same mesh, whose '
        'edge-midpoint nodes let each element curve, follows the rings further out. This '
        'representation error is the floor every solver on the mesh starts from.')])

def demo_poisson(length=7.0, height=4.0, chord=3.0, angle_of_attack=12.0,
                 n_points=80, min_angle=20, max_area_fraction=0.0015):
    """Poisson's equation as potential flow over a NACA airfoil, on P2 elements."""
    # An ideal (incompressible, irrotational) flow has a velocity potential phi with
    # v = grad(phi) and div(v) = 0, so phi solves Laplace's equation, Poisson's with no
    # source. The wing carries no
    # flow through it, the natural (zero-flux) condition of the weak form: say nothing
    # on its surface and it becomes a streamline the flow parts around.
    pslg = airfoil_channel_pslg(length, height, chord, angle_of_attack, n_points=n_points)
    pslg.validate()
    mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                             max_area=max_area_fraction * pslg.area()).refine()

    equation = Poisson(source=0)   # Laplace: no sources in the flow
    bc = BoundaryConditions()
    # phi rises from inlet to outlet, so v = grad(phi) runs left to right. The wing and
    # the walls take no condition, so they are no-flux streamlines.
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)      # inlet (left)
    bc.add(BCType.DIRICHLET, on_plane(0, length), 1.0)   # outlet (right)

    solver = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement)
    solution = solver.solve()
    # v = grad(phi), read at the nodes so the P2 tessellation draws it smoothly.
    speed = np.linalg.norm(solution.nodal_flux(), axis=1)   # (n_nodes,)
    # Ideal flow with no Kutta condition predicts a near-singular velocity at the sharp
    # edges; clip it to a high percentile so the flow over the wing stays legible.
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
               'at all, which in the weak form is zero flux, so it is a streamline the flow '
               'parts around. The speed is clipped near the sharp edges, where ideal flow '
               'with no Kutta condition predicts an unphysical velocity spike.',
               'flow'),
        Figure(conditions,
               'A potential difference across the channel (phi = 0 at the inlet, 1 at the '
               'outlet) drives the flow left to right; the walls and the wing surface take no '
               'condition, so no flow crosses them.',
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
    """Convergence rates in space and time against manufactured solutions, P1 against
    P2, and the load built two ways."""
    # The one demo that shows not what the solver computed but how wrong it was:
    #
    #   in space  P1 elements are O(h^2) (halve h, quarter the error) for a scalar
    #             unknown and for a coupled vector one alike; P2 is O(h^3);
    #   in time   the theta method's order is theta's to choose: 1 at backward Euler,
    #             2 at Crank-Nicolson, the default.
    #
    # The same studies run as assertions in tests/test_convergence{,_elasticity,_heat}.py.
    solves = poisson_convergence(resolutions)
    p2_solves = poisson_p2_convergence(resolutions)
    poisson_study = ConvergenceStudy.from_solves(solves)
    p2_study = ConvergenceStudy.from_solves(p2_solves)
    elastic_study = ConvergenceStudy.from_solves(elastic_convergence(elastic_resolutions))
    # DOF counts for the accuracy-per-cost view: P2 spends more unknowns per element,
    # and the question is whether its faster rate pays that back.
    p1_dofs = np.array([FunctionSpace(s.mesh).n_dofs for s in solves])
    p2_dofs = np.array([FunctionSpace(s.mesh, QuadraticTriangleElement).n_dofs
                        for s in p2_solves])
    # Step counts chosen to sit in the asymptotic band: over coarser steps
    # Crank-Nicolson reads an order near 3, because lambda*dt is not yet small and
    # the leading error term is not yet the one that dominates.
    crank_nicolson = theta_convergence(0.5, step_counts)
    backward_euler = theta_convergence(1.0, step_counts)
    # The same P1 solve with the source read only at the vertices (its linear interpolant)
    # against one sampled at the quadrature points: the rate is the same, the constant
    # is not.
    loads = load_comparison_convergence(resolutions)
    load_steps = np.array([lc.h for lc in loads])
    nodal = ConvergenceStudy(load_steps, np.array([lc.nodal_error for lc in loads]))
    sampled = ConvergenceStudy(load_steps, np.array([lc.sampled_error for lc in loads]))

    plotter = Plotter(2, 2, figsize=(10.0, 8.0),
                      title='Convergence against manufactured solutions')
    space = plotter.chart_ax(idx=(0, 0), xlabel='h', ylabel='L2 error')
    _plot_study(space, poisson_study, 'Poisson, P1', 'tab:blue', 2, 'h')
    _plot_study(space, elastic_study, 'Elasticity, P1', 'tab:green', 2, 'h')
    _plot_study(space, p2_study, 'Poisson, P2', 'tab:orange', 3, 'h')
    space.set_title('Space: P1 is second order, P2 third')
    _tidy_log_axis(space, poisson_study.step)

    cost = plotter.chart_ax(idx=(0, 1), xlabel='degrees of freedom', ylabel='L2 error')
    cost.loglog(p1_dofs, poisson_study.error, 'o-', color='tab:blue', label='P1')
    cost.loglog(p2_dofs, p2_study.error, 'o-', color='tab:orange', label='P2')
    cost.set_title('Cost: P2 reaches a given accuracy first')
    cost.grid(True, which='both', alpha=0.3)
    cost.legend()

    time = plotter.chart_ax(idx=(1, 0), xlabel='dt', ylabel='L2 error')
    _plot_study(time, crank_nicolson, 'Crank-Nicolson', 'tab:blue', 2, 'dt')
    _plot_study(time, backward_euler, 'Backward Euler', 'tab:red', 1, 'dt')
    time.set_title('Time: the order is theta\'s to choose')
    _tidy_log_axis(time, crank_nicolson.step)

    load = plotter.chart_ax(idx=(1, 1), xlabel='h', ylabel='L2 error')
    _plot_study(load, nodal, 'source at vertices', 'tab:red', 2, 'h')
    _plot_study(load, sampled, 'source at quadrature points', 'tab:blue', 2, 'h')
    load.set_title('Load: sampling the source wins the constant')
    _tidy_log_axis(load, load_steps)

    rows = ['                      fitted order   expected']
    for name, study, expected in (('Poisson P1 (h)', poisson_study, 2),
                                  ('Poisson P2 (h)', p2_study, 3),
                                  ('Elasticity (h)', elastic_study, 2),
                                  ('Crank-Nicolson (dt)', crank_nicolson, 2),
                                  ('Backward Euler (dt)', backward_euler, 1),
                                  ('Nodal load (h)', nodal, 2),
                                  ('Sampled load (h)', sampled, 2)):
        rows.append(f'{name:<22}{study.fitted_order:>9.2f}{expected:>11}')
    return DemoResult(
        [Figure(plotter,
                'Top left: on the same meshes, halving h quarters the P1 error (order 2), '
                'for a scalar unknown and for a coupled vector one alike, and divides the P2 '
                'error by eight (order 3). Top right: the same errors against the number of '
                'unknowns; P2 spends more DOFs per element but reaches a given accuracy with '
                'fewer of them. Bottom left: the error against the time step, where backward '
                'Euler is first order and Crank-Nicolson second, for the same cost per step. '
                'Bottom right: an oscillatory source read only at the vertices against one '
                'sampled at the quadrature points. Both are second order; the sampled load '
                'is about 3x more accurate on every mesh.')],
        text='\n'.join(rows),
    )

def _finite_plate_kt(hole_over_width: float) -> float:
    """Howland's stress concentration factor for a circular hole in a finite-width plate
    under tension, relative to the applied (gross) stress. Peterson's polynomial fit
    gives the factor on the net section; dividing by the net fraction of width puts it
    on the applied stress. Reads Kirsch's 3 for a vanishing hole."""
    r = hole_over_width
    net = 3.000 - 3.140 * r + 3.667 * r**2 - 1.527 * r**3
    return net / (1.0 - r)


def demo_stress_concentration(traction=1.0, length=6.0, height=3.0, radius=0.15,
                              min_angle=25, max_area_fraction=0.01, circle_segments=16,
                              refinement_iters=36, refinement_budget=40000):
    """A plate with a hole, from outline to the stress concentration at its rim, against
    Kirsch and Howland."""
    # The one demo that runs the whole pipeline, so it builds its own mesh: the outline,
    # what Ruppert's was asked for, and where the conditions went are part of what it
    # shows. The conditions are written against coordinates, so they resolve against
    # whatever triangulation arrives, including the ones adaptive refinement rebuilds.
    #
    # The hole is a coarse 16-gon, which is enough: `plate_with_hole_pslg` tags the hole
    # loop with a `Circle`, so Ruppert's split points, red-green refinement, and the
    # isoparametric element's edge nodes all land on the true rim.
    pslg = plate_with_hole_pslg(length, height, radius, segments=circle_segments)
    pslg.validate()
    # Coarse: resolving the rim is adaptive refinement's job below. The rim
    # still grades finer than the interior, since Ruppert's honours its short segments.
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area())
    mesh = rupperts.refine()
    n_initial = len(mesh.elements)
    initial_worst_angle = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()

    # The rim takes no condition: a free surface is the natural boundary condition of
    # the weak form, so "traction-free" is what an edge means when nothing is said.
    #
    # The left edge is a roller, not a clamp: pinned normal to itself (x = 0), free
    # tangentially (y) so the plate can narrow as it stretches. A clamp would resist
    # that Poisson contraction and add its own stress concentration, which competes with
    # the hole for the estimator's attention. Pinning y along the edge would do the same,
    # so a second condition pins y at one corner only, removing the last rigid-body mode.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, None])
    bc.add(BCType.DIRICHLET, intersect(on_plane(0, 0.0), on_plane(1, 0.0)), [None, 0])
    bc.add(BCType.NEUMANN, on_plane(0, length), [traction, 0])

    # Solved on the curved quadratic element and adaptively refined by the recovery
    # estimator, which reads the curved rim's stress correctly. Everything plotted and
    # measured below is read off the refined mesh. The rim splits project onto the true
    # circle, so more refinement keeps rounding the hole.
    equation = LinearElastic(E=200, nu=0.3)
    refinement = AdaptiveRefinement(
        mesh, lambda m: equation.problem(m, bc, element_type=IsoparametricTriangleElement),
        RecoveryEstimator(),
        max_triangles=n_initial + refinement_budget, max_iters=refinement_iters,
    )
    solution = refinement.run()
    mesh = refinement.mesh
    # The stress at the nodes: each element evaluated at its own nodes and averaged
    # where they meet, so the rim value is read on the rim itself.
    nodes = solution.space.node_coords
    sigma_xx = solution.nodal_stress()[:, 0, 0]

    # A vertical strip through the hole's centre: the line the concentration decays
    # along. The rim crossings are mesh nodes (the 16-gon has a vertex at the top and
    # bottom of the hole, and refinement keeps it), so the peak is the value there.
    strip = np.abs(nodes[:, 0] - length/2) < 0.25*radius
    order = np.argsort(nodes[strip, 1])
    y_strip, ratio_strip = nodes[strip, 1][order], (sigma_xx[strip] / traction)[order]
    on_rim = np.isclose(nodes[:, 0], length/2) & np.isclose(np.abs(nodes[:, 1] - height/2), radius)
    peak = float(sigma_xx[on_rim].max() / traction)

    # Two reference values. Kirsch's factor of 3 is for a hole in an infinite plate.
    # This plate is finite, and the hole removes some of its section, so the remaining
    # material carries slightly more stress and the exact peak is a little above 3.
    # Howland (1930) worked out that finite-width correction for a strip with a central
    # hole; `_finite_plate_kt` gives his value at this hole/width ratio, and that is the
    # line the measured peak is judged against.
    #
    # The peak converges to it from below, since a finite element solution is slightly
    # too stiff and the steepest gradient is the last thing it resolves: 2.97, 3.00,
    # 3.00, 3.03 over 624, 970, 1877 and 3301 elements. Thirty-six rounds is enough to
    # agree to within a hundredth.
    hole_over_width = 2*radius / height
    finite_kt = _finite_plate_kt(hole_over_width)

    # One figure, three plots: mesh with outline and conditions, the stress, the chart.
    figure = Plotter(1, 3, figsize=(14.0, 3.6),
                     title='From an outline to a stress concentration')
    figure.plot(mesh, mode='bc', bc=bc, idx=(0, 0),
                title=f'{len(mesh.elements)} triangles (refined from {n_initial}), '
                      'with conditions')
    ax0 = figure.get_ax((0, 0))
    # The triangulation under the conditions: thin and grey, below the glyphs.
    ax0.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements,
                color='0.55', linewidth=0.2, zorder=1.5)
    # The input segments over the triangulation: which of the outline the mesher kept.
    ax0.add_collection(LineCollection(
        rupperts.vertices[rupperts.segments], colors='blue', linewidths=1.0, zorder=2.0))
    # Passing the solution draws the P2 field on its own tessellation, rim and all.
    figure.plot(solution, sigma_xx, mode='colored', idx=(0, 1), label='sigma_xx',
                title='Stress concentration (sigma_xx)')
    ax = figure.chart_ax(idx=(0, 2), xlabel='y', ylabel='sigma_xx / applied')
    # Two runs, below the hole and above it, so nothing is drawn across the gap.
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
    # Over the flat far field, where the legend hides nothing.
    ax.legend(loc='center left', fontsize='small')

    # Ruppert's angle guarantee does not survive red-green refinement, which bisects
    # existing triangles rather than re-triangulating for shape; reported rather than
    # hidden.
    worst_angle = calculate_triangle_min_angle(
        np.asarray(mesh.vertices)[np.asarray(mesh.elements)]).min()
    # rupperts.boundary_loops describes the initial triangulation, not the refined mesh
    # (see BACKLOG.md), so this counts the rim facets before refinement.
    rim_facets = int(np.sum(rupperts.boundary_loops == 1))
    return DemoResult(
        [Figure(figure,
                f'The whole pipeline in one row. Left: the mesh after adaptive refinement, '
                f'grown from {n_initial} triangles to {len(mesh.elements)} where the '
                f'recovery estimator found the most error, with the input outline in blue '
                f'and the conditions drawn on it; the rim and long edges carry none and are '
                f'traction-free. Middle: the stress sigma_xx on curved quadratic elements, '
                f'crowding into the material either side of the hole and relaxing to the '
                f'applied value within about a diameter. Right: that stress along a strip '
                f'through the hole centre, peaking at {peak:.2f}x the applied value at the '
                f'rim. The classic Kirsch factor of 3 is for a hole in an infinite plate; '
                f"Howland's value for a hole {hole_over_width:.2f} of this plate's width is "
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
    """The stress at an L-bracket's inner corner, sharp and filleted, as the mesh
    refines into it.

    Solved on quadratic (P2) elements: the sharp bracket on straight `QuadraticTriangleElement`,
    the filleted one on `IsoparametricTriangleElement` so the arc is a true circle rather than
    a polygon. The recovery estimator drives refinement (it reads the curved fillet's flux
    correctly), and the peak is read from the recovered nodal von Mises."""
    # At the re-entrant corner the exact elastic stress is infinite (it grows like
    # r^(-0.46) into the corner), so no mesh resolves it and the computed peak keeps
    # climbing under refinement. A fillet removes the singularity and the peak settles.
    equation = LinearElastic(E, nu)
    corner = np.array([width, width])

    def make_bc():
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(1, arm), [0, 0])        # clamp the top of the upright limb
        bc.add(BCType.NEUMANN, on_plane(0, arm), [0, -traction])  # pull the horizontal tip down
        return bc

    def corner_peak(solution):
        # The von Mises peak near the inner corner, clear of the clamp's own concentration
        # at the top. Read from the same L2-recovered nodal field the panels draw.
        space = solution.space
        nodal_vm = solution.nodal_von_mises(method='l2')
        near = np.linalg.norm(space.node_coords - corner, axis=1) < 0.8 * width
        return float(nodal_vm[near].max())

    def refine_and_track(fillet, element_type):
        """Adaptively refine one bracket, recording the corner peak each round.

        `AdaptiveRefinement`'s loop, unrolled so the corner peak can be read off every
        intermediate mesh. `element_type` is the straight quadratic triangle for the
        sharp corner and the isoparametric one for the fillet.
        """
        pslg = l_bracket_pslg(arm, width, fillet_radius=fillet, n_fillet=20)
        pslg.validate()
        mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area()).refine()
        bc = make_bc()

        def solve(m):
            problem = equation.problem(m, bc, element_type=element_type)
            return problem, problem.solve()

        refiner = RedGreenRefiner(mesh)
        estimator = RecoveryEstimator()
        problem, solution = solve(mesh)
        sizes, peaks = [], []
        for _ in range(n_rounds):
            sizes.append(len(mesh.elements))
            peaks.append(corner_peak(solution))
            residuals = estimator.estimate(problem, solution)
            refine_idxs = np.flatnonzero(residuals >= refine_fraction * residuals.max())
            mesh = refiner.refine([int(i) for i in refine_idxs])
            problem, solution = solve(mesh)
        sizes.append(len(mesh.elements))
        peaks.append(corner_peak(solution))
        return mesh, solution, np.array(sizes), np.array(peaks)

    sharp_mesh, sharp, sharp_sizes, sharp_peaks = refine_and_track(0.0, QuadraticTriangleElement)
    round_mesh, rounded, round_sizes, round_peaks = refine_and_track(
        fillet_radius, IsoparametricTriangleElement)

    conditions = Plotter(panel_aspect=1.0)
    conditions.plot(sharp_mesh, mode='bc', bc=make_bc())

    # Independent colour scales: the sharp peak would wash the fillet's to a flat colour.
    fields = Plotter(1, 2, title='An L-bracket under a tip load')
    for i, (name, mesh, solution, peaks) in enumerate((
            ('Sharp corner', sharp_mesh, sharp, sharp_peaks),
            (f'Fillet r = {fillet_radius:g}', round_mesh, rounded, round_peaks))):
        # Passing the solution draws the P2 field on its tessellation; warp=True deforms it.
        fields.plot(solution, solution.nodal_von_mises(method='l2'), mode='colored',
                    idx=(0, i), warp=True, label='von Mises stress',
                    title=f'{name}\n{len(mesh.elements)} elements, corner peak {peaks[-1]:.0f}')
        # Support glyphs at the deformed vertex positions, so the load follows the tip.
        deformed_vertices = mesh.vertices + solution.u.reshape(-1, 2)[:len(mesh.vertices)]
        fields.overlay_supports(mesh, make_bc(), idx=(0, i), coords=deformed_vertices)

    sweep = Plotter(1, 1, title='The corner peak against mesh refinement')
    ax = sweep.chart_ax(xlabel='elements', ylabel='von Mises at the inner corner')
    ax.semilogx(sharp_sizes, sharp_peaks, 'o-', color='tab:red', label='sharp (singular)')
    ax.semilogx(round_sizes, round_peaks, 'o-', color='tab:blue',
                label=f'fillet r = {fillet_radius:g} (converges)')
    ax.set_title('Sharp corner keeps climbing; the fillet settles')
    # The element counts span under a decade, where a log axis crowds its minor labels.
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
               'corner. The sharp corner is a stress singularity, so its peak climbs without '
               'bound; the "stress" there is a property of the mesh, not of the part. The '
               'fillet removes the singularity, and its peak settles on a finite value. This '
               'is why real parts round their inner corners.',
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
    """One clamped block stretched by a linear solve, by energy minimisation, and at
    finite strain."""
    # Panels 1 and 2 are the same physics reached two ways: solving K u = f, and Newton
    # on the elastic energy that system is the stationary point of. Their displacements
    # agree to machine precision (printed below). Their stress does not, since the two
    # recover different measures: sigma = D:eps against the true Cauchy stress J^-1 P F^T
    # at the deformed configuration, which agree only to O(||grad u||).
    #
    # Panel 3 changes the physics: the small-strain eps is the leading term of the
    # Green-Lagrange S = (F^T F - I)/2, so the finite-strain model stiffens as the
    # stretch grows and the linear one cannot.
    #
    # The stress peak sits at the clamped corners, where the imposed displacement is
    # singular, so the median is quoted beside it.
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, w), [stretch*w, 0])

    linear = LinearElastic(E=200, nu=0.4)
    finite = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    # Panel 2 states small strain as an energy and minimises it: the same density the
    # linear stiffness is the Hessian of, under Newton.
    energy_problem = Problem(linear.space(mesh), EnergyForm(linear.energy_density()), bc=bc)
    energy_u = NewtonSolve(line_search=BacktrackingLineSearch()).solve(energy_problem)
    solutions = [
        ('Linear solve\n(small strain)', Solver(mesh, linear, bc).solve()),
        ('Energy minimisation\n(small strain)', energy_problem.solution(energy_u)),
        ('Energy minimisation\n(Green-Lagrange)', Solver(mesh, finite, bc).solve()),
    ]

    conditions = Plotter()
    conditions.plot(mesh, mode='bc', bc=bc)

    plotter = Plotter(1, 3, title=f'One {stretch:.0%} stretch, three ways to solve it')
    for i, (name, solution) in enumerate(solutions):
        vm = solution.von_mises
        plotter.plot(solution.deformed_mesh(), vm, mode='colored', idx=(0, i),
                     label='von Mises stress',
                     title=f'{name}\nmedian {np.median(vm):.0f}, peak {vm.max():.0f}')
    linear_u = solutions[0][1].u
    drift = np.linalg.norm(energy_u - linear_u) / np.linalg.norm(linear_u)
    return DemoResult(
        [Figure(plotter,
                'The first two are the same physics solved two ways, as a linear system and '
                'by Newton on the energy that system is the stationary point of. Their '
                'displacements are identical to machine precision (below). Their stress is '
                'not, because the two recover different measures, sigma = D:eps against the '
                'Cauchy stress at the deformed configuration, which agree only for small '
                'gradients. The third changes the physics: Green-Lagrange stiffens as the '
                'stretch grows, which small strain cannot.',
                'stress'),
         Figure(conditions,
                'Both ends are Dirichlet. The left is held at zero and the right is displaced '
                f'to {stretch:.0%} of the width. Nothing is loaded; the stress above is what '
                'it costs to hold that shape.',
                'conditions', setup=True)],
        text=(f'displacement, linear solve vs energy minimisation: '
              f'relative difference {drift:.1e}\n'
              f'minimised elastic energy: {energy_problem.energy(energy_u):.4g}'),
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
    """Warm a finned heatsink from a cold start, then compare it with a solid block and
    with beam theory."""
    # The heat equation is Poisson's operator integrated in time (see fem.problem.heat).
    # A heatsink conducts heat up its fins and sheds it, so the shape is worth measuring;
    # the mesh is built here because it is part of what the demo says.
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
    # convective film, du/dn + kappa*(u - u_ambient) = 0. A cold start at ambient makes
    # the run a warm-up, the front climbing the fins to a steady gradient.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), u_hot)
    bc.add_robin(_heatsink_film(mesh), kappa=kappa, g=kappa * u_ambient)
    heat = Poisson().problem(mesh, bc)
    u_initial = heat.space.interpolate(u_ambient)
    solution = ThetaMethod(dt=dt, steps=steps).solve(heat, u_initial)
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
    # Mark only the base, below the field: arrows for the Neumann flux (fixed-power row),
    # a bar for the held Dirichlet base (fixed-temperature row). The Robin film is every
    # other surface, named in the legend.
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
               'ways. Top, the same heat flux into each base (a chip of fixed power). The '
               f'block runs {u_block_p.max()-u_ambient:.0f} C above ambient, the finned sink '
               f'only {u_fin_p.max()-u_ambient:.0f} C, roughly halving the thermal resistance '
               f'(R {r_block:.2f} -> {r_fin:.2f}). Bottom, each base held at {u_hot:.0f}. The '
               f'finned sink sheds {effectiveness:.1f}x the heat with {metal_ratio:.2f}x the '
               'metal, since the fins trade material for surface area.',
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
               "long fin runs cold toward the tip and carries less of its share. This sink's "
               f'fins (L = 1.4) sit near {eta_here:.0%}, trading efficiency for surface area.',
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

def demo_wave_equation(c=1.0, front_x=1.0, front_width=0.25, dt=0.02, steps=400,
                       min_angle=28, max_area=0.04, uniform_rounds=2):
    """A wave front meeting a harbor breakwater, diffracting through its gap into the
    sheltered water behind."""
    wall_x, wall_thickness = 2.5, 0.15
    pslg = harbor_pslg(wall_x=wall_x, wall_thickness=wall_thickness)
    # Ruppert's meshes the outline coarsely; uniform red refinement then supplies the
    # resolution the front needs, keeping the angle bound at a fraction of the cost.
    mesh = RuppertsAlgorithm(pslg, min_angle=min_angle, max_area=max_area).refine()
    for _ in range(uniform_rounds):
        mesh = RedGreenRefiner(mesh).refine(range(len(mesh.elements)))

    # No conditions, so every edge is a wall: the natural du/dn = 0 reflects a wave
    # the same way up.
    bc = BoundaryConditions()
    wave = Wave(c).problem(mesh, bc)

    # A straight front on the open side, travelling toward the wall. Given d'Alembert's
    # pairing u = g(x - ct), du/dt = -c g'(x), so it moves one way instead of splitting.
    def profile(p):
        return np.exp(-((p[0] - front_x) / front_width) ** 2)

    u_initial = wave.space.interpolate(profile)
    dudt_initial = wave.space.interpolate(
        lambda p: 2 * c * (p[0] - front_x) / front_width**2 * profile(p))
    solution = NewmarkMethod(dt=dt, steps=steps).solve(wave, u_initial, dudt_initial)
    u_values, t_values = solution.u, solution.t

    setup = Plotter(1, 3, figsize=(15.0, 3.8))
    setup.plot(mesh, mode='mesh', idx=(0, 0), title='Basin and breakwater')
    setup.plot(mesh, u_initial, mode='colored', idx=(0, 1), label='height',
               title='Initial height u(x, 0)')
    setup.plot(mesh, dudt_initial, mode='colored', idx=(0, 2), label='velocity',
               title='Initial velocity, a front moving right')

    # One colour scale, set by the harbor side, so the diffracted wave reads even
    # though it is far lower than the front that made it (which doubles again when it
    # reflects off the far wall).
    shown = [int(i) for i in np.linspace(len(u_values) // 8, len(u_values) - 1, 8)]
    harbor = mesh.vertices[:, 0] > wall_x + wall_thickness
    span = float(max(abs(u_values[i][harbor]).max() for i in shown))
    clim = (-span, span)

    animation = Plotter(1, 1, figsize=(7.4, 4.8))
    animation.plot_animation(mesh, u_values, mode='colored', cbar_lims=clim, label='height',
                             cmap='RdBu_r', titles=[f'Harbor breakwater  t={t:.2f}' for t in t_values],
                             idx=(0, 0))

    snapshots = Plotter(2, 4, figsize=(18.0, 6.4), title='Diffraction through the gap')
    for panel, i in enumerate(shown):
        snapshots.plot(mesh, u_values[i], mode='colored', idx=divmod(panel, 4),
                       title=f't={t_values[i]:.2f}', clim=clim, colorbar=panel == 7,
                       cmap='RdBu_r', label='height')


    return DemoResult([
        Figure(animation,
               'Newmark time integration of the front.',
               'animation'),
        Figure(snapshots,
               'The front reaches the breakwater, reflects off the wall, and passes the '
               'gap, where it spreads into the harbor as a circular wave centred on the '
               'opening, lower than the front that made it. The later frames show that wave '
               'reflecting around the harbor while the front, reflected off the wall and '
               'then the far edge, comes back through the gap.',
               'snapshots'),
        Figure(setup,
               'A basin with a breakwater across it, open on the left and sheltered on the '
               'right. The initial height and velocity together make a front travelling '
               'right; every edge is a wall, reflecting the wave the same way up.',
               'conditions', setup=True),
    ])

def demo_linear_elastic(mesh, n_3d=14):
    """A cantilever under a tip load in 2D and 3D, with four stress invariants of the 2D
    solve."""
    E, nu = 200.0, 0.4

    # -- 2D: clamped on the left, pulled down over the middle of the right edge ---------
    w = np.max(mesh.vertices[:, 0])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    # Transverse, so the beam bends. Sized for a tip deflection near 9% of the span,
    # inside the small-strain regime.
    bc.add(BCType.NEUMANN,
           intersect(on_plane(0, w), in_box([None, 0.2], [None, 0.8])),
           [0, -0.5])
    solution = Solver(mesh, LinearElastic(E, nu), bc).solve()
    deformed = solution.deformed_mesh()

    # -- 3D: the same clamp-and-load, one dimension up ---------------------------------
    # The same assembly, Solver reading the tetrahedron off the connectivity. AMG-CG
    # rather than a direct factorization, whose fill-in hurts in 3D. Only the boundary
    # surface is drawn.
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

    # Other rotation-invariant reductions of the same stress tensor: mean normal stress,
    # the Tresca measure, and the largest tensile principal value.
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
               'The same clamp and load solved in 2D and 3D. The bending stress is largest '
               'at the clamp, with tension over the neutral axis and compression under it. '
               'The 3D solve carries a three-component displacement and recovers stress the '
               'same way, drawn on its boundary surface.',
               'fields'),
        Figure(invariants,
               'Four rotation-invariant reductions of the same 2D stress tensor: von Mises, '
               'mean normal stress, the Tresca measure, and the largest tensile principal '
               'value.',
               'invariants'),
        Figure(conditions,
               'Clamped along the left edge, pulled down over the middle of the right one; '
               'everything between is traction-free. The 3D solve imposes the same clamp '
               'and tip load, one dimension up.',
               'conditions', setup=True),
    ], text=(f'2D triangles           {len(mesh.elements)}\n'
             f'3D tetrahedra          {len(box.elements)}\n'
             f'3D degrees of freedom  {3 * len(box.vertices)}\n'
             f'3D peak deflection     {tip_3d:.4f}'))

def demo_topology_optimization(mesh, iters=60):
    """SIMP topology optimization of a beam to half its material, compared with the
    solid beam."""
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
    # A load over the central fifth of the top rather than a point, so it lands on a
    # boundary edge on any mesh, including the tiny smoke-test one.
    bc.add(BCType.NEUMANN, intersect(top, in_box([0.4 * w, None], [0.6 * w, None])), [0, -0.5])

    equation = LinearElastic(E, nu)

    # The solid block first: 100% material, the baseline the optimized one is measured against.
    solid = Solver(mesh, equation, bc).solve()
    compliance_solid = float(solid.compliance.sum())
    solid_disp = np.linalg.norm(solid.u.reshape(-1, 2), axis=1)

    # Then optimize where to put half of it. Compliance is u.f, the work the load does, so
    # a lower value is a stiffer structure; SIMP minimizes it under the volume constraint.
    model = SIMPModel(equation.problem(mesh, bc),
                      sensitivity_filter=calculate_smoothing_matrix(mesh, 0.05))
    design = DesignOptimizer(model, volume_frac=0.5, iters=iters, move=0.1)
    history = design.run()
    compliance_opt = history.objective[-1]
    ratio = compliance_opt / compliance_solid

    # Explicit figsize: two 4:1 panels stacked, each filling its row.
    comparison = Plotter(2, 1, figsize=(6.5, 4.6),
                         title='Half the material, comparable stiffness')
    comparison.plot(solid.deformed_mesh(), solid_disp, mode='colored', idx=(0, 0), label='|u|',
                    title=f'Solid: 100% material, compliance {compliance_solid:.3f}')
    assert design.solution is not None
    comparison.plot(design.solution.deformed_mesh(), history.rho[-1], mode='colored', idx=(1, 0),
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
               'Compliance is the work the load does, so it measures deflection under load. '
               f'The optimized truss is only {ratio:.2f}x as compliant as the fully solid '
               'block on half the material; what it removed was near the neutral axis, '
               'where it was barely resisting the bending.',
               'comparison'),
        Figure(animation,
               'Density evolving over the SIMP iterations, from an even grey to the '
               'black-and-white truss.',
               'animation'),
        Figure(conditions,
               'Simply supported, pinned at one bottom corner (both directions held) with a '
               'vertical roller at the other (free to slide horizontally), and a downward '
               'load at the top centre.',
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
    """Buckling loads and modes of a slender column, checked against Euler's column
    formula."""
    # Buckling is an eigenproblem: a reference load puts the column under a prestress,
    # BucklingAnalysis assembles the geometric stiffness K_g from it and solves
    # K phi = -lambda K_g phi, and lambda multiplies the reference load. P2 elements
    # throughout: the constant-strain triangle locks in bending.
    E, nu = 200.0, 0.3
    E_star = E / (1 - nu**2)     # plane-strain effective modulus, the one bending sees
    moment = height**3 / 12      # second moment of area of the rectangular section
    n_across += n_across % 2      # a vertex on the neutral axis, for the pinned anchor
    equation = LinearElastic(E, nu)

    def solve_buckling(mesh, bc, span, modes=n_modes):
        problem = equation.problem(mesh, bc, element_type=QuadraticTriangleElement)
        solution = BucklingAnalysis(n_modes=modes).solve(problem)
        # The load factor multiplies the reference load; the physical buckling load is
        # that factor times the actual axial force the column carries, read at mid-span
        # where it is uniform and clear of the end disturbances.
        centroids = mesh.vertices[mesh.elements].mean(axis=1)
        dy = span / (len(np.unique(mesh.vertices[:, 1])) - 1)
        midspan = np.abs(centroids[:, 1] - span / 2) < dy
        assert solution.reference is not None
        axial = -float(np.mean(solution.reference.stress[midspan, 1, 1])) * height
        return solution, solution.load_factors * axial

    # The four classic end conditions. What sets an end's effective-length factor is
    # whether it can rotate: a traction-loaded edge (u_y free) rotates (a pin or a free
    # end), an imposed uniform axial displacement (u_y fixed) cannot (a clamp). u_x = 0
    # along an edge holds it transversely without touching its rotation. The column
    # stands along y, so the ends are at y = 0 and y = span and the load pushes in -y.
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
    # Upright columns in a row, with one glyph-and-colour key below all of them.
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
        # Drop the x ticks: on these tall, thin columns the labels only collide.
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
               'if the lower ones are braced out. A support at mid-span, a node of mode 2 '
               'but not of mode 1, buys the jump to it. The shapes are the eigenvectors of '
               'K phi = -lambda K_g phi and the load factors its eigenvalues.',
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
               'Euler\'s column formula gives the buckling load of an ideal slender elastic '
               'column, P_cr = pi^2 E* I / (K L)^2. Left: sweeping the length of a pinned '
               'column, the critical '
               'load falls as 1/L^2 (a slope of -2 on log-log) and lands on it, with '
               'E* = E/(1-nu^2) the plane-strain modulus a 2D solve sees. Right: the '
               'effective-length factor K read back from each end condition\'s buckling load, '
               'against the textbook values.',
               'laws'),
        Figure(conditions,
               'A pinned-pinned column: both ends held across their width (u_y = 0) so they '
               'stay in line but can still rotate, one point anchoring the axial slide, and a '
               'compressive traction on the right. The transverse support and the axial load '
               'share the loaded edge, a roller carrying a tangential traction.',
               'conditions', setup=True),
    ], text=text)


def demo_modal(tine_length=0.088, tine_thickness=0.004, n_across_tine=5, min_angle=27,
               n_modes=6, n_shown=4, sweep_lengths=(0.075, 0.088, 0.105, 0.125),
               n_frames=24):
    """Natural frequencies and modes of a steel tuning fork meshed from its outline,
    against beam theory."""
    # Real SI steel, so the frequencies come out in Hz a musician would recognise.
    # E* = E/(1-nu^2) is the plane-strain modulus a 2D solve sees.
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
        # Element size is set by resolving the thin tine: bending curves across it.
        mesh = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=0.5*(tine_thickness/across)**2).refine()
        equation = LinearElastic(E, NU, density=RHO)
        problem = equation.problem(mesh, clamp(), element_type=QuadraticTriangleElement)
        solution = ModalAnalysis(n_modes=modes).solve(problem)
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
        # No colorbar: the amplitude is arbitrary, and one caption below names the colour.
        # The symmetric clim keeps the still tine white in every panel.
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
    # Not a dynamics simulation: a standing-wave mode is a fixed shape times cos(omega t),
    # evaluated frame by frame.
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
               'fork sheds the others and settles onto this one, so it sounds a single '
               'clean tone.',
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
               'the stem base: that clamp grounds the structure (a free body has rigid-body '
               'modes the shift-invert eigensolve cannot factor through) and is where a fork '
               'is held, the one place that does not damp the voice.',
               'built', setup=True),
    ], text=text)


SOLVING = 'Meshing & solving PDEs'
SOLIDS = 'Solids & structures'
ACCURACY = 'Accuracy & performance'

DEMOS = [
    # Builds its own heatsink and a solid-block baseline, so it takes no domain.
    Demo('heat', demo_heat_equation, section=SOLVING,
         smoke_kwargs={'max_area_fraction': 0.03, 'steps': 4, 'fin_lengths': (0.8, 2.0)}),
    # Builds its own harbor basin, so it takes no domain.
    Demo('wave', demo_wave_equation, section=SOLVING,
         smoke_kwargs={'steps': 6, 'max_area': 0.5, 'uniform_rounds': 0}),
    # Builds its own airfoil-in-a-channel from the NACA formula, so it takes no domain.
    Demo('poisson', demo_poisson, section=SOLVING,
         smoke_kwargs={'n_points': 40, 'max_area_fraction': 0.02}),

    # The 2D cantilever whose domain this is, plus a 3D box the demo builds for itself.
    Demo('linear_elastic', demo_linear_elastic, section=SOLIDS,
         domain=partial(beam, 4.0, 1.0, 140), smoke_kwargs={'n_3d': 6}),
    # A square keeps the deformed and undeformed shapes comparable at a glance.
    Demo('elasticity_models', demo_elasticity_models, section=SOLIDS,
         domain=partial(square, 60)),
    # The pipeline demo builds its own domain, from an outline through to a stress.
    Demo('stress_concentration', demo_stress_concentration, section=SOLIDS,
         smoke_kwargs={'max_area_fraction': 0.05, 'refinement_iters': 3, 'refinement_budget': 200}),
    # Builds its own L-brackets (sharp and filleted) from outlines, so it takes no domain.
    Demo('bracket', demo_bracket, section=SOLIDS,
         smoke_kwargs={'max_area_fraction': 0.08, 'n_rounds': 2}),
    # Builds its own columns (several lengths, four end conditions), so it takes no domain.
    Demo('buckling', demo_buckling, section=SOLIDS,
         smoke_kwargs={'n_length': 12, 'n_across': 4, 'n_modes': 2,
                       'sweep_lengths': (12.0, 18.0)}),
    # Builds its own fork from an outline, so it takes no domain.
    Demo('modal', demo_modal, section=SOLIDS,
         smoke_kwargs={'n_across_tine': 3, 'min_angle': 25, 'n_modes': 4, 'n_shown': 3,
                       'sweep_lengths': (0.088, 0.125), 'n_frames': 6}),
    # A 4:1 simply supported (MBB) beam, the aspect that optimizes into the classic arch.
    # `smoothing_radius` is a physical length, so a finer mesh resolves the same structure
    # rather than growing thinner members.
    Demo('topology_optimization', demo_topology_optimization, section=SOLIDS,
         domain=partial(beam, 4.0, 1.0, 160), smoke_kwargs={'iters': 3}),

    # Meshed coarse, so sin(40 r^2)'s slow inner rings resolve but the fast
    # outer ones alias into the triangulation.
    Demo('l2_projection', demo_l2_projection, section=ACCURACY, domain=partial(square, 28),
         smoke_kwargs={'reference_resolution': 60}),
    # Builds its own refinement sequence; the smoke run keeps the two coarsest meshes.
    Demo('convergence', demo_convergence, section=ACCURACY,
         smoke_kwargs={'resolutions': (11, 21), 'elastic_resolutions': (9, 17),
              'step_counts': (16, 32)}),
]
