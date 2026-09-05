"""The figures of the adaptive refinement demo, drawn from a `RefinementStudy`."""
from functools import partial

from demo_registry import Demo, DemoResult, Figure

from demos.refinement import physics
from demos.refinement.physics import RefinementStudy, run
from fem.mesh.structured import box_mesh
from fem.plot.plotter import Plotter


def _error_clim(s: RefinementStudy) -> tuple[float, float]:
    """A shared log scale for both error plots."""
    min_error = min(s.coarse_error.min(), s.refined_error.min())
    max_error = max(s.coarse_error.max(), s.refined_error.max())
    return (max(min_error, 1e-10), max_error)


def _before_figure(s: RefinementStudy) -> Figure:
    before = Plotter(1, 3, title=f'Before: uniform mesh ({s.n_coarse} elements)')
    before.plot(s.coarse_mesh, s.coarse_solution.dofs, mode='surface', title='Solution',
                idx=(0, 0))
    before.plot(s.coarse_mesh, mode='mesh', title='Mesh', idx=(0, 1))
    before.plot(s.coarse_mesh, s.coarse_error, mode='colored',
                title=f'Error η (max: {s.coarse_max:.3f}, ‖η‖: {s.coarse_norm:.2f})',
                idx=(0, 2), clim=_error_clim(s), cmap='YlOrRd', log_scale=True)
    return Figure(
        before,
        'Uniform mesh with a posteriori error estimate η. The estimator bounds '
        'the local discretization error; high values (red) indicate where the '
        'mesh under-resolves the solution.',
        'before')


def _after_figure(s: RefinementStudy) -> Figure:
    after = Plotter(1, 3, title=f'After: adaptive refinement ({s.n_refined} elements)')
    after.plot(s.refined_mesh, s.refined_solution.dofs, mode='surface', title='Solution',
               idx=(0, 0))
    after.plot(s.refined_mesh, mode='mesh', title='Mesh', idx=(0, 1))
    after.plot(s.refined_mesh, s.refined_error, mode='colored',
               title=f'Error η (max: {s.refined_max:.3f}, ‖η‖: {s.refined_norm:.2f})',
               idx=(0, 2), clim=_error_clim(s), cmap='YlOrRd', log_scale=True)
    return Figure(
        after,
        f'After adaptive refinement driven by η: max error dropped {s.max_reduction:.0f}% '
        f'({s.coarse_max:.3f} → {s.refined_max:.3f}), ‖η‖ dropped {s.norm_reduction:.0f}% '
        f'({s.coarse_norm:.2f} → {s.refined_norm:.2f}).',
        'after')


def _payoff_figure(s: RefinementStudy) -> Figure:
    """The payoff: error against cost, refining uniformly and adaptively."""
    payoff = Plotter(title='Error against cost')
    chart = payoff.chart_ax(xlabel='degrees of freedom', ylabel='L2 error')
    chart.loglog(s.uniform_dofs, s.uniform_errors, 'o-', color='tab:blue', label='uniform')
    chart.loglog(s.adaptive_dofs, s.adaptive_errors, '.-', color='tab:red', label='adaptive')
    chart.set_title('Uniform against adaptive refinement')
    chart.grid(True, which='both', alpha=0.3)
    chart.legend()
    return Figure(
        payoff,
        'The error against the number of unknowns, refining the whole mesh (blue) '
        'and refining where the estimator points (red). The peak is small next to '
        'the domain, so uniform refinement spends most of its unknowns where the '
        'solution is already flat; the adaptive mesh reaches the same error with '
        'about a third as many.',
        'payoff')


def _machinery_figure(s: RefinementStudy) -> Figure:
    machinery = Plotter(1, 2, title='Red-green refinement', axis_labels=False)
    machinery.plot(s.red_green_original, mode='mesh', idx=(0, 0), title='Original')
    machinery.plot(s.red_green_refined, values=s.red_green_classes, mode='refinement',
                   idx=(0, 1), title='Refined (red / green)')
    return Figure(
        machinery,
        'Red splits an element into four; green bisects a neighbour so '
        'the mesh stays conforming.',
        'red-green')


def demo(_mesh, **kwargs) -> DemoResult:
    """Adaptive refinement driven by an error estimator on a peaked Poisson source,
    against uniform refinement at the same cost."""
    s = run(_mesh, **kwargs)
    return DemoResult([
        _payoff_figure(s),
        _before_figure(s),
        _after_figure(s),
        _machinery_figure(s),
    ])


DEMO = Demo('refinement', demo, section='Accuracy & performance',
            domain=partial(box_mesh, [[0.0, 0.0], [1.0, 1.0]], (40, 40)),
            show_source=physics,
            smoke_kwargs={'uniform_resolutions': (10, 20), 'adaptive_rounds': 2})
