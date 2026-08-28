"""The figures of the airfoil potential-flow demo, drawn from a `FlowStudy`."""
from fem.plot.plotter import Plotter

from demo_registry import Demo, DemoResult, Figure
from demos.poisson import physics
from demos.poisson.physics import FlowStudy, run


def _flow_figure(s: FlowStudy) -> Figure:
    # `space=solution.space` opts both panels onto the P2 tessellation: the potential
    # shows its within-element curvature and the recovered speed draws smoothly.
    plotter = Plotter(1, 2, title='Potential flow over an airfoil', panel_aspect=1.8)
    plotter.plot(s.mesh, s.solution.u, mode='colored', idx=(0, 0), label='velocity potential',
                 title='Potential and its equipotentials', contour=22, space=s.solution.space)
    plotter.plot(s.mesh, s.speed, mode='colored', idx=(0, 1), label='flow speed',
                 clim=(0.0, s.speed_cap), title='Flow speed (clipped near the edges)',
                 space=s.solution.space)
    return Figure(
        plotter,
        'Ideal (irrotational, incompressible) flow over a NACA 2412 airfoil at a '
        f'{s.angle_of_attack:g}-degree angle of attack, generated from the standard '
        'formula rather than a data file. Left: the velocity potential phi (Laplace) '
        'with its equipotentials, which crowd over the upper surface where the flow '
        'speeds up. Right: the flow speed, faster over the top than the bottom, with '
        'stagnation near the leading and trailing edges. The wing takes no condition '
        'at all, which in the weak form is zero flux, so it is a streamline the flow '
        'parts around. The speed is clipped near the sharp edges, where ideal flow '
        'with no Kutta condition predicts an unphysical velocity spike.',
        'flow')


def _conditions_figure(s: FlowStudy) -> Figure:
    conditions = Plotter(panel_aspect=1.8)
    conditions.plot(s.mesh, mode='bc', bc=s.bc)
    return Figure(
        conditions,
        'A potential difference across the channel (phi = 0 at the inlet, 1 at the '
        'outlet) drives the flow left to right; the walls and the wing surface take no '
        'condition, so no flow crosses them.',
        'conditions', setup=True)


def demo(**kwargs) -> DemoResult:
    """Poisson's equation as potential flow over a NACA airfoil, on P2 elements."""
    s = run(**kwargs)
    return DemoResult([_flow_figure(s), _conditions_figure(s)])


# Builds its own airfoil-in-a-channel from the NACA formula, so it takes no domain.
DEMO = Demo('poisson', demo, section='Meshing & solving PDEs',
            show_source=physics,
            smoke_kwargs={'n_points': 40, 'max_area_fraction': 0.02})
