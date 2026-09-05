"""The figures of the harbor breakwater demo, drawn from a `HarborStudy`."""
import numpy as np
from demo_registry import Demo, DemoResult, Figure

from demos.wave import physics
from demos.wave.physics import HarborStudy, run
from fem.plot.plotter import Plotter


def _snapshot_steps(s: HarborStudy, n_shown) -> list[int]:
    """The steps the snapshot panels show, spread over the run once the front is under way."""
    n = len(s.u_values)
    return [int(i) for i in np.linspace(n // 8, n - 1, n_shown)]


def _colour_limits(s: HarborStudy, shown) -> tuple[float, float]:
    """One colour scale, set by the harbor side, so the diffracted wave reads even
    though it is far lower than the front that made it (which doubles again when it
    reflects off the far wall)."""
    span = float(max(abs(s.u_values[i][s.harbor]).max() for i in shown))
    return (-span, span)


def _animation_figure(s: HarborStudy, clim) -> Figure:
    animation = Plotter(1, 1, figsize=(7.4, 4.8))
    animation.plot_animation(s.mesh, s.u_values, mode='colored', clim=clim, label='height',
                             cmap='RdBu_r',
                             titles=[f'Harbor breakwater  t={t:.2f}' for t in s.t_values],
                             idx=(0, 0))
    return Figure(animation, 'Newmark time integration of the front.', 'animation')


def _snapshots_figure(s: HarborStudy, shown, clim) -> Figure:
    snapshots = Plotter(2, 4, figsize=(18.0, 6.4), title='Diffraction through the gap')
    for panel, i in enumerate(shown):
        snapshots.plot(s.mesh, s.u_values[i], mode='colored', idx=divmod(panel, 4),
                       title=f't={s.t_values[i]:.2f}', clim=clim, colorbar=panel == 7,
                       cmap='RdBu_r', label='height')
    return Figure(
        snapshots,
        'The front reaches the breakwater, reflects off the wall, and passes the '
        'gap, where it spreads into the harbor as a circular wave centred on the '
        'opening, lower than the front that made it. The later frames show that wave '
        'reflecting around the harbor while the front, reflected off the wall and '
        'then the far edge, comes back through the gap.',
        'snapshots')


def _setup_figure(s: HarborStudy) -> Figure:
    setup = Plotter(1, 3, figsize=(15.0, 3.8))
    setup.plot(s.mesh, mode='mesh', idx=(0, 0), title='Basin and breakwater')
    setup.plot(s.mesh, s.u_initial, mode='colored', idx=(0, 1), label='height',
               title='Initial height u(x, 0)')
    setup.plot(s.mesh, s.dudt_initial, mode='colored', idx=(0, 2), label='velocity',
               title='Initial velocity, a front moving right')
    return Figure(
        setup,
        'A basin with a breakwater across it, open on the left and sheltered on the '
        'right. The initial height and velocity together make a front travelling '
        'right; every edge is a wall, reflecting the wave the same way up.',
        'conditions', setup=True)


def demo(**kwargs) -> DemoResult:
    """A wave front meeting a harbor breakwater, diffracting through its gap into the
    sheltered water behind."""
    s = run(**kwargs)
    shown = _snapshot_steps(s, 8)
    clim = _colour_limits(s, shown)
    return DemoResult([
        _animation_figure(s, clim),
        _snapshots_figure(s, shown, clim),
        _setup_figure(s),
    ])


# Builds its own harbor basin, so it takes no domain.
DEMO = Demo('wave', demo, section='Meshing & solving PDEs',
            show_source=physics,
            smoke_kwargs={'steps': 6, 'max_area': 0.5, 'uniform_rounds': 0})
