"""Regressions for `Plotter` writing to, and decorating, the wrong thing.

Each of these was silent: the output existed, looked plausible, and was wrong. They
are grouped because they share a cause -- reaching for pyplot's global current figure
and axes instead of the ones the Plotter owns.
"""
from pathlib import Path

import numpy as np
import pytest

from fem.mesh.ruppert import create_rect_mesh
from fem.plot.plotter import Plotter


@pytest.fixture
def mesh():
    return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(4, 4))


def _pixel_size(fig):
    width, height = fig.get_size_inches()
    return round(width * fig.dpi), round(height * fig.dpi)


def test_save_writes_its_own_figure_not_the_current_one(mesh, tmp_path):
    """`plt.savefig` writes whichever figure was created last, so a caller holding two
    Plotters wrote the second one under both names. Distinguished by figure shape: a
    1x1 Plotter and a 2x2 one have different pixel dimensions."""
    from PIL import Image

    first = Plotter(1, 1)
    first.plot(mesh, mode='mesh')
    second = Plotter(2, 2)          # created later, so pyplot's "current" figure
    second.plot(mesh, mode='mesh')

    path = tmp_path / 'first.png'
    first.save(str(path))

    assert Image.open(path).size == _pixel_size(first.fig), (
        'save() wrote a figure of the wrong shape -- it saved the current figure'
    )
    assert _pixel_size(first.fig) != _pixel_size(second.fig), 'test cannot distinguish them'


def test_close_closes_its_own_figure(mesh):
    import matplotlib.pyplot as plt

    first = Plotter(1, 1)
    second = Plotter(1, 1)
    first.close()

    open_figures = plt.get_fignums()
    assert second.fig.number in open_figures, 'close() closed the wrong figure'
    assert first.fig.number not in open_figures
    plt.close('all')


def test_surface_animation_adds_no_colorbar(mesh):
    """A colorbar was set up for every mode, but only the colored mode reads one --
    and for a surface it landed on the 2D axes that get replaced by 3D ones, so it
    survived as a legend attached to nothing."""
    values = [np.linspace(0, 1, len(mesh.vertices)) * k for k in (1.0, 2.0)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='surface')

    assert len(plotter.fig.axes) == 1, 'a surface animation grew an extra colorbar axes'
    plotter.close()


def test_colored_animation_scales_to_the_data(mesh):
    """cbar_lims defaulted to (0, 1); a 300 K field against that renders as one
    saturated block. Absent an explicit range it now spans the series."""
    values = [np.full(len(mesh.vertices), 300.0), np.full(len(mesh.vertices), 350.0)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored')

    norm = plotter.cbar_infos[(0, 0)].norm
    assert (norm.vmin, norm.vmax) == (300.0, 350.0)
    plotter.close()


def test_surface_axes_are_labelled_in_z(mesh):
    """`set_label` sets an artist's legend entry, not an axis; a 3D panel went its whole
    life without a z label because that was called instead of `set_zlabel`."""
    values = np.linspace(0, 1, len(mesh.vertices))

    plotter = Plotter(1, 1)
    plotter.plot(mesh, values, mode='surface')
    plotter.format_axs()

    assert plotter.axs[0, 0].get_zlabel() == 'z'
    plotter.close()


def test_axis_labels_can_be_turned_off(mesh):
    """An outline in SVG user units gains nothing from being told its axes are x and y."""
    plotter = Plotter(1, 1, axis_labels=False)
    plotter.plot(mesh, mode='mesh')
    plotter.format_axs()

    ax = plotter.axs[0, 0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('', '')
    plotter.close()


def test_colorbar_carries_the_quantity_it_shows(mesh):
    values = np.linspace(300.0, 350.0, len(mesh.vertices))

    plotter = Plotter(1, 1)
    plotter.plot(mesh, values, mode='colored', label='temperature')

    # The colorbar is the second axes on the figure; its label is the y-axis label.
    assert plotter.fig.axes[1].get_ylabel() == 'temperature'
    plotter.close()


def test_colorbar_matches_the_height_of_the_panel_it_annotates():
    """Constrained layout sizes a colorbar to the subplot *cell*; `set_aspect('equal')`
    then shrinks the axes inside that cell. On a flat domain the bar ended up around
    three times the height of the plot it was annotating.

    Measured after `format_axs`, which is what every save and show path calls.
    """
    wide = create_rect_mesh(corners=[[0, 0], [4, 1]], resolution=(8, 4))
    values = np.linspace(0.0, 1.0, len(wide.vertices))

    plotter = Plotter(1, 1, panel_aspect=4.0)
    plotter.plot(wide, values, mode='colored', label='v')
    plotter.format_axs()

    panel = plotter.axs[0, 0].get_position()
    bar = plotter.cbar_infos[(0, 0)].bar.ax.get_position()
    assert bar.height == pytest.approx(panel.height, rel=0.02), (
        f'colorbar is {bar.height / panel.height:.1f}x the height of its panel'
    )
    assert bar.y0 == pytest.approx(panel.y0, abs=0.02), 'colorbar is not aligned with it'
    plotter.close()


def test_frames_can_be_sampled_down_to_a_cap(mesh, tmp_path):
    """Rasterizing frames is the largest cost of a gallery build, and a player does not
    need one image per solver step. The last frame is never the one dropped -- for a
    topology optimization it is the result."""
    values = [np.full(len(mesh.vertices), float(k)) for k in range(10)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored')
    written = plotter.save_frames(str(tmp_path / '{:03d}.png'), max_frames=4)

    assert len(written) == 4
    # Contiguously numbered, so the player steps through them without gaps.
    assert [Path(p).name for p in written] == ['000.png', '001.png', '002.png', '003.png']
    assert all(Path(p).exists() for p in written)

    # The last image is the last *frame*, not the fourth of ten: compared against the
    # same figure writing the run in full, whose final image is frame 9 by definition.
    full = plotter.save_frames(str(tmp_path / 'full{:03d}.png'))
    assert Path(written[-1]).read_bytes() == Path(full[-1]).read_bytes()
    plotter.close()


def test_uncapped_frames_write_every_step(mesh, tmp_path):
    values = [np.full(len(mesh.vertices), float(k)) for k in range(5)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored')
    assert len(plotter.save_frames(str(tmp_path / '{:03d}.png'))) == 5
    plotter.close()


def test_a_chart_panel_keeps_its_own_labels_and_scale(mesh):
    """Domain formatting applied to a log-log plot squashes it to equal aspect, labels
    its axes x and y, and `ticklabel_format` raises outright on a log scale."""
    plotter = Plotter(1, 2)
    plotter.plot(mesh, mode='mesh', idx=(0, 0))
    ax = plotter.chart_ax(idx=(0, 1), xlabel='h', ylabel='L2 error')
    ax.loglog([0.1, 0.05], [1e-2, 2.5e-3])

    plotter.format_axs()   # raised here before chart panels were exempt

    assert (ax.get_xlabel(), ax.get_ylabel()) == ('h', 'L2 error')
    assert ax.get_aspect() == 'auto'
    # The domain panel beside it is unaffected.
    assert plotter.axs[0, 0].get_aspect() == 1.0
    plotter.close()


def test_a_conditions_panel_keeps_its_aspect_but_not_the_x_y_labels(mesh):
    """`plot_bc` puts a legend under the panel, which is where the words x and y sit.
    The panel is still a picture of the domain, so it keeps equal aspect."""
    from fem.boundary import BCType, BoundaryConditions
    from fem.regions import on_plane

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)

    plotter = Plotter(1, 2)
    plotter.plot(mesh, mode='bc', bc=bc, idx=(0, 0))
    plotter.plot(mesh, mode='mesh', idx=(0, 1))
    plotter.format_axs()

    conditions, plain = plotter.axs[0, 0], plotter.axs[0, 1]
    assert (conditions.get_xlabel(), conditions.get_ylabel()) == ('', '')
    assert conditions.get_aspect() == 1.0
    # Every other domain panel is unaffected.
    assert (plain.get_xlabel(), plain.get_ylabel()) == ('x', 'y')
    plotter.close()


def test_clim_holds_one_scale_across_panels(mesh):
    """Each panel renormalized to its own extremes, so six snapshots of a field decaying
    by 70% drew as six near-identical squares -- the run visible only in the colorbar
    ticks. `plot_animation` had always fixed this across frames; `plot` could not."""
    cool = np.full(len(mesh.vertices), 300.0)
    cool[0] = 314.0
    hot = np.full(len(mesh.vertices), 300.0)
    hot[0] = 350.0

    plotter = Plotter(1, 2)
    for i, values in enumerate((hot, cool)):
        plotter.plot(mesh, values, mode='colored', idx=(0, i), clim=(300.0, 350.0))

    for idx in ((0, 0), (0, 1)):
        norm = plotter.cbar_infos[idx].norm
        assert (norm.vmin, norm.vmax) == (300.0, 350.0)
    plotter.close()


def test_without_clim_each_panel_still_scales_to_itself(mesh):
    """Sharing is opt-in: a grid of unrelated quantities must not be forced onto one."""
    values = np.linspace(0.0, 1.0, len(mesh.vertices))

    plotter = Plotter(1, 2)
    plotter.plot(mesh, values, mode='colored', idx=(0, 0))
    plotter.plot(mesh, 10*values, mode='colored', idx=(0, 1))

    assert plotter.cbar_infos[(0, 0)].norm.vmax == pytest.approx(1.0)
    assert plotter.cbar_infos[(0, 1)].norm.vmax == pytest.approx(10.0)
    plotter.close()


def test_clim_fixes_the_z_axis_of_a_surface_too(mesh):
    """A grid of surfaces autoscales each to its own height, so a pulse that has spread
    out is drawn the same size as one that has not."""
    plotter = Plotter(1, 1)
    plotter.plot(mesh, np.linspace(0.0, 0.1, len(mesh.vertices)), mode='surface',
                 clim=(-1.0, 1.0))
    assert plotter.axs[0, 0].get_zlim() == (-1.0, 1.0)
    plotter.close()


def test_explicit_colorbar_limits_are_respected(mesh):
    values = [np.full(len(mesh.vertices), 300.0), np.full(len(mesh.vertices), 350.0)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored', cbar_lims=(0.0, 400.0))

    norm = plotter.cbar_infos[(0, 0)].norm
    assert (norm.vmin, norm.vmax) == (0.0, 400.0)
    plotter.close()
