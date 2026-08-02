"""Regressions for `Plotter` writing to, and decorating, the wrong thing.

Each of these was silent: the output existed, looked plausible, and was wrong. They
are grouped because they share a cause -- reaching for pyplot's global current figure
and axes instead of the ones the Plotter owns.
"""
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

    _cmap, norm = plotter.cbar_infos[(0, 0)]
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


def test_explicit_colorbar_limits_are_respected(mesh):
    values = [np.full(len(mesh.vertices), 300.0), np.full(len(mesh.vertices), 350.0)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored', cbar_lims=(0.0, 400.0))

    _cmap, norm = plotter.cbar_infos[(0, 0)]
    assert (norm.vmin, norm.vmax) == (0.0, 400.0)
    plotter.close()
