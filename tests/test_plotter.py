"""`Plotter` writes to, and decorates, its own figure and axes rather than pyplot's
current ones, and its P2 tessellation and warp draw what they claim.
"""
from pathlib import Path

import numpy as np
import pytest

from fem.convergence import ANNULUS_INNER, ANNULUS_OUTER, create_annulus_mesh
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.mesh.structured import create_rect_mesh
from fem.plot.plotter import Plotter
from fem.space import FunctionSpace


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
    """A surface animation draws no colorbar."""
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


def test_contour_overlays_isolines_on_a_colored_panel(mesh):
    """`contour=n` draws level-set isolines over the flat colouring; without it the
    panel carries only the tripcolor collection."""
    values = np.linspace(0.0, 1.0, len(mesh.vertices))

    plain = Plotter(1, 1)
    plain.plot(mesh, values, mode='colored')
    with_lines = Plotter(1, 1)
    with_lines.plot(mesh, values, mode='colored', contour=6)

    assert len(with_lines.axs[0, 0].collections) > len(plain.axs[0, 0].collections), (
        'contour=6 added no isoline collections over the colored panel'
    )
    plain.close()
    with_lines.close()


def test_colorbar_matches_the_height_of_the_panel_it_annotates():
    """Constrained layout sizes a colorbar to the subplot cell; `set_aspect('equal')` then
    shrinks the axes inside it. Measured after `format_axs`."""
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
    """Frames are sampled down to `max_frames`, and the last frame is never the one dropped."""
    values = [np.full(len(mesh.vertices), float(k)) for k in range(6)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored')
    written = plotter.save_frames(str(tmp_path / '{:03d}.png'), max_frames=3)

    assert len(written) == 3
    # Contiguously numbered, so the player steps through them without gaps.
    assert [Path(p).name for p in written] == ['000.png', '001.png', '002.png']
    assert all(Path(p).exists() for p in written)

    # The last image is the last *frame*, not the third of six: compared against the
    # same figure writing the run in full, whose final image is frame 5 by definition.
    full = plotter.save_frames(str(tmp_path / 'full{:03d}.png'))
    assert Path(written[-1]).read_bytes() == Path(full[-1]).read_bytes()
    plotter.close()


def test_save_gif_writes_a_looping_file_with_the_sampled_frames(mesh, tmp_path):
    from PIL import Image

    values = [np.full(len(mesh.vertices), float(k)) for k in range(6)]
    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored')
    plotter.save_gif(str(tmp_path / 'run.gif'), max_frames=3)
    plotter.close()

    with Image.open(tmp_path / 'run.gif') as gif:
        assert gif.n_frames == 3
        assert gif.info.get('loop') == 0


def test_uncapped_frames_write_every_step(mesh, tmp_path):
    values = [np.full(len(mesh.vertices), float(k)) for k in range(3)]

    plotter = Plotter(1, 1)
    plotter.plot_animation(mesh, values, mode='colored')
    assert len(plotter.save_frames(str(tmp_path / '{:03d}.png'))) == 3
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
    from fem.boundary import BoundaryConditions, Dirichlet
    from fem.regions import on_plane

    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), 0.0),
    )

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
    """`clim` holds one colour scale across panels, so a decaying field reads as decaying."""
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


# -- curved / P2-aware rendering --------------------------------------------------
#
# A curved space places its boundary nodes on the true curve and its field varies
# quadratically within each element, neither of which the straight P1 arrays a plot
# consumes can show. These check the display tessellation the plot layer draws instead:
# that its sub-triangles sit on the true geometry, that it interpolates a field faithfully,
# and that the plot path actually reaches for it.


def _annulus_spaces(n=9):
    mesh = create_annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, n, 4 * n)
    return (mesh,
            FunctionSpace(mesh, QuadraticTriangleElement, n_components=1),
            FunctionSpace(mesh, IsoparametricTriangleElement, n_components=1))


def _distance_off_rim(points):
    radius = np.hypot(points[:, 0], points[:, 1])
    return np.minimum(np.abs(radius - ANNULUS_INNER),
                      np.abs(radius - ANNULUS_OUTER)).max()


def test_tessellation_indices_and_size_are_consistent():
    """Every sub-triangle indexes a real point, and the count is one refined patch per
    element: `subdivisions**2` sub-triangles over `(subdivisions+1)(subdivisions+2)/2`
    reference points."""
    _, _, curved = _annulus_spaces()
    tess = curved.tessellation(subdivisions=3)

    n_el = len(curved.element_nodes)
    assert tess.triangles.shape == (n_el * 9, 3)
    assert tess.points.shape == (n_el * 10, 2)
    assert tess.triangles.min() == 0 and tess.triangles.max() == len(tess.points) - 1


def test_curved_boundary_polylines_follow_the_true_circle():
    """The defining property, at display resolution: an isoparametric facet's sampled
    polyline tracks the true rim, where a straight P2 facet is a chord well off it."""
    _, straight, curved = _annulus_spaces()

    curved_off = _distance_off_rim(curved.boundary_polylines(subdivisions=4).reshape(-1, 2))
    straight_off = _distance_off_rim(straight.boundary_polylines(subdivisions=4).reshape(-1, 2))

    assert curved_off < 1e-4, f'curved boundary sampled off the rim by {curved_off}'
    assert straight_off > 1e-3, 'straight P2 facets should sit visibly inside the rim'
    assert curved_off < straight_off / 20


def test_tessellation_reproduces_an_affine_field_on_a_curved_space():
    """`interpolate` samples the field through the same shape functions the geometry
    uses, so a field linear in x, which lies in the P2 span even through the quadratic
    map, is reproduced exactly at every sub-point."""
    _, _, curved = _annulus_spaces()
    tess = curved.tessellation(subdivisions=3)

    def affine(xy):
        return 2.0 * xy[:, 0] - 3.0 * xy[:, 1] + 1.5

    got = tess.interpolate(affine(curved.node_coords))
    assert np.abs(got - affine(tess.points)).max() < 1e-10


def test_tessellation_reproduces_a_quadratic_field_on_straight_p2():
    """On an affine (straight) P2 element the field interpolant of a quadratic is that
    quadratic, so the tessellation shows the within-element curvature exactly rather than
    the flat average one triangle per element would draw."""
    _, straight, _ = _annulus_spaces()
    tess = straight.tessellation(subdivisions=3)

    def quadratic(xy):
        x, y = xy[:, 0], xy[:, 1]
        return 1.0 + 2.0 * x - y + 0.5 * x**2 - x * y + 2.0 * y**2

    got = tess.interpolate(quadratic(straight.node_coords))
    assert np.abs(got - quadratic(tess.points)).max() < 1e-10


def test_colored_with_a_space_draws_a_denser_tessellation():
    """Passing `space` opts a P2 solve into the tessellated path: the colored panel is
    drawn on `subdivisions**2` sub-triangles per element instead of one, so the field's
    curvature and the curved boundary can show."""
    mesh, _, curved = _annulus_spaces()
    values = np.hypot(curved.node_coords[:, 0], curved.node_coords[:, 1])  # per-node field

    plain = Plotter(1, 1)
    p1_artist = plain.plot(mesh, mesh.vertices[:, 0], mode='colored')
    tessellated = Plotter(1, 1)
    curved_artist = tessellated.plot(mesh, values, mode='colored', space=curved)

    # subdivisions=3 splits each element into 9 sub-triangles, against one per element
    # on the P1 path.
    assert len(curved_artist.get_array()) == 9 * len(p1_artist.get_array())
    plain.close()
    tessellated.close()


def _p2_square(n=5):
    mesh = create_rect_mesh(corners=[[0, 0], [2, 1]], resolution=(n, n))
    return mesh, FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)


def test_surface_with_a_space_lifts_the_tessellated_field():
    """The surface mode is P2-aware: with a space it lifts the per-node field over the
    element tessellation, so a length-n_nodes field draws where the P1 path, which only
    knows per-vertex or per-element fields, would reject it outright."""
    mesh, space = _p2_square()
    field = space.node_coords[:, 0] ** 2                       # length n_nodes

    plotter = Plotter(1, 1)
    plotter.plot(mesh, field, mode='surface', idx=(0, 0), space=space)
    assert plotter.get_ax((0, 0)).has_data()
    plotter.close()

    with pytest.raises(ValueError, match='Invalid values shape'):
        Plotter(1, 1).plot(mesh, field, mode='surface')       # no space: P1 path rejects it


def test_arrows_with_a_space_draw_at_the_nodes():
    """A per-node vector field with a space is drawn at the nodes, not one arrow per
    element centroid, so a recovered flux is shown where it was recovered."""
    mesh, space = _p2_square(6)
    vectors = np.column_stack([space.node_coords[:, 1], -space.node_coords[:, 0]])

    plotter = Plotter(1, 1)
    plotter.plot(mesh, vectors, mode='arrows', idx=(0, 0), space=space)
    (quiver,) = plotter.get_ax((0, 0)).collections
    offsets = np.asarray(quiver.get_offsets())
    # Every arrow sits on a node (the subsample is drawn from node_coords, not centroids).
    distances = np.linalg.norm(offsets[:, None, :] - space.node_coords[None, :, :], axis=2)
    assert distances.min(axis=1).max() < 1e-9
    plotter.close()


def test_warp_tessellates_the_deformed_configuration():
    """`warp` maps the sub-points through the deformed node positions, so the tessellation
    is the reference one shifted by the interpolated displacement, and the colored panel
    of a P2 field then lands on the warped shape."""
    mesh, space = _p2_square()
    displacement = np.column_stack([0.1 * space.node_coords[:, 1],
                                    -0.05 * space.node_coords[:, 0]])

    reference = space.tessellation(subdivisions=3)
    deformed = space.tessellation(subdivisions=3, node_coords=space.node_coords + displacement)
    # Mapping through node + disp is mapping through node, plus the interpolated disp.
    assert np.allclose(deformed.points, reference.points + reference.interpolate(displacement))

    # And the colored panel accepts the warp and draws (on the deformed tessellation).
    plotter = Plotter(1, 1)
    plotter.plot(mesh, space.node_coords[:, 0], mode='colored', space=space, warp=displacement)
    assert plotter.get_ax().has_data()
    plotter.close()


# -- a Solution supplies its own space, so P2 rendering needs no space= ---------


def _p2_scalar_solution(n=5):
    """A ScalarFieldSolution on a P2 space, so its per-node field needs the space to draw."""
    from fem.solution import ScalarFieldSolution
    mesh, space = _p2_square(n)
    u = space.node_coords[:, 0] ** 2
    return ScalarFieldSolution(space, u, flux=np.zeros((len(mesh.elements), 2))), space


def test_a_solution_supplies_its_own_space():
    """Passing the solution draws its P2 field faithfully with no `space=`: the same
    per-node field on the bare mesh (no space) is rejected, so the solution is what
    carried the space that made it renderable."""
    solution, space = _p2_scalar_solution()
    field = solution.u                                         # length n_nodes

    plotter = Plotter(1, 1)
    plotter.plot(solution, field, mode='colored', idx=(0, 0))
    assert plotter.get_ax((0, 0)).has_data()
    plotter.close()

    with pytest.raises(ValueError):
        Plotter(1, 1).plot(solution.mesh, field, mode='colored')   # bare mesh, no space


def test_an_explicit_space_overrides_the_solutions():
    """A `space=` passed alongside a solution wins, so a caller can still draw the field
    on a different discretization."""
    solution, space = _p2_scalar_solution()
    plotter = Plotter(1, 1)
    plotter.plot(solution, solution.u, mode='colored', idx=(0, 0), space=space)
    assert plotter.get_ax((0, 0)).has_data()
    plotter.close()


def test_warp_true_deforms_by_the_solutions_own_displacement():
    """`warp=True` with a solution draws the field on the shape deformed by that
    solution's displacement, so an elastic field lands on the warped body with no explicit
    displacement array."""
    from fem.solution import ElasticSolution
    mesh, space = _p2_square()
    vspace = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)
    u = np.column_stack([0.1 * vspace.node_coords[:, 1],
                         -0.05 * vspace.node_coords[:, 0]]).ravel()
    n_el = len(mesh.elements)
    solution = ElasticSolution(vspace, u, strain=np.zeros((n_el, 3, 3)),
                               stress=np.zeros((n_el, 3, 3)), compliance=np.zeros(n_el))

    plotter = Plotter(1, 1)
    plotter.plot(solution, solution.nodal_von_mises(), mode='colored', idx=(0, 0), warp=True)
    assert plotter.get_ax((0, 0)).has_data()
    plotter.close()


def test_warp_true_needs_a_solution_not_a_bare_mesh():
    """`warp=True` has no displacement to read off a raw mesh, so it is rejected."""
    mesh, space = _p2_square()
    with pytest.raises(ValueError, match='warp=True needs a Solution'):
        Plotter(1, 1).plot(mesh, space.node_coords[:, 0], mode='colored', space=space, warp=True)


def test_refinement_plot_draws(mesh):
    """plot_refinement colours red/green leaves on a refined mesh."""
    from fem.mesh.refinement import RedGreenRefiner
    from fem.plot.helpers import plot_refinement

    refiner = RedGreenRefiner(mesh)
    refined = refiner.refine([0])
    ax = Plotter().get_ax()
    plot_refinement(ax, refined, refiner.leaf_classifications())
    assert ax.has_data()
