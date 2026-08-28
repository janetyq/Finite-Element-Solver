"""What a boundary-conditions panel is able to say. Each kind of condition must reach
the axes, checked by legend label rather than by pixels.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from matplotlib.quiver import Quiver

from fem.boundary import BoundaryConditions, Dirichlet, Neumann, Robin
from fem.mesh.structured import box_mesh
from fem.plot.bc import overlay_supports, plot_bc
from fem.regions import everywhere, intersect, on_plane


@pytest.fixture
def mesh():
    return box_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))


def _labels(mesh, bc):
    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)
    return labels


def test_robin_conditions_are_drawn(mesh):
    """A Robin condition is drawn."""
    bc = BoundaryConditions(
        Robin(everywhere(), kappa=2.0, g=1.0),
    )
    assert any(label.startswith('Robin') for label in _labels(mesh, bc))


def test_a_robin_label_gives_the_coefficient_and_the_ambient_value(mesh):
    """A Robin condition's two numbers, how freely it exchanges and with what, appear in the
    legend."""
    bc = BoundaryConditions(
        Robin(everywhere(), kappa=0.5, g=0.5*300.0),
    )
    assert 'Robin: du/dn + 0.5(u - 300) = 0' in _labels(mesh, bc)


def test_a_dirichlet_label_gives_the_value_it_pins_to(mesh):
    """A clamp and an imposed 50% stretch were the same picture: two rows of red dots."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [0.5, 0]),
    )
    labels = _labels(mesh, bc)
    assert 'Dirichlet: u = (0, 0)' in labels
    assert 'Dirichlet: u = (0.5, 0)' in labels


def test_a_displacement_pinned_away_from_zero_is_drawn_as_arrows(mesh):
    """Held at zero and dragged somewhere are different physics; dots alone say neither."""
    from matplotlib.quiver import Quiver

    def arrows(value):
        bc = BoundaryConditions(
            Dirichlet(on_plane(0, 1.0), value),
        )
        fig, ax = plt.subplots()
        plot_bc(ax, mesh, bc)
        count = sum(isinstance(c, Quiver) for c in ax.collections)
        plt.close(fig)
        return count

    assert arrows([0.5, 0]) == 1
    assert arrows([0, 0]) == 0


def test_a_scalar_flux_is_drawn_as_a_run_not_an_arrow(mesh):
    """A scalar Neumann acts along a normal this panel does not draw, so it is a run."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), 0.0),
        Neumann(on_plane(0, 1.0), 3.0),
    )

    from matplotlib.quiver import Quiver

    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    assert 'Neumann: du/dn = 3' in ax.get_legend_handles_labels()[1]
    assert not any(isinstance(c, Quiver) for c in ax.collections)
    plt.close(fig)


def test_a_vector_traction_still_gets_arrows(mesh):
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )

    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    from matplotlib.quiver import Quiver
    assert any(isinstance(c, Quiver) for c in ax.collections)
    plt.close(fig)


def test_boundary_carrying_no_condition_is_drawn_as_natural(mesh):
    """Saying nothing about an edge is the natural condition of the weak form, and it is drawn."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
    )
    assert 'Natural: t = 0' in _labels(mesh, bc)


def test_a_fully_constrained_boundary_claims_nothing_is_natural(mesh):
    """The free run is self-suppressing: it must not appear where every facet is spoken for."""
    bc = BoundaryConditions(
        Dirichlet(everywhere(), 0.0),
    )
    assert not any(label.startswith('Natural') for label in _labels(mesh, bc))


def test_repeated_conditions_of_one_kind_get_one_legend_entry(mesh):
    """A sweep repeats one kind of condition over several regions; the legend folds them
    into a single entry."""
    conditions = []
    for side in (0.0, 1.0):
        conditions.append(Dirichlet(on_plane(0, side), 0.0))
    bc = BoundaryConditions(*conditions)
    assert _labels(mesh, bc).count('Dirichlet: u = 0') == 1


def test_a_roller_reads_as_free_rather_than_nan(mesh):
    """A roller's free component reads as 'free', not the literal NaN it is internally."""
    from matplotlib.quiver import Quiver

    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, None]),
    )
    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)

    assert 'Dirichlet: u = (0, free)' in labels
    assert not any('nan' in label for label in labels)
    assert not any(isinstance(c, Quiver) for c in ax.collections)


# -- overlay_supports: the drafting glyphs laid over a (buckled) shape ------------------
#
# `plot_bc` draws the labelled conditions panel; `overlay_supports` draws only the
# symbols, for reading an end condition off a deformed shape. These check that each
# condition reaches the axes as the right *kind* of mark -- a clamp as a hatched wall, a
# pin as triangles, a load as arrows -- since that shape is the whole message, there
# being no legend to name it.


def _overlay(mesh, bc):
    fig, ax = plt.subplots()
    overlay_supports(ax, mesh, bc)
    return fig, ax


def _has_arrows(ax):
    return any(isinstance(c, Quiver) for c in ax.collections)


def _walls(ax):
    return [p for p in ax.patches if isinstance(p, Polygon) and p.get_hatch()]


def _triangles(ax):
    return [p for p in ax.patches if isinstance(p, Polygon) and not p.get_hatch()]


def test_a_clamp_overlays_as_a_hatched_wall(mesh):
    """A fully-fixed edge is a built-in end: a hatched wall, and no load arrow."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
    )
    fig, ax = _overlay(mesh, bc)
    assert _walls(ax)
    assert not _has_arrows(ax)
    plt.close(fig)


def test_a_pin_overlays_as_triangles_not_a_wall(mesh):
    """A roller edge (a component left free) is a pin: support triangles, no wall."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [None, 0]),
    )
    fig, ax = _overlay(mesh, bc)
    assert _triangles(ax)
    assert not _walls(ax)
    assert not _has_arrows(ax)
    plt.close(fig)


def test_a_traction_overlays_as_load_arrows(mesh):
    bc = BoundaryConditions(
        Neumann(on_plane(0, 1.0), [-1.0, 0]),
    )
    fig, ax = _overlay(mesh, bc)
    assert _has_arrows(ax)
    plt.close(fig)


def test_a_driven_end_overlays_as_a_wall_and_an_arrow(mesh):
    """An imposed nonzero displacement both clamps and pushes: a wall with a load arrow."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 1.0), [-0.3, 0]),
    )
    fig, ax = _overlay(mesh, bc)
    assert _walls(ax)
    assert _has_arrows(ax)
    plt.close(fig)


def test_a_single_anchor_point_overlays_as_a_dot_not_a_support(mesh):
    """The lone point that ties off a pinned column's axial slide is a small marker, not a
    wall or triangles."""
    bc = BoundaryConditions(
        Dirichlet(intersect(on_plane(0, 0.0), on_plane(1, 0.5)), [0, 0]),
    )
    fig, ax = _overlay(mesh, bc)
    assert not _walls(ax)
    assert not _triangles(ax)
    plt.close(fig)


# -- the unification's two laws: colour is the type, shape is the role ------------------


def test_colour_encodes_the_weak_form_type(mesh):
    """Colour is the weak-form type: Dirichlet blue, Neumann red, in the panel and the overlay."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [1.0, 0]),
    )
    fig, ax = _overlay(mesh, bc)

    walls = _walls(ax)
    assert walls
    assert all(np.allclose(wall.get_edgecolor(), to_rgba('tab:blue')) for wall in walls)

    arrows = [c for c in ax.collections if isinstance(c, Quiver)]
    assert arrows
    assert all(np.allclose(arrow.get_facecolor()[0], to_rgba('red')) for arrow in arrows)
    plt.close(fig)


def test_shape_is_the_role_only_where_the_components_allow(mesh):
    """Shape is the mechanical role, as specific as the field permits: a vector clamp on an
    edge is a wall; the same edge in a scalar field stays a row of dots."""
    vector = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
    )
    fig, ax = _overlay(mesh, vector)
    assert _walls(ax)
    plt.close(fig)

    scalar = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), 0.0),
    )
    fig, ax = _overlay(mesh, scalar)
    assert not _walls(ax)
    assert not _triangles(ax)
    plt.close(fig)
