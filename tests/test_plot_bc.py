"""What a boundary-conditions panel is able to say.

The panel is a demo's only claim about what was imposed, so a condition it cannot draw
is a condition a reader has to take on faith from the prose beside it. These check that
each kind reaches the axes, by legend label rather than by pixels -- the label is what
tells a reader which mark is which, so a mark drawn without one is not much better than
a mark not drawn at all.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from matplotlib.quiver import Quiver

from fem.boundary import BCType, BoundaryConditions
from fem.mesh.structured import create_rect_mesh
from fem.plot.bc import overlay_supports, plot_bc
from fem.regions import everywhere, intersect, on_plane


@pytest.fixture
def mesh():
    return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))


def _labels(mesh, bc):
    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)
    return labels


def test_robin_conditions_are_drawn(mesh):
    """`plot_bc` branched on Dirichlet and Neumann and fell through silently for Robin,
    so the one demo about a boundary condition could not draw its own."""
    bc = BoundaryConditions()
    bc.add_robin(everywhere(), kappa=2.0, g=1.0)
    assert any(label.startswith('Robin') for label in _labels(mesh, bc))


def test_a_robin_label_gives_the_coefficient_and_the_ambient_value(mesh):
    """Drawn only as a coloured edge, the two numbers the condition is about -- how
    freely it exchanges and what with -- appear nowhere on the panel."""
    bc = BoundaryConditions()
    bc.add_robin(everywhere(), kappa=0.5, g=0.5*300.0)
    assert 'Robin: du/dn + 0.5(u - 300) = 0' in _labels(mesh, bc)


def test_a_dirichlet_label_gives_the_value_it_pins_to(mesh):
    """A clamp and an imposed 50% stretch were the same picture: two rows of red dots."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [0.5, 0])
    labels = _labels(mesh, bc)
    assert 'Dirichlet: u = (0, 0)' in labels
    assert 'Dirichlet: u = (0.5, 0)' in labels


def test_a_displacement_pinned_away_from_zero_is_drawn_as_arrows(mesh):
    """Held at zero and dragged somewhere are different physics; dots alone say neither."""
    from matplotlib.quiver import Quiver

    def arrows(value):
        bc = BoundaryConditions()
        bc.add(BCType.DIRICHLET, on_plane(0, 1.0), value)
        fig, ax = plt.subplots()
        plot_bc(ax, mesh, bc)
        count = sum(isinstance(c, Quiver) for c in ax.collections)
        plt.close(fig)
        return count

    assert arrows([0.5, 0]) == 1
    assert arrows([0, 0]) == 0


def test_a_scalar_flux_is_drawn_as_a_run_not_an_arrow(mesh):
    """A scalar Neumann acts along a normal this panel does not draw, so an arrow would
    have to point somewhere; it also used to index a second component that is not there."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), 3.0)

    from matplotlib.quiver import Quiver

    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    assert 'Neumann: du/dn = 3' in ax.get_legend_handles_labels()[1]
    assert not any(isinstance(c, Quiver) for c in ax.collections)
    plt.close(fig)


def test_a_vector_traction_still_gets_arrows(mesh):
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])

    fig, ax = plt.subplots()
    plot_bc(ax, mesh, bc)
    from matplotlib.quiver import Quiver
    assert any(isinstance(c, Quiver) for c in ax.collections)
    plt.close(fig)


def test_boundary_carrying_no_condition_is_drawn_as_natural(mesh):
    """Saying nothing about an edge is a statement -- the natural condition of the weak
    form -- and `stress_concentration` exists to make exactly that point about its rim."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    assert 'Natural: t = 0' in _labels(mesh, bc)


def test_a_fully_constrained_boundary_claims_nothing_is_natural(mesh):
    """The free run is self-suppressing: it must not appear where every facet is spoken for."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    assert not any(label.startswith('Natural') for label in _labels(mesh, bc))


def test_repeated_conditions_of_one_kind_get_one_legend_entry(mesh):
    """`robin` sweeps the same kind of condition over several regions."""
    bc = BoundaryConditions()
    for side in (0.0, 1.0):
        bc.add(BCType.DIRICHLET, on_plane(0, side), 0.0)
    assert _labels(mesh, bc).count('Dirichlet: u = 0') == 1


def test_a_roller_reads_as_free_rather_than_nan(mesh):
    """A roller's free component must read as 'free', not the literal NaN it is
    internally -- and must not be mistaken for a displacement being imposed."""
    from matplotlib.quiver import Quiver

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, None])
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
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    fig, ax = _overlay(mesh, bc)
    assert _walls(ax)
    assert not _has_arrows(ax)
    plt.close(fig)


def test_a_pin_overlays_as_triangles_not_a_wall(mesh):
    """A roller edge (a component left free) is a pin: support triangles, no wall."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [None, 0])
    fig, ax = _overlay(mesh, bc)
    assert _triangles(ax)
    assert not _walls(ax)
    assert not _has_arrows(ax)
    plt.close(fig)


def test_a_traction_overlays_as_load_arrows(mesh):
    bc = BoundaryConditions()
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [-1.0, 0])
    fig, ax = _overlay(mesh, bc)
    assert _has_arrows(ax)
    plt.close(fig)


def test_a_driven_end_overlays_as_a_wall_and_an_arrow(mesh):
    """An imposed nonzero displacement both clamps and pushes: a wall with a load arrow,
    which is what tells it apart from a plain clamp."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [-0.3, 0])
    fig, ax = _overlay(mesh, bc)
    assert _walls(ax)
    assert _has_arrows(ax)
    plt.close(fig)


def test_a_single_anchor_point_overlays_as_a_dot_not_a_support(mesh):
    """The lone point that ties off a pinned column's axial slide is scaffolding, so it is
    a small marker -- not a wall or triangles competing with the supports on the ends."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, intersect(on_plane(0, 0.0), on_plane(1, 0.5)), [0, 0])
    fig, ax = _overlay(mesh, bc)
    assert not _walls(ax)
    assert not _triangles(ax)
    plt.close(fig)


# -- the unification's two laws: colour is the type, shape is the role ------------------


def test_colour_encodes_the_weak_form_type(mesh):
    """First law: colour is the weak-form type -- Dirichlet blue (held), Neumann red
    (applied) -- so the panel and the overlay speak one language rather than two."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])   # clamp -> blue wall
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [1.0, 0])   # traction -> red arrows
    fig, ax = _overlay(mesh, bc)

    walls = _walls(ax)
    assert walls
    assert all(np.allclose(wall.get_edgecolor(), to_rgba('tab:blue')) for wall in walls)

    arrows = [c for c in ax.collections if isinstance(c, Quiver)]
    assert arrows
    assert all(np.allclose(arrow.get_facecolor()[0], to_rgba('red')) for arrow in arrows)
    plt.close(fig)


def test_shape_is_the_role_only_where_the_components_allow(mesh):
    """Second law: shape is the mechanical role, as specific as the field permits. A vector
    clamp on an edge is a wall; the same edge in a scalar field -- which has no clamp or
    roller -- stays a row of dots, no drafting glyph."""
    vector = BoundaryConditions()
    vector.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    fig, ax = _overlay(mesh, vector)
    assert _walls(ax)
    plt.close(fig)

    scalar = BoundaryConditions()
    scalar.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)
    fig, ax = _overlay(mesh, scalar)
    assert not _walls(ax)
    assert not _triangles(ax)
    plt.close(fig)
