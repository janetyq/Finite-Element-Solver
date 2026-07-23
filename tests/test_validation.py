"""Tests for the guardrails around half-implemented / easily-misused surfaces.

These lock in that the gated features fail loudly (with a clear error) rather
than silently returning wrong or empty results.
"""
import numpy as np
import pytest

from fem.energies import NeohookeanEnergyDensity
from fem.energy_solver import EnergySolver
from fem.boundary import BoundaryConditions, BCType
from fem.mesh.mesh import Mesh
from fem.solver import LinearElastic
from fem.regions import everywhere, on_plane, at_indices
from fem.plot.plotter import PlotMode


def test_neohookean_is_gated():
    """The unfinished Neohookean material must raise, not silently do nothing."""
    density = NeohookeanEnergyDensity(E=200, nu=0.3)
    with pytest.raises(NotImplementedError):
        density.evaluate(np.zeros((1, 2, 2)))


def test_named_interior_vertex_is_rejected(make_unit_square):
    """Naming a node explicitly is a claim about that node, so an interior one is
    a modeling error. (A *geometric* region intersects with the boundary instead,
    which is why only the at_indices path can trip this.)"""
    mesh = make_unit_square(8)
    interior = (set(range(len(mesh.vertices))) - set(int(i) for i in mesh.boundary_idxs)).pop()

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, at_indices([interior]), 0)
    with pytest.raises(ValueError):
        bc.resolve(mesh, n_components=1)


def test_geometric_region_never_selects_interior_vertices(make_unit_square):
    """A plane cutting through the domain still yields only boundary DOFs: the
    old 'BC on a non-boundary vertex' error is now unrepresentable."""
    mesh = make_unit_square(9)  # odd, so x = 0.5 is a grid line
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.5), 0)  # a line through the middle

    interior_on_plane = np.isclose(mesh.vertices[:, 0], 0.5).sum() - 2
    assert interior_on_plane > 0, "region does not actually cross the interior"

    fixed = bc.resolve(mesh, n_components=1).fixed_idxs
    assert len(fixed) == 2  # only where the line meets the boundary
    assert set(fixed) <= set(int(i) for i in mesh.boundary_idxs)


def test_dirichlet_neumann_overlap_is_rejected(make_unit_square):
    """A vertex fixed by Dirichlet silently ignores its Neumann load; flag it."""
    mesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0)
    bc.add(BCType.NEUMANN, on_plane(0, 0.0), 0)
    with pytest.raises(ValueError):
        bc.resolve(mesh, n_components=1)


def test_conflicting_dirichlet_values_are_rejected(make_unit_square):
    """Overlapping regions are fine (a corner is on two edges); overlapping
    regions that disagree are a conflict, and last-write-wins would bury it."""
    mesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)   # left edge
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), 1.0)   # bottom edge; corner disagrees
    with pytest.raises(ValueError):
        bc.resolve(mesh, n_components=1)


def test_agreeing_overlapping_regions_are_fine(make_unit_square):
    mesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), 0.0)
    bc.add(BCType.DIRICHLET, on_plane(1, 0.0), 0.0)
    resolved = bc.resolve(mesh, n_components=1)
    assert len(resolved.fixed_idxs) == len(set(resolved.fixed_idxs))


def test_value_shape_is_checked_not_guessed(make_unit_square):
    """The old API chose between 'one value per node' and 'one value for all' by
    comparing len(indices) == len(values), so the same call changed meaning with
    mesh resolution. A wrong-width value must now simply raise."""
    mesh = make_unit_square(4)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0, 0])  # 3 components for n_components=2
    with pytest.raises(ValueError):
        bc.resolve(mesh, n_components=2)


def test_two_node_edge_pins_both_dofs(make_unit_square):
    """Regression for the length-coincidence bug: on a 2x2 mesh the left edge has
    exactly 2 nodes, which used to make [0, 0] mean 'one value per node' and
    crash. It must mean 'both components, at every node on the edge'."""
    mesh = make_unit_square(2)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    resolved = bc.resolve(mesh, n_components=2)
    assert len(resolved.fixed_idxs) == 4  # 2 nodes x 2 components
    assert len(resolved.fixed_values) == 4


def test_bctype_accepts_enum_and_string_but_rejects_typo():
    """The BC type is a canonical enum, yet the matching string still resolves;
    an unknown string is a typo and must raise."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0)  # enum form
    bc.add("neumann", everywhere(), 0)         # string form still works
    assert [t for t, _, _ in bc.conditions] == [BCType.DIRICHLET, BCType.NEUMANN]
    with pytest.raises(ValueError):
        bc.add("dirchlet", everywhere(), 0)    # typo


def test_index_list_as_region_is_rejected():
    """A bare index list is the old API's shape; it must fail with a message
    pointing at regions rather than being silently treated as a callable."""
    bc = BoundaryConditions()
    with pytest.raises(TypeError):
        bc.add(BCType.DIRICHLET, [0, 1, 2], 0)


def test_energy_solver_rejects_a_source_term(make_unit_square):
    """EnergySolver builds no load vector, so a source term would be accepted and
    then silently dropped, returning the unforced answer."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])

    eq = LinearElastic(E=200, nu=0.4, source=[0, -0.5])
    with pytest.raises(NotImplementedError):
        EnergySolver(mesh, eq, bc, verbose=False)

    # ...and without one it still constructs.
    EnergySolver(mesh, LinearElastic(E=200, nu=0.4), bc, verbose=False)


def test_energy_solver_accepts_a_3d_mesh():
    """The energy densities are now dimension-general, so a tet mesh is accepted."""
    mesh = Mesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        elements=[[0, 1, 2, 3]],
        boundary=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    )
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(2, 0.0), [0, 0, 0])

    solver = EnergySolver(mesh, LinearElastic(E=200, nu=0.4), bc, verbose=False)
    assert solver.n_components == 3


def test_robin_is_gated(make_unit_square):
    """Robin conditions need a left-hand-side term that is not wired up yet."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions()
    bc.add(BCType.ROBIN, everywhere(), 0)
    with pytest.raises(NotImplementedError):
        bc.resolve(mesh, n_components=1)


def test_plotmode_rejects_typo():
    assert PlotMode("surface") is PlotMode.SURFACE  # string resolves to the member
    with pytest.raises(ValueError):
        PlotMode("surfce")  # typo
