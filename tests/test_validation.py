"""Guardrails around half-implemented or easily-misused surfaces fail loudly rather than
returning wrong or empty results.
"""
import numpy as np
import pytest

from fem.solver import Solver
from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.mesh.mesh import Mesh
from fem.physics.equations import FiniteStrainElastic
from fem.regions import everywhere, on_plane, at_indices


def test_named_interior_vertex_is_rejected(make_unit_square):
    """Naming an interior node explicitly is a modelling error; only the at_indices path can
    trip this, since a geometric region intersects with the boundary."""
    mesh = make_unit_square(8)
    interior = (set(range(len(mesh.vertices))) - set(int(i) for i in mesh.boundary_idxs)).pop()

    bc = BoundaryConditions(Dirichlet(at_indices([interior]), 0))
    with pytest.raises(ValueError):
        bc.resolve(mesh, n_components=1)


def test_geometric_region_never_selects_interior_vertices(make_unit_square):
    """A plane cutting through the domain still yields only boundary DOFs: the
    old 'BC on a non-boundary vertex' error is now unrepresentable."""
    mesh = make_unit_square(9)  # odd, so x = 0.5 is a grid line
    bc = BoundaryConditions(Dirichlet(on_plane(0, 0.5), 0))

    interior_on_plane = np.isclose(mesh.vertices[:, 0], 0.5).sum() - 2
    assert interior_on_plane > 0, "region does not actually cross the interior"

    fixed = bc.resolve(mesh, n_components=1).fixed_idxs
    assert len(fixed) == 2  # only where the line meets the boundary
    assert set(fixed) <= set(int(i) for i in mesh.boundary_idxs)


def test_dirichlet_neumann_same_component_is_rejected(make_unit_square):
    """Pinning and loading the same component is the ambiguity to flag."""
    mesh = make_unit_square(8)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 0.0), [3.0, 0]),
    )
    with pytest.raises(ValueError, match='same'):
        bc.resolve(mesh, n_components=2)


def test_dirichlet_neumann_different_components_is_allowed(make_unit_square):
    """Pinning one component while a traction drives another (a roller carrying a tangential
    load) is well-posed."""
    mesh = make_unit_square(8)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [None, 0]),
        Neumann(on_plane(0, 0.0), [3.0, 0]),
    )
    resolved = bc.resolve(mesh, n_components=2)

    # The y-DOFs are fixed; the x-DOFs stay free and carry the traction load.
    left = np.flatnonzero(np.isclose(mesh.vertices[:, 0], 0.0))
    assert set(2 * left + 1) <= set(resolved.fixed_idxs)      # u_y fixed
    assert set(2 * left) <= set(resolved.free_idxs)           # u_x free
    assert np.any(resolved.neumann_load[left, 0] != 0)


def test_agreeing_overlapping_regions_are_fine(make_unit_square):
    mesh = make_unit_square(8)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), 0.0),
        Dirichlet(on_plane(1, 0.0), 0.0),
    )
    resolved = bc.resolve(mesh, n_components=1)
    assert len(resolved.fixed_idxs) == len(set(resolved.fixed_idxs))


def test_a_specification_is_a_frozen_tuple_of_conditions():
    """Conditions are collected by the constructor; anything else is refused, and the
    collection cannot be mutated."""
    pinned = Dirichlet(everywhere(), 0)
    bc = BoundaryConditions(pinned, Neumann(everywhere(), 0))
    assert [type(c) for c in bc] == [Dirichlet, Neumann]
    assert BoundaryConditions(*bc, pinned).conditions == (*bc.conditions, pinned)
    with pytest.raises(TypeError, match='Dirichlet, Neumann, or Robin'):
        BoundaryConditions(('dirichlet', everywhere(), 0))  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        bc.conditions = ()  # type: ignore[misc]


def test_index_list_as_region_is_rejected():
    """A bare index list fails with a message pointing at regions."""
    with pytest.raises(TypeError):
        BoundaryConditions(Dirichlet([0, 1, 2], 0))


def test_finite_strain_accepts_a_3d_mesh():
    """The energy densities are dimension-general, so a tet mesh is accepted."""
    mesh = Mesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        elements=[[0, 1, 2, 3]],
        boundary=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    )
    bc = BoundaryConditions(Dirichlet(on_plane(2, 0.0), [0, 0, 0]))

    equation = FiniteStrainElastic(E=200, nu=0.4)
    assert Solver(mesh, equation, bc).space.n_components == 3


def test_finite_strain_rejects_a_per_element_modulus(make_unit_square):
    """A density carries one pair of Lame parameters for the whole mesh, so an
    array E broadcasts wrongly against the constant d2W/dS2 rather than giving
    per-element moduli. The small-strain path is the one that supports them."""
    mesh = make_unit_square(6)
    bc = BoundaryConditions(Dirichlet(on_plane(0, 0.0), [0, 0]))

    E = np.full(len(mesh.elements), 200.0)
    equation = FiniteStrainElastic(E=E, nu=0.4)
    with pytest.raises(NotImplementedError):
        Solver(mesh, equation, bc).problem()


def test_a_shared_specification_is_unchanged_by_resolution(make_unit_square):
    """One spec resolved against two spaces, at two times, is the same spec after."""
    from fem.regions import TimeDependent
    from fem.space import FunctionSpace
    from fem.elements import QuadraticTriangleElement
    mesh = make_unit_square(5)
    bc = BoundaryConditions(Dirichlet(on_plane(0, 0.0), TimeDependent(lambda p, t: t)),
                            Neumann(on_plane(0, 1.0), 1.0))
    before = bc.conditions
    p1 = bc.resolve(FunctionSpace(mesh).nodes, 1)
    p2 = bc.resolve(FunctionSpace(mesh, QuadraticTriangleElement).nodes, 1, t=2.0)
    assert bc.conditions is before
    assert len(p2.fixed_idxs) > len(p1.fixed_idxs)   # the P2 edge nodes are pinned too
    np.testing.assert_allclose(p2.fixed_values, 2.0)
    later = p1.at(3.0)
    np.testing.assert_allclose(later.fixed_values, 3.0)
    np.testing.assert_array_equal(later.fixed_idxs, p1.fixed_idxs)
