"""Tests for FunctionSpace."""
import numpy as np
import pytest

from fem.elements import LinearLineElement, LinearTetrahedralElement
from fem.mesh.ruppert import create_box_mesh, create_rect_mesh
from fem.mesh.mesh import Mesh
from fem.space import FunctionSpace


@pytest.fixture
def unit_square():
    return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))


# --- numbering and sizing ---

def test_n_dofs_counts_components(unit_square):
    assert FunctionSpace(unit_square).n_dofs == len(unit_square.vertices)
    assert FunctionSpace(unit_square, n_components=2).n_dofs == 2 * len(unit_square.vertices)


def test_dof_indices_interleave_by_node(unit_square):
    space = FunctionSpace(unit_square, n_components=2)
    # node 3 -> DOFs 6, 7; node 5 -> 10, 11
    assert list(space.dof_indices([3, 5])) == [6, 7, 10, 11]
    assert list(FunctionSpace(unit_square).dof_indices([3, 5])) == [3, 5]


def test_spatial_dim_is_not_n_components(unit_square):
    space = FunctionSpace(unit_square, n_components=2)
    assert space.spatial_dim == 2
    box = create_box_mesh([[0, 0, 0], [1, 1, 1]], (3, 3, 3))
    scalar_on_tets = FunctionSpace(box, element_type=LinearTetrahedralElement, n_components=1)
    assert scalar_on_tets.spatial_dim == 3
    assert scalar_on_tets.n_components == 1


# --- the property that motivated the split ---

def test_two_spaces_share_one_mesh_without_interfering(unit_square):
    """Two discretizations of one domain must not share mutable operator state,
    which is what forces the space to be a separate object from the mesh."""
    scalar = FunctionSpace(unit_square, n_components=1)
    vector = FunctionSpace(unit_square, n_components=2)

    scalar_mass = scalar.mass_matrix.copy()
    _ = vector.mass_matrix

    assert scalar.mass_matrix.shape == (len(unit_square.vertices),) * 2
    assert vector.mass_matrix.shape == (2 * len(unit_square.vertices),) * 2
    # Sparse operators: compare densely, and confirm nothing mutated.
    assert np.allclose(scalar.mass_matrix.toarray(), scalar_mass.toarray())
    assert scalar.mesh is vector.mesh


def test_operators_and_geometry_are_cached(unit_square):
    space = FunctionSpace(unit_square)
    assert space.geometry is space.geometry
    assert space.boundary_geometry is space.boundary_geometry
    assert space.mass_matrix is space.mass_matrix


# --- guardrails ---

def test_element_without_facets_is_rejected(unit_square):
    """A line element's facets would be points, which no boundary integral
    supports yet -- refuse at construction rather than fail on a None call."""
    with pytest.raises(NotImplementedError):
        FunctionSpace(unit_square, element_type=LinearLineElement)


def test_nonpositive_n_components_is_rejected(unit_square):
    with pytest.raises(ValueError):
        FunctionSpace(unit_square, n_components=0)


# -- element -> vertex projection --------------------------------------------
#
# Moved here from tests/test_mesh.py with the operation itself: projecting an
# element field onto the nodes needs element measures, so it belongs to the
# discretization rather than the geometry.


def _two_triangle_square() -> Mesh:
    """The unit square as two equal triangles sharing the diagonal 0--2.

    Vertices 0 and 2 lie on the shared diagonal (in both elements); 1 and 3 each
    belong to a single element -- the configuration a last-writer bug corrupts.
    """
    return Mesh(
        vertices=[[0, 0], [1, 0], [1, 1], [0, 1]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )


def test_shared_vertex_combines_the_values_of_its_elements():
    space = FunctionSpace(_two_triangle_square())
    values = space.element_to_vertex(np.array([10.0, 20.0]))

    # Both triangles have the same area, so the weighting reduces to the mean.
    assert values[0] == 15.0
    assert values[2] == 15.0
    assert values[1] == 10.0
    assert values[3] == 20.0


def test_projection_does_not_depend_on_element_ordering():
    """An earlier version assigned rather than accumulated, so a shared vertex
    kept only the last element to touch it -- an order-dependent, silently wrong
    field. Reversing the elements must leave the result unchanged."""
    mesh = _two_triangle_square()
    reversed_mesh = Mesh(
        vertices=mesh.vertices, elements=mesh.elements[::-1], boundary=mesh.boundary,
    )
    forward = FunctionSpace(mesh).element_to_vertex(np.array([10.0, 20.0]))
    backward = FunctionSpace(reversed_mesh).element_to_vertex(np.array([20.0, 10.0]))
    assert np.allclose(forward, backward)


def test_constant_element_field_is_reproduced_at_every_vertex(make_unit_square):
    """The patch test: a constant per-element field must come back as the same
    constant at every vertex, whatever the valence and whatever the weighting."""
    space = FunctionSpace(make_unit_square(6))
    constant = np.full(len(space.mesh.elements), 3.5)
    assert np.allclose(space.element_to_vertex(constant), 3.5)


def test_projection_weights_by_element_volume():
    """A large element and a sliver meeting at a node are not equally good
    evidence about the field there, so the projection weights by measure. Pinned
    on a deliberately graded mesh, where an unweighted mean gives a different
    answer -- vertex 0 is shared by a triangle of area 0.5 and one of area 0.05.
    """
    mesh = Mesh(
        vertices=[[0, 0], [1, 0], [0, 1], [-0.1, 0]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )
    space = FunctionSpace(mesh)
    areas = space.element_volumes
    np.testing.assert_allclose(areas, [0.5, 0.05])

    values = space.element_to_vertex(np.array([10.0, 20.0]))
    expected = (10.0 * areas[0] + 20.0 * areas[1]) / areas.sum()

    np.testing.assert_allclose(values[0], expected)
    assert not np.isclose(values[0], 15.0)  # what an unweighted mean would give


def test_projection_rejects_a_field_of_the_wrong_length(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='one value per element'):
        space.element_to_vertex(np.zeros(3))
