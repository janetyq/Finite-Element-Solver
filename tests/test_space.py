"""Tests for FunctionSpace."""
import numpy as np
import pytest

from fem.elements import LinearLineElement, LinearTetrahedralElement
from fem.mesh.generation import create_box_mesh, create_rect_mesh
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
    assert np.allclose(scalar.mass_matrix, scalar_mass)
    assert scalar.mesh is vector.mesh


def test_operators_and_element_objs_are_cached(unit_square):
    space = FunctionSpace(unit_square)
    assert space.element_objs is space.element_objs
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
