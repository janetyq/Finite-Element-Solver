"""Tests for FunctionSpace."""
import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
    QuadraticTriangleElement,
)
from fem.mesh.structured import box_mesh
from fem.space import FunctionSpace


@pytest.fixture
def unit_square():
    return box_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))


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
    box = box_mesh([[0, 0, 0], [1, 1, 1]], (3, 3, 3))
    scalar_on_tets = FunctionSpace(box, element_type=LinearTetrahedralElement, n_components=1)
    assert scalar_on_tets.spatial_dim == 3
    assert scalar_on_tets.n_components == 1


# --- the property that motivated the split ---

def test_two_spaces_share_one_mesh_without_interfering(unit_square):
    """Two discretizations of one domain do not share mutable operator state."""
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
    """A line element's facets would be points, which no boundary integral supports; refused
    at construction."""
    with pytest.raises(NotImplementedError):
        FunctionSpace(unit_square, element_type=LinearLineElement)


def test_nonpositive_n_components_is_rejected(unit_square):
    with pytest.raises(ValueError):
        FunctionSpace(unit_square, n_components=0)


# --- interpolation ---

@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
@pytest.mark.parametrize('value', [1.5, [1.5, 1.5]])
def test_interpolate_fills_every_dof_of_the_space(unit_square, element_type, value):
    n_components = 1 if np.isscalar(value) else len(value)
    space = FunctionSpace(unit_square, element_type, n_components=n_components)
    assert space.interpolate(value).dofs.shape == (space.n_dofs,)
    assert np.all(space.interpolate(value).dofs == 1.5)


def test_interpolate_samples_a_callable_at_the_p2_edge_nodes(unit_square):
    space = FunctionSpace(unit_square, QuadraticTriangleElement)
    u = space.interpolate(lambda p: p[:, 0] + 2 * p[:, 1])
    expected = space.node_coords[:, 0] + 2 * space.node_coords[:, 1]
    np.testing.assert_allclose(u, expected)
    assert len(u) > len(unit_square.vertices)


def test_interpolate_interleaves_vector_components(unit_square):
    space = FunctionSpace(unit_square, n_components=2)
    u = space.interpolate([1.0, 2.0]).dofs
    assert np.all(u[0::2] == 1.0) and np.all(u[1::2] == 2.0)
