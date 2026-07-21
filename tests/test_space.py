"""Tests for FunctionSpace.

The equivalence tests are the important ones: they pin FunctionSpace's operators
to the FEMesh operators they will replace, so the migration that follows can be
checked against something rather than eyeballed.
"""
import numpy as np
import pytest

from fem.elements import LinearLineElement, LinearTetrahedralElement
from fem.materials import Enu_to_Lame
from fem.mesh.femesh import FEMesh
from fem.mesh.generation import create_box_mesh, create_rect_mesh
from fem.space import FunctionSpace, dof_indices


@pytest.fixture
def unit_square():
    return create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(6, 6))


def _material(mesh, n_components):
    '''Per-element Lame parameters, or nothing for a scalar space.'''
    if n_components == 1:
        return {}
    n = len(mesh.elements)
    mu, lamb = Enu_to_Lame(np.full(n, 200.0), np.full(n, 0.3))
    return {'mu': mu, 'lamb': lamb}


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


# --- equivalence with the FEMesh operators it replaces ---

@pytest.mark.parametrize('n_components', [1, 2])
def test_mass_matrix_matches_femesh(unit_square, n_components):
    femesh = FEMesh(unit_square.vertices, unit_square.elements, unit_square.boundary)
    # prepare_matrices assembles stiffness too, so the vector case has to be
    # handed material data just to obtain a mass matrix that does not use it.
    # The split into a cached mass_matrix and an explicit assemble_stiffness is
    # what removes that coupling.
    femesh.prepare_matrices(n_components=n_components, **_material(unit_square, n_components))
    space = FunctionSpace(unit_square, n_components=n_components)

    assert np.allclose(space.mass_matrix, femesh.M)
    assert np.allclose(space.boundary_mass_matrix, femesh.M_b)


def test_scalar_stiffness_matches_femesh(unit_square):
    femesh = FEMesh(unit_square.vertices, unit_square.elements, unit_square.boundary)
    femesh.prepare_matrices(n_components=1)
    space = FunctionSpace(unit_square, n_components=1)

    assert np.allclose(space.assemble_stiffness(), femesh.K)


def test_elastic_stiffness_matches_femesh(unit_square):
    material = _material(unit_square, n_components=2)
    femesh = FEMesh(unit_square.vertices, unit_square.elements, unit_square.boundary)
    femesh.prepare_matrices(n_components=2, **material)
    space = FunctionSpace(unit_square, n_components=2)

    assert np.allclose(space.assemble_stiffness(**material), femesh.K)


def test_dof_indices_free_function_is_the_same_numbering(unit_square):
    space = FunctionSpace(unit_square, n_components=3)
    assert np.array_equal(space.dof_indices([1, 4]), dof_indices([1, 4], 3))


# --- the property FEMesh cannot provide ---

def test_two_spaces_share_one_mesh_without_interfering(unit_square):
    """The motivating case. FEMesh.prepare_matrices rebuilds operators in place,
    so a second solver at a different component count silently corrupts the
    first. Separate spaces have separate operators over the same geometry."""
    scalar = FunctionSpace(unit_square, n_components=1)
    vector = FunctionSpace(unit_square, n_components=2)

    scalar_mass = scalar.mass_matrix.copy()
    _ = vector.mass_matrix  # would overwrite, under the old model

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
