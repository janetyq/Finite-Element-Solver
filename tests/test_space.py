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
from fem.mesh.mesh import Mesh
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


# -- element -> vertex projection --------------------------------------------
#
# Moved here from tests/test_mesh.py with the operation itself: projecting an
# element field onto the nodes needs element measures, so it belongs to the
# discretization rather than the geometry.


def _two_triangle_square() -> Mesh:
    """The unit square as two equal triangles sharing the diagonal 0-2, so vertices 0 and 2
    belong to both elements and 1 and 3 to one each."""
    return Mesh(
        vertices=[[0, 0], [1, 0], [1, 1], [0, 1]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )


def test_shared_vertex_combines_the_values_of_its_elements():
    space = FunctionSpace(_two_triangle_square())
    values = space.recover_nodal(np.array([10.0, 20.0]))

    # Both triangles have the same area, so the weighting reduces to the mean.
    assert values[0] == 15.0
    assert values[2] == 15.0
    assert values[1] == 10.0
    assert values[3] == 20.0


def test_projection_does_not_depend_on_element_ordering():
    """A shared vertex accumulates every element's contribution, so reversing the elements
    leaves the result unchanged."""
    mesh = _two_triangle_square()
    reversed_mesh = Mesh(
        vertices=mesh.vertices, elements=mesh.elements[::-1], boundary=mesh.boundary,
    )
    forward = FunctionSpace(mesh).recover_nodal(np.array([10.0, 20.0]))
    backward = FunctionSpace(reversed_mesh).recover_nodal(np.array([20.0, 10.0]))
    assert np.allclose(forward, backward)


def test_constant_element_field_is_reproduced_at_every_vertex(make_unit_square):
    """The patch test: a constant per-element field must come back as the same
    constant at every vertex, whatever the valence and whatever the weighting."""
    space = FunctionSpace(make_unit_square(6))
    constant = np.full(len(space.mesh.elements), 3.5)
    assert np.allclose(space.recover_nodal(constant), 3.5)


def test_projection_weights_by_element_volume():
    """The projection weights by element measure: on a graded mesh (vertex 0 shared by a
    triangle of area 0.5 and one of area 0.05) an unweighted mean gives a different
    answer."""
    mesh = Mesh(
        vertices=[[0, 0], [1, 0], [0, 1], [-0.1, 0]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )
    space = FunctionSpace(mesh)
    areas = space.element_volumes
    np.testing.assert_allclose(areas, [0.5, 0.05])

    values = space.recover_nodal(np.array([10.0, 20.0]))
    expected = (10.0 * areas[0] + 20.0 * areas[1]) / areas.sum()

    np.testing.assert_allclose(values[0], expected)
    assert not np.isclose(values[0], 15.0)  # what an unweighted mean would give


def test_projection_rejects_a_field_of_the_wrong_length(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='one value per element'):
        space.recover_nodal(np.zeros(3))


def test_unknown_recovery_method_is_rejected(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='unknown recovery method'):
        space.recover_nodal(np.zeros(len(space.mesh.elements)), method='patch')


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_l2_recovery_reproduces_a_constant_field(element_type):
    """The patch test for the L2 projection: a constant per-element field projects to
    that same constant at every node, since the constant lies in the nodal space."""
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 5])
    space = FunctionSpace(mesh, element_type, n_components=1)
    constant = np.full(len(mesh.elements), 3.5)
    assert np.allclose(space.recover_nodal(constant, method='l2'), 3.5)


def test_l2_recovery_conserves_the_field_integral():
    """The L2 projection satisfies ∫ q = ∫ f (test against the constant 1, which the
    nodal space represents), so the recovered field carries the per-element field's
    integral to machine precision on any mesh."""
    uniform = box_mesh([[0.0, 0.0], [1.0, 1.0]], [10, 10])
    grading = np.column_stack([uniform.vertices[:, 0] ** 2 - uniform.vertices[:, 0],
                               np.zeros(uniform.n_vertices)])
    mesh = uniform.displaced(grading)                        # x -> x^2, a graded mesh
    space = FunctionSpace(mesh)
    field = mesh.vertices[mesh.elements].mean(axis=1)[:, 0]  # varies element to element
    exact = float((field * space.element_volumes).sum())

    recovered = space.recover_nodal(field, method='l2')
    assert space.integrate(recovered) == pytest.approx(exact, rel=1e-12)


def test_l2_and_average_recovery_differ_on_a_varying_field():
    """The two recoveries are different operators: the local weighted average
    and the global mass projection agree only on a field the space reproduces exactly
    (a constant), and differ on one that varies element to element."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [8, 8])
    space = FunctionSpace(mesh)
    field = mesh.vertices[mesh.elements].mean(axis=1)[:, 0] ** 2

    average = space.recover_nodal(field, method='average')
    l2 = space.recover_nodal(field, method='l2')
    assert not np.allclose(average, l2)


# --- interpolation ---

@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
@pytest.mark.parametrize('value', [1.5, [1.5, 1.5]])
def test_interpolate_fills_every_dof_of_the_space(unit_square, element_type, value):
    n_components = 1 if np.isscalar(value) else len(value)
    space = FunctionSpace(unit_square, element_type, n_components=n_components)
    assert space.interpolate(value).shape == (space.n_dofs,)
    assert np.all(space.interpolate(value) == 1.5)


def test_interpolate_samples_a_callable_at_the_p2_edge_nodes(unit_square):
    space = FunctionSpace(unit_square, QuadraticTriangleElement)
    u = space.interpolate(lambda p: p[0] + 2 * p[1])
    expected = space.node_coords[:, 0] + 2 * space.node_coords[:, 1]
    np.testing.assert_allclose(u, expected)
    assert len(u) > len(unit_square.vertices)


def test_interpolate_interleaves_vector_components(unit_square):
    space = FunctionSpace(unit_square, n_components=2)
    u = space.interpolate([1.0, 2.0])
    assert np.all(u[0::2] == 1.0) and np.all(u[1::2] == 2.0)
