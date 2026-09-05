"""`NodalField`: DOF values on a space, and what the pairing lets it answer."""
import numpy as np
import pytest

from fem.elements import QuadraticTriangleElement
from fem.field import NodalField
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.regions import on_plane
from fem.space import FunctionSpace


@pytest.fixture
def square():
    return box_mesh(corners=[[0, 0], [1, 1]], resolution=(4, 4))


# -- construction --------------------------------------------------------------


def test_interpolate_returns_a_field_on_the_space(square):
    space = FunctionSpace(square)
    field = space.interpolate(lambda p: p[:, 0])
    assert isinstance(field, NodalField)
    assert field.space is space
    np.testing.assert_allclose(field.dofs, square.vertices[:, 0])


def test_field_size_is_checked_against_the_space(square):
    p1, p2 = FunctionSpace(square), FunctionSpace(square, QuadraticTriangleElement)
    with pytest.raises(ValueError, match='DOFs'):
        NodalField(p2, np.zeros(p1.n_dofs))
    with pytest.raises(ValueError, match='DOFs'):
        NodalField(FunctionSpace(square, n_components=2), np.zeros(p1.n_dofs))
    with pytest.raises(ValueError, match='DOFs'):
        NodalField(p1, np.zeros((p1.n_nodes, 1)))


def test_field_values_are_frozen_and_copied(square):
    space = FunctionSpace(square)
    values = np.ones(space.n_dofs)
    field = NodalField(space, values)
    values[0] = 5.0
    assert field.dofs[0] == 1.0
    with pytest.raises(ValueError):
        field.dofs[0] = 2.0


def test_field_is_an_array_to_numpy(square):
    space = FunctionSpace(square)
    field = space.interpolate(2.0)
    np.testing.assert_array_equal(np.asarray(field), field.dofs)
    assert len(field) == space.n_dofs
    np.testing.assert_allclose(space.mass_matrix @ field, space.mass_matrix @ field.dofs)


# -- the values by node -------------------------------------------------------


def test_nodal_values_and_components_of_a_vector_field(square):
    space = FunctionSpace(square, n_components=2)
    field = space.interpolate(lambda p: [p[:, 0], 10 * p[:, 1]])
    assert field.nodal_values.shape == (space.n_nodes, 2)
    np.testing.assert_allclose(field.component(0), square.vertices[:, 0])
    np.testing.assert_allclose(field.component(1), 10 * square.vertices[:, 1])
    np.testing.assert_allclose(field.nodal_values[:, 1], field.component(1))
    with pytest.raises(IndexError):
        field.component(2)


def test_scalar_nodal_values_are_one_dimensional(square):
    field = FunctionSpace(square).interpolate(3.0)
    assert field.nodal_values.shape == (square.n_vertices,)
    np.testing.assert_array_equal(field.component(0), field.nodal_values)


def test_element_values_gather_by_the_space_nodes(square):
    space = FunctionSpace(square, QuadraticTriangleElement, n_components=2)
    field = space.interpolate(lambda p: [p[:, 0], p[:, 1]])
    assert field.element_values.shape == (square.n_elements, 6, 2)
    np.testing.assert_allclose(field.element_values, space.node_coords[space.element_nodes])


# -- integrals and derivatives -------------------------------------------------


@pytest.mark.parametrize('element_type', [None, QuadraticTriangleElement])
def test_integral_and_mean_of_a_linear_field_are_exact(square, element_type):
    space = FunctionSpace(square, element_type)
    field = space.interpolate(lambda p: 2 * p[:, 0] + 3 * p[:, 1])
    assert field.integrate() == pytest.approx(2.5, rel=1e-12)
    assert field.mean() == pytest.approx(2.5, rel=1e-12)


def test_integral_of_a_vector_field_is_per_component(square):
    space = FunctionSpace(square, n_components=2)
    field = space.interpolate([1.0, 4.0])
    np.testing.assert_allclose(field.integrate(), [1.0, 4.0])
    np.testing.assert_allclose(field.mean(), [1.0, 4.0])


@pytest.mark.parametrize('element_type', [None, QuadraticTriangleElement])
def test_gradient_of_a_linear_field_is_its_slope(square, element_type):
    space = FunctionSpace(square, element_type)
    scalar = space.interpolate(lambda p: 2 * p[:, 0] + 3 * p[:, 1]).gradient()
    assert scalar.shape == (square.n_elements, 2)
    np.testing.assert_allclose(scalar, [[2.0, 3.0]] * square.n_elements, atol=1e-12)

    vector = FunctionSpace(square, element_type, n_components=2)
    grad = vector.interpolate(lambda p: [p[:, 0], 2 * p[:, 0] + 3 * p[:, 1]]).gradient()
    assert grad.shape == (square.n_elements, 2, 2)
    np.testing.assert_allclose(grad, [[[1.0, 0.0], [2.0, 3.0]]] * square.n_elements, atol=1e-12)


def test_space_gradient_takes_a_field_or_a_vector(square):
    space = FunctionSpace(square)
    field = space.interpolate(lambda p: p[:, 0])
    np.testing.assert_allclose(space.gradient(field), space.gradient(field.dofs))


# -- evaluation -----------------------------------------------------------------


def test_locate_finds_the_element_and_reference_coordinates(square):
    points = np.array([[0.3, 0.2], [0.9, 0.95], [0.5, 0.5], [0.0, 0.0]])
    elements, reference = square.locate(points)
    corners = square.vertices[square.elements[elements]]         # (n_points, 3, 2)
    # Corner 0 plus the reference coordinates along the edges reproduces the point.
    rebuilt = corners[:, 0] + np.einsum('pr,pri->pi', reference, corners[:, 1:] - corners[:, :1])
    np.testing.assert_allclose(rebuilt, points, atol=1e-12)
    assert np.all(reference >= -1e-9) and np.all(reference.sum(axis=1) <= 1 + 1e-9)


def test_locate_batches_points_the_way_it_takes_them_one_at_a_time(square):
    """Every point is tested against its candidates at once; the answer is the one a
    point-by-point search gives, and the search structure is built once per mesh."""
    points = np.random.default_rng(0).random((200, 2))
    elements, reference = square.locate(points)
    for p, e, r in zip(points, elements, reference, strict=True):
        (e1,), (r1,) = square.locate([p])
        assert e1 == e
        np.testing.assert_allclose(r1, r)
    assert square._locator is square._locator


def test_locate_falls_back_to_every_element_when_the_nearest_miss():
    """A point in a large element near a cluster of small ones has none of the
    nearest centroids in its own element; the fallback still finds it."""
    fine = box_mesh(corners=[[0, 0], [1, 1]], resolution=(30, 30))
    n = fine.n_vertices
    vertices = np.vstack([fine.vertices, [[1.0, 0.0], [1.0, 1.0], [40.0, 0.5]]])
    elements = np.vstack([fine.elements, [[n, n + 1, n + 2]]])
    mesh = Mesh(vertices, elements)
    (element,), (reference,) = mesh.locate(np.array([[1.5, 0.5]]))
    assert element == mesh.n_elements - 1
    assert np.all(reference >= 0) and reference.sum() <= 1
    # The miss is reported by the point that missed, not the batch.
    with pytest.raises(ValueError, match=r'-1\.\s*\] lies outside'):
        mesh.locate(np.array([[0.2, 0.3], [0.5, -1.0]]))


def test_locate_rejects_a_point_outside(square):
    with pytest.raises(ValueError, match='outside'):
        square.locate([[1.5, 0.5]])
    with pytest.raises(ValueError, match='dimensional'):
        square.locate([[0.5, 0.5, 0.5]])


def test_locate_in_three_dimensions():
    mesh = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(2, 2, 2))
    points = np.array([[0.1, 0.7, 0.4], [0.95, 0.05, 0.5]])
    elements, reference = mesh.locate(points)
    corners = mesh.vertices[mesh.elements[elements]]
    rebuilt = corners[:, 0] + np.einsum('pr,pri->pi', reference, corners[:, 1:] - corners[:, :1])
    np.testing.assert_allclose(rebuilt, points, atol=1e-12)


def test_p1_field_evaluates_a_linear_function_exactly(square):
    space = FunctionSpace(square)
    field = space.interpolate(lambda p: 1 + 2 * p[:, 0] - p[:, 1])
    points = np.array([[0.13, 0.71], [0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(field.evaluate(points), 1 + 2 * points[:, 0] - points[:, 1], atol=1e-12)
    assert field.evaluate(points).shape == (3,)


def test_p2_field_evaluates_a_quadratic_exactly(square):
    space = FunctionSpace(square, QuadraticTriangleElement)
    field = space.interpolate(lambda p: p[:, 0] ** 2 + p[:, 0] * p[:, 1])
    points = np.array([[0.13, 0.71], [0.6, 0.35]])
    expected = points[:, 0] ** 2 + points[:, 0] * points[:, 1]
    np.testing.assert_allclose(field.evaluate(points), expected, atol=1e-12)


def test_vector_field_evaluates_per_component(square):
    space = FunctionSpace(square, n_components=2)
    field = space.interpolate(lambda p: [p[:, 0], 3 * p[:, 1]])
    values = field.evaluate([[0.25, 0.5]])
    assert values.shape == (1, 2)
    np.testing.assert_allclose(values[0], [0.25, 1.5], atol=1e-12)


def test_evaluate_at_a_node_reads_the_nodal_value(square):
    space = FunctionSpace(square)
    field = NodalField(space, np.arange(space.n_dofs, dtype=float))
    np.testing.assert_allclose(field.evaluate(space.node_coords), field.dofs, atol=1e-12)


# -- deformation ----------------------------------------------------------------


def test_boundary_integral_of_one_is_the_perimeter(square):
    space = FunctionSpace(square)
    assert space.interpolate(1.0).boundary_integral() == pytest.approx(4.0)


def test_boundary_integral_over_a_region_takes_only_its_facets(square):
    """x over the whole boundary is 2 (the right edge, plus half of the top and bottom);
    over the right edge alone it is 1."""
    space = FunctionSpace(square)
    x = space.interpolate(lambda p: p[:, 0])
    assert x.boundary_integral() == pytest.approx(2.0)
    assert x.boundary_integral(on_plane(0, 1.0)) == pytest.approx(1.0)
    assert x.boundary_integral(on_plane(0, 0.0)) == pytest.approx(0.0)


def test_p2_boundary_integral_is_exact_for_a_quadratic(square):
    """∫_0^1 y² dy along the right edge."""
    space = FunctionSpace(square, QuadraticTriangleElement)
    field = space.interpolate(lambda p: p[:, 1] ** 2)
    assert field.boundary_integral(on_plane(0, 1.0)) == pytest.approx(1 / 3)


def test_vector_boundary_integral_is_per_component(square):
    space = FunctionSpace(square, n_components=2)
    field = space.interpolate(lambda p: [1.0, p[:, 0]])
    np.testing.assert_allclose(field.boundary_integral(), [4.0, 2.0])


def test_deformed_mesh_moves_the_vertices_by_the_field(square):
    space = FunctionSpace(square, n_components=2)
    field = space.interpolate([0.5, 0.0])
    moved = field.deformed_mesh(scale=2.0)
    np.testing.assert_allclose(moved.vertices, square.vertices + [1.0, 0.0])


def test_deformed_mesh_needs_a_displacement(square):
    with pytest.raises(ValueError, match='component per spatial dimension'):
        FunctionSpace(square).interpolate(1.0).deformed_mesh()
