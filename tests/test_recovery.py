"""Nodal recovery (`fem.post.recovery`): the volume-weighted average and the L2
projection of a per-element field onto the nodes."""
import numpy as np

from fem.field import NodalField
import pytest

from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.post.recovery import recover_nodal
from fem.space import FunctionSpace


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
    values = recover_nodal(space, np.array([10.0, 20.0]))

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
    forward = recover_nodal(FunctionSpace(mesh), np.array([10.0, 20.0]))
    backward = recover_nodal(FunctionSpace(reversed_mesh), np.array([20.0, 10.0]))
    assert np.allclose(forward, backward)


def test_constant_element_field_is_reproduced_at_every_vertex(make_unit_square):
    """The patch test: a constant per-element field must come back as the same
    constant at every vertex, whatever the valence and whatever the weighting."""
    space = FunctionSpace(make_unit_square(6))
    constant = np.full(len(space.mesh.elements), 3.5)
    assert np.allclose(recover_nodal(space, constant), 3.5)


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

    values = recover_nodal(space, np.array([10.0, 20.0]))
    expected = (10.0 * areas[0] + 20.0 * areas[1]) / areas.sum()

    np.testing.assert_allclose(values[0], expected)
    assert not np.isclose(values[0], 15.0)  # what an unweighted mean would give


def test_projection_rejects_a_field_of_the_wrong_length(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='one value per element'):
        recover_nodal(space, np.zeros(3))


def test_unknown_recovery_method_is_rejected(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='unknown recovery method'):
        recover_nodal(space, np.zeros(len(space.mesh.elements)), method='patch')


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_l2_recovery_reproduces_a_constant_field(element_type):
    """The patch test for the L2 projection: a constant per-element field projects to
    that same constant at every node, since the constant lies in the nodal space."""
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 5])
    space = FunctionSpace(mesh, element_type, n_components=1)
    constant = np.full(len(mesh.elements), 3.5)
    assert np.allclose(recover_nodal(space, constant, method='l2'), 3.5)


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

    recovered = recover_nodal(space, field, method='l2')
    assert NodalField(space, recovered).integrate() == pytest.approx(exact, rel=1e-12)


def test_l2_and_average_recovery_differ_on_a_varying_field():
    """The two recoveries are different operators: the local weighted average
    and the global mass projection agree only on a field the space reproduces exactly
    (a constant), and differ on one that varies element to element."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [8, 8])
    space = FunctionSpace(mesh)
    field = mesh.vertices[mesh.elements].mean(axis=1)[:, 0] ** 2

    average = recover_nodal(space, field, method='average')
    l2 = recover_nodal(space, field, method='l2')
    assert not np.allclose(average, l2)
