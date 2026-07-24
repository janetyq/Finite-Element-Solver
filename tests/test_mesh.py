"""Mesh geometry helpers -- the element<->vertex field projections.

`convert_element_values_to_vertex_values` used to assign each element's value into
its vertices, so a vertex shared by several elements kept only whichever element was
visited last -- an order-dependent, silently wrong field. These pin the averaging.
"""
import numpy as np

from fem.mesh.mesh import Mesh


def _two_triangle_square() -> Mesh:
    """The unit square as two triangles sharing the diagonal 0--2.

    Vertices 0 and 2 lie on the shared diagonal (in both elements); 1 and 3 each
    belong to a single element -- exactly the configuration a last-writer bug
    corrupts.
    """
    return Mesh(
        vertices=[[0, 0], [1, 0], [1, 1], [0, 1]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )


def test_shared_vertex_gets_the_mean_of_its_elements():
    mesh = _two_triangle_square()
    vertex_values = mesh.convert_element_values_to_vertex_values(np.array([10.0, 20.0]))

    # Vertices 0 and 2 are in both elements -> mean; 1 and 3 in one each -> that value.
    assert vertex_values[0] == 15.0
    assert vertex_values[2] == 15.0
    assert vertex_values[1] == 10.0
    assert vertex_values[3] == 20.0


def test_order_of_elements_does_not_change_the_result():
    """The last-writer bug made the answer depend on element ordering; averaging
    is order-independent, so reversing the elements must leave the field unchanged."""
    mesh = _two_triangle_square()
    reversed_mesh = Mesh(
        vertices=mesh.vertices,
        elements=mesh.elements[::-1],
        boundary=mesh.boundary,
    )
    forward = mesh.convert_element_values_to_vertex_values(np.array([10.0, 20.0]))
    backward = reversed_mesh.convert_element_values_to_vertex_values(np.array([20.0, 10.0]))
    assert np.allclose(forward, backward)


def test_constant_element_field_is_reproduced_at_every_vertex(make_unit_square):
    """A constant per-element field averages to the same constant at every vertex,
    whatever the valence -- the patch test for a nodal projection."""
    mesh = make_unit_square(6)
    constant = np.full(len(mesh.elements), 3.5)
    vertex_values = mesh.convert_element_values_to_vertex_values(constant)
    assert np.allclose(vertex_values, 3.5)
