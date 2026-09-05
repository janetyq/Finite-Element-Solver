"""The element primitives: shape functions and their derivatives, and the fields a space
builds from them.
"""
import numpy as np
import pytest

from fem.elements import QuadraticTetrahedralElement, QuadraticTriangleElement
from fem.mesh.structured import box_mesh
from fem.space import FunctionSpace

# One interior sample set per quadratic element, in its own reference coordinates.
QUADRATIC_POINTS = {
    QuadraticTriangleElement: [[0.3, 0.4], [0.1, 0.25], [0.5, 0.2]],
    QuadraticTetrahedralElement: [[0.3, 0.4, 0.1], [0.1, 0.25, 0.3], [0.2, 0.2, 0.2]],
}


@pytest.mark.parametrize('element_type', list(QUADRATIC_POINTS))
def test_p2_shape_hessians_match_finite_differences(element_type):
    """The analytic P2 Hessians are the derivative of the shape gradients: a central
    difference of `shape_gradients` reproduces them, so the hand-written constants are
    the real second derivatives, not a transcription slip."""
    points = np.array(QUADRATIC_POINTS[element_type])
    dim = points.shape[1]
    hessians = element_type.shape_hessians(points)   # (n_pts, N, dim, dim)

    h = 1e-6
    fd = np.zeros_like(hessians)
    for j, step in enumerate(np.eye(dim) * h):
        plus = element_type.shape_gradients(points + step)    # (n_pts, N, dim)
        minus = element_type.shape_gradients(points - step)
        fd[..., j] = (plus - minus) / (2 * h)

    assert np.allclose(hessians, fd, atol=1e-6)
    # Symmetric, as a Hessian must be.
    assert np.allclose(hessians, np.swapaxes(hessians, -1, -2))


@pytest.mark.parametrize('element_type', list(QUADRATIC_POINTS))
def test_p2_shape_gradients_match_finite_differences(element_type):
    """The gradients are the derivative of the shape values, the tier below the
    Hessians: a wrong sign in one hat's gradient shows here directly."""
    points = np.array(QUADRATIC_POINTS[element_type])
    dim = points.shape[1]
    gradients = element_type.shape_gradients(points)

    h = 1e-6
    fd = np.zeros_like(gradients)
    for j, step in enumerate(np.eye(dim) * h):
        fd[..., j] = (element_type.shape_values(points + step)
                      - element_type.shape_values(points - step)) / (2 * h)

    assert np.allclose(gradients, fd, atol=1e-7)


def test_p2_tet_edge_nodes_sit_at_the_midpoints_they_are_named_for():
    """`EDGE_NODES` is the element's statement of its own ordering, and
    `reference_nodes` must agree with it: node `4 + k` is the midpoint of edge `k`."""
    nodes = QuadraticTetrahedralElement.reference_nodes()
    assert nodes.shape == (10, 3)
    for k, (i, j) in enumerate(QuadraticTetrahedralElement.EDGE_NODES):
        np.testing.assert_allclose(nodes[4 + k], 0.5 * (nodes[i] + nodes[j]))


def test_element_field_hessian_recovers_a_quadratic_fields_curvature():
    """u = a x^2 + b xy + c y^2 has constant Hessian [[2a, b], [b, 2c]]; the space
    recovers exactly that on every element, the physical mapping of the reference
    Hessian through the inverse Jacobian working out."""
    mesh = box_mesh(corners=[[0, 0], [2, 1]], resolution=(4, 3))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)
    a, b, c = 1.5, -0.7, 2.0
    x, y = space.node_coords[:, 0], space.node_coords[:, 1]
    u = a * x**2 + b * x * y + c * y**2

    hessian = space.element_hessian(u[space.element_nodes])   # (n_el, 2, 2)

    expected = np.array([[2 * a, b], [b, 2 * c]])
    assert np.allclose(hessian, expected)


def test_element_field_hessian_recovers_a_3d_quadratic_fields_curvature():
    """The same on tets: every second derivative of a general 3D quadratic, including
    the three mixed ones, comes back on every element."""
    mesh = box_mesh(corners=[[0, 0, 0], [2, 1, 1]], resolution=(3, 3, 3))
    space = FunctionSpace(mesh, QuadraticTetrahedralElement, n_components=1)
    a, b, c, d, e, f = 1.5, -0.7, 2.0, 0.4, -1.1, 0.9
    x, y, z = space.node_coords.T
    u = a * x**2 + b * y**2 + c * z**2 + d * x * y + e * x * z + f * y * z

    hessian = space.element_hessian(u[space.element_nodes])   # (n_el, 3, 3)

    expected = np.array([[2 * a, d, e], [d, 2 * b, f], [e, f, 2 * c]])
    assert np.allclose(hessian, expected)
