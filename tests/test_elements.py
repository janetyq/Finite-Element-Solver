"""The element primitives: shape functions and their derivatives, and the fields a space
builds from them.
"""
import numpy as np

from fem.elements import QuadraticTriangleElement
from fem.mesh.structured import box_mesh
from fem.space import FunctionSpace


def test_p2_shape_hessians_match_finite_differences():
    """The analytic P2 Hessians are the derivative of the shape gradients: a central
    difference of `shape_gradients` reproduces them, so the hand-written constants are
    the real second derivatives, not a transcription slip."""
    points = np.array([[0.3, 0.4], [0.1, 0.25], [0.5, 0.2]])
    hessians = QuadraticTriangleElement.shape_hessians(points)   # (n_pts, 6, 2, 2)

    h = 1e-6
    fd = np.zeros_like(hessians)
    for j, step in enumerate(np.eye(2) * h):
        plus = QuadraticTriangleElement.shape_gradients(points + step)    # (n_pts, 6, 2)
        minus = QuadraticTriangleElement.shape_gradients(points - step)
        fd[..., j] = (plus - minus) / (2 * h)

    assert np.allclose(hessians, fd, atol=1e-6)
    # Symmetric, as a Hessian must be.
    assert np.allclose(hessians, np.swapaxes(hessians, -1, -2))


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
