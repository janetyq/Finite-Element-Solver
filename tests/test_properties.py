"""Property-based tests: invariants that must hold over a whole family of inputs, with
`hypothesis` drawing the family rather than a fixed seed doing it by hand.

Three properties the solver rests on: a tensor reduction does not move when the frame
turns; a P1/P2 element is a partition of unity everywhere; and the assembled operators
do not depend on the order a mesh lists an element's nodes (the elements take `abs(det)`,
so a reversed element is the same element). Each is checked over many generated cases.
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fem.conditions import Conditions
from fem.elements import (
    LinearTetrahedralElement,
    LinearTriangleElement,
    QuadraticTriangleElement,
)
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.physics.forms import DiffusionForm
from fem.post.invariants import frobenius, max_shear, pressure, trace, von_mises
from fem.problem import LinearProblem
from fem.space import FunctionSpace

_coord = st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False)
_entry = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_angle = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)


def _rotation_3d(a: float, b: float, c: float) -> np.ndarray:
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


@settings(max_examples=60, deadline=None)
@given(a=_coord, b=_coord, c=_coord)
def test_shape_functions_are_a_partition_of_unity(a, b, c):
    """The P1 and P2 shape functions sum to one at every point (so a constant field is
    reproduced) and their gradients sum to zero (so a constant has no gradient), which
    holds off the reference simplex too since the functions are polynomial."""
    for element, point in (
        (LinearTriangleElement, [a, b]),
        (LinearTetrahedralElement, [a, b, c]),
        (QuadraticTriangleElement, [a, b]),
    ):
        points = np.array([point])
        np.testing.assert_allclose(element.shape_values(points).sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(element.shape_gradients(points).sum(axis=1), 0.0, atol=1e-12)


@settings(max_examples=60, deadline=None)
@given(vals=st.lists(_entry, min_size=6, max_size=6), a=_angle, b=_angle, c=_angle)
def test_tensor_reductions_are_rotation_invariant(vals, a, b, c):
    """A scalar reduction of a stress tensor is the same number in any frame: build a
    random symmetric tensor, read it in a randomly rotated frame, and the invariants do
    not move. The fixed-seed version lives in test_invariants.py; this covers the family."""
    sxx, syy, szz, sxy, syz, sxz = vals
    tensor = np.array([[[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]]])
    R = _rotation_3d(a, b, c)
    rotated = np.einsum('ij,ejk,lk->eil', R, tensor, R)

    for reduce in (frobenius, trace, von_mises, max_shear, pressure):
        np.testing.assert_allclose(reduce(rotated), reduce(tensor), rtol=1e-8, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(delta=arrays(np.float64, (25, 2),
                    elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False)))
def test_reversed_node_order_gives_the_same_operators(delta):
    """Each element takes the absolute Jacobian determinant, so reversing the order a
    mesh lists an element's nodes flips the orientation without changing the geometry:
    the assembled global stiffness and mass are unchanged. A perturbed 5x5 grid (25
    vertices, h = 0.25) varies the geometry while keeping the topology non-degenerate."""
    base = box_mesh(corners=[[0, 0], [1, 1]], resolution=(5, 5))
    vertices = base.vertices + delta
    forward = Mesh(vertices, base.elements, base.boundary)
    reversed_order = Mesh(vertices, base.elements[:, ::-1], base.boundary)

    def stiffness(mesh):
        return LinearProblem(FunctionSpace(mesh), DiffusionForm(), Conditions()).tangent(None).toarray()

    np.testing.assert_allclose(stiffness(forward), stiffness(reversed_order), atol=1e-12)
    np.testing.assert_allclose(
        FunctionSpace(forward).mass_matrix.toarray(),
        FunctionSpace(reversed_order).mass_matrix.toarray(),
        atol=1e-12,
    )
