"""Unit tests for the pure geometry / material helper functions.

These are deterministic, dependency-free functions with known closed-form
answers, so they make a good first regression net.
"""
import numpy as np
import pytest

from fem.materials import Enu_to_Lame, Lame_to_Enu
from fem.elements import LinearTetrahedralElement, LinearTriangleElement
from fem.geometry import (
    calculate_polygon_area,
    calculate_tetrahedron_volume,
    calculate_circumcenter,
    point_in_polygon,
)
from fem.mesh.mesh import Mesh


class TestLameConversion:
    def test_round_trip(self):
        # E, nu -> mu, lambda -> E, nu should recover the inputs
        E, nu = 200.0, 0.3
        mu, lamb = Enu_to_Lame(E, nu)
        E_back, nu_back = Lame_to_Enu(mu, lamb)
        assert E_back == pytest.approx(E)
        assert nu_back == pytest.approx(nu)

    def test_known_value(self):
        # For nu = 0, mu = E/2 and lambda = 0
        mu, lamb = Enu_to_Lame(100.0, 0.0)
        assert mu == pytest.approx(50.0)
        assert lamb == pytest.approx(0.0)


class TestPolygonArea:
    def test_unit_square(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert calculate_polygon_area(square) == pytest.approx(1.0)

    def test_unit_triangle(self):
        triangle = np.array([[0, 0], [1, 0], [0, 1]])
        assert calculate_polygon_area(triangle) == pytest.approx(0.5)

    def test_invariant_to_translation(self):
        triangle = np.array([[0, 0], [1, 0], [0, 1]]) + np.array([5.0, -3.0])
        assert calculate_polygon_area(triangle) == pytest.approx(0.5)


class TestTetrahedronVolume:
    def test_unit_tetrahedron(self):
        tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert calculate_tetrahedron_volume(tet) == pytest.approx(1.0 / 6.0)


class TestPointInPolygon:
    def test_inside_and_outside(self):
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        assert point_in_polygon(np.array([1.0, 1.0]), square)
        assert not point_in_polygon(np.array([3.0, 1.0]), square)


class TestCircumcenter:
    def test_right_triangle(self):
        # Circumcenter of a right triangle is the midpoint of its hypotenuse.
        triangle = np.array([[0, 0], [2, 0], [0, 2]])
        center = calculate_circumcenter(triangle)
        assert center[0] == pytest.approx(1.0)
        assert center[1] == pytest.approx(1.0)


class TestDimensions:
    """spatial_dim and reference_dim must stay distinguishable.

    They coincide for a planar triangle mesh and a tet mesh, which is why one
    number served for both. The surface case is what separates them.
    """

    def test_planar_triangle_mesh(self):
        mesh = Mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], [[0, 1], [1, 2], [2, 0]])
        assert mesh.spatial_dim == 2
        assert LinearTriangleElement(mesh.vertices).reference_dim == 2

    def test_tet_element(self):
        assert LinearTetrahedralElement.N - 1 == 3
        tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert LinearTetrahedralElement(tet).reference_dim == 3

    def test_surface_mesh_separates_them(self):
        # A triangle embedded in 3D: 3 ambient coordinates, still a 2D element.
        mesh = Mesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 1]], [[0, 1, 2]], [[0, 1], [1, 2], [2, 0]],
        )
        assert mesh.spatial_dim == 3
        assert LinearTriangleElement.N - 1 == 2
