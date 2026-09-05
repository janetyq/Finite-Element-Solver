"""Unit tests for the pure geometry and material helpers, against closed-form answers."""
import numpy as np
import pytest

from fem.elements import LinearTetrahedralElement, LinearTriangleElement
from fem.mesh.mesh import Mesh
from fem.mesh.pslg import point_in_polygon, polygon_area
from fem.mesh.ruppert import circumcenter
from fem.physics.forms import MassForm
from fem.physics.materials import Enu_to_Lame, Lame_to_Enu


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
        assert polygon_area(square) == pytest.approx(1.0)

    def test_unit_triangle(self):
        triangle = np.array([[0, 0], [1, 0], [0, 1]])
        assert polygon_area(triangle) == pytest.approx(0.5)

    def test_invariant_to_translation(self):
        triangle = np.array([[0, 0], [1, 0], [0, 1]]) + np.array([5.0, -3.0])
        assert polygon_area(triangle) == pytest.approx(0.5)

    def test_triangle_in_3d(self):
        # Same triangle as test_unit_triangle, embedded in the z = 0 plane.
        triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert polygon_area(triangle) == pytest.approx(0.5)

    def test_tilted_triangle_in_3d(self):
        # Legs of length 1 and sqrt(2), meeting at a right angle.
        triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
        assert polygon_area(triangle) == pytest.approx(0.5 * np.sqrt(2))

    def test_general_3d_polygon_is_refused(self):
        # Needs Newell's method; refuse rather than return a wrong number.
        quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        with pytest.raises(NotImplementedError):
            polygon_area(quad)


class TestPointInPolygon:
    def test_inside_and_outside(self):
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        assert point_in_polygon(np.array([1.0, 1.0]), square)
        assert not point_in_polygon(np.array([3.0, 1.0]), square)


class TestCircumcenter:
    def test_right_triangle(self):
        # Circumcenter of a right triangle is the midpoint of its hypotenuse.
        triangle = np.array([[0, 0], [2, 0], [0, 2]])
        center = circumcenter(triangle)
        assert center[0] == pytest.approx(1.0)
        assert center[1] == pytest.approx(1.0)

    @pytest.mark.parametrize('triangle', [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9]]),          # well shaped
        np.array([[0.0, 0.0], [1.0, 1e-13], [0.5, 0.9]]),        # near-horizontal edge
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1e-6]]),         # sliver
        np.array([[1e6, 1e6], [1e6 + 1, 1e6], [1e6, 1e6 + 1]]),  # far from the origin
    ])
    def test_is_equidistant_from_all_three_vertices(self, triangle):
        """The defining property of the circumcenter, including a near-horizontal edge."""
        center = circumcenter(triangle)
        radii = np.linalg.norm(triangle - center, axis=1)
        assert radii.max() - radii.min() == pytest.approx(0.0, abs=1e-9 * radii.mean())

    def test_batched_matches_one_at_a_time(self):
        rng = np.random.default_rng(0)
        triangles = rng.random((20, 3, 2))
        batch = circumcenter(triangles)
        singly = np.array([circumcenter(t) for t in triangles])
        assert batch.shape == (20, 2)
        np.testing.assert_allclose(batch, singly)

    def test_degenerate_triangle_is_refused(self):
        """Collinear points have no circumcircle. Refuse rather than return an
        infinity that would be inserted into a mesh as a vertex."""
        with pytest.raises(ValueError, match='degenerate'):
            circumcenter(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


class TestMassMatrix:
    """MassForm repeats the scalar element mass matrix once per component."""

    ELEMENTS = [
        (LinearTriangleElement, np.array([[0.0, 0], [1, 0], [0, 1]]), 2),
        (LinearTetrahedralElement,
         np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3),
    ]

    @pytest.mark.parametrize('element_type, vertices, n_components', ELEMENTS)
    def test_is_scalar_matrix_per_component(self, element_type, vertices, n_components):
        geometry = element_type.geometry(vertices[None])
        scalar = element_type.reference_mass_matrix() * geometry.volumes[0]
        vector = MassForm(n_components).element_matrices(geometry)[0]
        assert np.allclose(vector, np.kron(scalar, np.eye(n_components)))

    @pytest.mark.parametrize('element_type, vertices, n_components', ELEMENTS)
    def test_integrates_a_constant_force_in_every_component(
        self, element_type, vertices, n_components,
    ):
        # int_element 1 dV == volume, componentwise.
        geometry = element_type.geometry(vertices[None])
        mass = MassForm(n_components).element_matrices(geometry)[0]
        for component in range(n_components):
            load = np.zeros((element_type.N, n_components))
            load[:, component] = 1.0
            assert (mass @ load.flatten()).sum() == pytest.approx(geometry.volumes[0])


class TestDimensions:
    """spatial_dim and reference_dim coincide for planar and tet meshes; the surface case
    separates them."""

    def test_planar_triangle_mesh(self):
        mesh = Mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], [[0, 1], [1, 2], [2, 0]])
        assert mesh.spatial_dim == 2
        geometry = LinearTriangleElement.geometry(mesh.vertices[mesh.elements])
        assert (geometry.reference_dim, geometry.spatial_dim) == (2, 2)

    def test_tet_element(self):
        assert LinearTetrahedralElement.reference_dim() == 3
        tet = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        geometry = LinearTetrahedralElement.geometry(tet)
        assert (geometry.reference_dim, geometry.spatial_dim) == (3, 3)

    def test_surface_mesh_separates_them(self):
        # A triangle embedded in 3D: 3 ambient coordinates, still a 2D element.
        mesh = Mesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 1]], [[0, 1, 2]], [[0, 1], [1, 2], [2, 0]],
        )
        assert mesh.spatial_dim == 3
        geometry = LinearTriangleElement.geometry(mesh.vertices[mesh.elements])
        # The gradients carry all three ambient components, but the element is still 2D.
        assert (geometry.reference_dim, geometry.spatial_dim) == (2, 3)
        # (n_el, n_qp, N, spatial): one element, one quad point, 3 nodes, 3D grads.
        assert geometry.grad_phi.shape == (1, 1, 3, 3)
