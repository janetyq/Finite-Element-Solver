"""The structured builders: a box in each dimension, and the annulus."""
import numpy as np
import pytest

from fem.mesh.structured import annulus_mesh, box_mesh


def test_box_mesh_in_1d_is_a_chain_of_lines():
    mesh = box_mesh(corners=[[0.0], [3.0]], resolution=(4,))
    assert mesh.element_dim == 1 and mesh.spatial_dim == 1
    assert np.allclose(mesh.vertices[:, 0], [0, 1, 2, 3])
    assert mesh.elements.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert sorted(mesh.boundary_idxs.tolist()) == [0, 3]
    assert mesh.measure == pytest.approx(3.0)


def test_box_mesh_in_2d_covers_the_rectangle():
    mesh = box_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 3))
    assert mesh.element_dim == 2
    assert mesh.n_elements == 2 * 4 * 2
    assert mesh.area == pytest.approx(2.0)
    assert np.allclose(mesh.bounds[0], [0, 0]) and np.allclose(mesh.bounds[1], [2, 1])


def test_box_mesh_in_3d_is_conforming_and_fills_the_box():
    mesh = box_mesh(corners=[[0, 0, 0], [1, 2, 1]], resolution=(3, 4, 3))
    assert mesh.element_dim == 3
    assert mesh.n_elements == 6 * 2 * 3 * 2
    assert mesh.measure == pytest.approx(2.0)
    # Every face belongs to one element (boundary) or two (interior), never more.
    faces = np.sort(mesh.elements[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]
                    .reshape(-1, 3), axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    assert set(counts.tolist()) <= {1, 2}


def test_box_mesh_refuses_mismatched_dimensions():
    with pytest.raises(ValueError):
        box_mesh(corners=[[0, 0], [1, 1]], resolution=(4,))
    with pytest.raises(ValueError):
        box_mesh(corners=[[0, 0, 0, 0], [1, 1, 1, 1]], resolution=(2, 2, 2, 2))


def test_annulus_rims_carry_their_circles():
    mesh = annulus_mesh(1.0, 2.0, n_radial=4, n_theta=12)
    assert mesh.boundary_curves is not None
    radii = {round(float(np.hypot(*mesh.vertices[f[0]])), 6) for f in mesh.boundary}
    assert radii == {1.0, 2.0}
    assert all(curve is not None for curve in mesh.boundary_curves)
