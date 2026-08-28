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


def test_box_mesh_in_3d_is_conforming():
    """Every face belongs to one element (boundary) or two (interior), never more."""
    mesh = box_mesh(corners=[[0, 0, 0], [1, 2, 1]], resolution=(3, 4, 3))
    assert mesh.element_dim == 3
    faces = np.sort(mesh.elements[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]
                    .reshape(-1, 3), axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    assert set(counts.tolist()) <= {1, 2}


@pytest.mark.parametrize('n', [2, 3, 5])
def test_box_mesh_tiles_the_cube_exactly(n):
    """Kuhn's decomposition gives 6 tets per cell, and they must partition the
    cube: element volumes summing to 1 catches a mis-numbered corner, which
    would otherwise produce overlapping or inverted tets."""
    mesh = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(n, n, n))
    assert mesh.n_vertices == n**3
    assert mesh.n_elements == 6 * (n - 1)**3
    assert mesh.measure == pytest.approx(1.0)


def test_box_mesh_boundary_is_the_cube_surface():
    """Every boundary vertex lies on a face of the cube and every interior one does not, so
    the cells agree on their shared diagonals."""
    n = 4
    mesh = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(n, n, n))
    on_face = np.isclose(mesh.vertices, 0) | np.isclose(mesh.vertices, 1)

    boundary_idxs = set(int(i) for i in mesh.boundary_idxs)
    assert on_face[mesh.boundary_idxs].any(axis=1).all()
    interior = set(range(len(mesh.vertices))) - boundary_idxs
    assert len(interior) == (n - 2)**3
    assert not on_face[sorted(interior)].any()


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
