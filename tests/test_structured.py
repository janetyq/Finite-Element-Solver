"""The structured builder: a box in each dimension."""
import numpy as np
import pytest

from fem.mesh.structured import box_mesh


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


@pytest.mark.parametrize('tet_split', ['regular', 'kuhn'])
def test_box_mesh_in_3d_is_conforming(tet_split):
    """Every face belongs to one element (boundary) or two (interior), never more. The
    regular split alternates two mirror forms, so this also checks its checkerboard
    keeps neighbours agreeing on shared faces."""
    mesh = box_mesh(corners=[[0, 0, 0], [1, 2, 1]], resolution=(3, 4, 3), tet_split=tet_split)
    assert mesh.element_dim == 3
    faces = np.sort(mesh.elements[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]
                    .reshape(-1, 3), axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    assert set(counts.tolist()) <= {1, 2}


@pytest.mark.parametrize('n', [2, 3, 5])
@pytest.mark.parametrize('tet_split, per_cell', [('regular', 5), ('kuhn', 6)])
def test_box_mesh_tiles_the_cube_exactly(n, tet_split, per_cell):
    """Each decomposition partitions the cube: element volumes summing to 1 catches a
    mis-numbered corner, which would otherwise produce overlapping or inverted tets. The
    regular split gives five tets per cell, Kuhn six."""
    mesh = box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(n, n, n), tet_split=tet_split)
    assert mesh.n_vertices == n**3
    assert mesh.n_elements == per_cell * (n - 1)**3
    assert mesh.measure == pytest.approx(1.0)


def test_box_mesh_rejects_an_unknown_tet_split():
    with pytest.raises(ValueError, match='tet_split'):
        box_mesh(corners=[[0, 0, 0], [1, 1, 1]], resolution=(3, 3, 3), tet_split='banana')  # type: ignore[arg-type]


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



_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
_TET5_EVEN = [(0, 3, 5, 6), (0, 1, 3, 5), (0, 2, 3, 6), (0, 4, 5, 6), (3, 5, 6, 7)]
_TET5_ODD = [(1, 2, 4, 7), (0, 1, 2, 4), (1, 2, 3, 7), (1, 4, 5, 7), (2, 4, 6, 7)]


def _cell_loop_box(nx, ny, nz, tet_split):
    """A decomposition written as the loop over cells it describes, the readable form the
    vectorized `box_mesh` must reproduce vertex for vertex and tet for tet. Kuhn splits
    every cell the same way; the regular split alternates its two mirror forms on the
    (i + j + k) checkerboard."""
    def node(i, j, k):
        return (i * ny + j) * nz + k

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner = [node(i + (c >> 2 & 1), j + (c >> 1 & 1), k + (c & 1)) for c in range(8)]
                if tet_split == 'kuhn':
                    tets = _KUHN
                else:
                    tets = _TET5_EVEN if (i + j + k) % 2 == 0 else _TET5_ODD
                elements.extend([[corner[c] for c in tet] for tet in tets])
    return np.array(elements)


def _cell_loop_rect(nx, ny):
    """The alternating-diagonal split as the loop over cells it describes."""
    def node(i, j):
        return j * nx + i

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if (i + j) % 2 == 0:
                elements += [[node(i, j), node(i+1, j), node(i+1, j+1)], [node(i, j), node(i+1, j+1), node(i, j+1)]]
            else:
                elements += [[node(i, j), node(i+1, j), node(i, j+1)], [node(i+1, j), node(i+1, j+1), node(i, j+1)]]
    return np.array(elements)


@pytest.mark.parametrize('tet_split', ['regular', 'kuhn'])
def test_box_mesh_connectivity_is_the_cell_loop_written_out(tet_split):
    """The vectorized builders reproduce the per-cell loops exactly: the same vertex
    order (x fastest in 2D, z fastest in 3D) and the same element order, so a mesh
    fingerprint recorded on one is valid on the other."""
    rect = box_mesh(corners=[[0, 0], [2, 1]], resolution=(6, 4))
    np.testing.assert_array_equal(rect.elements, _cell_loop_rect(6, 4))
    np.testing.assert_array_equal(rect.vertices[:6, 1], 0.0)          # the first row is y = 0, x varying

    box = box_mesh(corners=[[0, 0, 0], [2, 1, 1]], resolution=(5, 4, 3), tet_split=tet_split)
    np.testing.assert_array_equal(box.elements, _cell_loop_box(5, 4, 3, tet_split))
    np.testing.assert_array_equal(box.vertices[:3, :2], 0.0)          # the first column is x = y = 0, z varying
