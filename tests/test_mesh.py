"""Mesh geometry: topology queries over vertices, elements, and facets."""
import numpy as np
import pytest

from fem.mesh.mesh import Mesh


def _two_triangle_square() -> Mesh:
    """The unit square as two triangles sharing the diagonal 0--2."""
    return Mesh(
        vertices=[[0, 0], [1, 0], [1, 1], [0, 1]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )


def test_edges_are_every_vertex_pair_of_each_simplex():
    """For a linear simplex the edge set is the pairs of its nodes, so two triangles give
    five distinct edges."""
    mesh = _two_triangle_square()
    assert len(mesh.edges) == 5
    assert (0, 2) in {tuple(e) for e in mesh.edges}


def test_interior_edge_maps_to_both_its_elements():
    """The shared diagonal belongs to two elements; a boundary side to one."""
    mesh = _two_triangle_square()
    assert sorted(mesh.edge_to_elements[(0, 2)]) == [0, 1]
    assert mesh.edge_to_elements[(0, 1)] == [0]


def test_edge_elements_pairs_each_edge_with_its_elements():
    """The batched form: each edge in `edges` gets both its elements, with -1 in
    the second slot of a boundary edge. Consistent with `edge_to_elements`."""
    mesh = _two_triangle_square()
    by_edge = {tuple(e): ee for e, ee in zip(mesh.edges, mesh.edge_elements)}

    assert sorted(by_edge[(0, 2)]) == [0, 1]           # shared diagonal: two elements
    assert by_edge[(0, 1)][1] == -1                    # boundary side: one element
    assert by_edge[(0, 1)][0] == 0
    for edge, ee in by_edge.items():
        both = ee[ee >= 0].tolist()
        assert sorted(both) == sorted(mesh.edge_to_elements[edge])


def test_element_neighbours_are_those_sharing_an_edge():
    mesh = _two_triangle_square()
    assert mesh.element_neighbours == [[1], [0]]


def test_edge_extraction_refuses_non_simplex_elements():
    """Pairing every node spells out the edges only for a linear simplex."""
    with pytest.raises(NotImplementedError, match='linear simplices'):
        Mesh(
            vertices=np.zeros((6, 2)),
            elements=[[0, 1, 2, 3, 4, 5]],
            boundary=[[0, 1]],
        )


def test_arrays_are_read_only():
    """A mesh is shared by everything built on it, and its derived tables are cached,
    so a change in place is refused rather than silently leaving them stale."""
    mesh = _two_triangle_square()
    with pytest.raises(ValueError):
        mesh.vertices[0] = [9.0, 9.0]
    with pytest.raises(ValueError):
        mesh.elements[0, 0] = 3
    with pytest.raises(AttributeError):
        mesh.vertices = np.zeros((4, 2))  # type: ignore[misc]


# --- construction ---

def test_boundary_is_derived_when_not_given():
    mesh = Mesh(vertices=[[0, 0], [1, 0], [1, 1], [0, 1]], elements=[[0, 1, 2], [0, 2, 3]])
    facets = {tuple(f) for f in mesh.boundary}
    assert facets == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert list(mesh.boundary_idxs) == [0, 1, 2, 3]


def test_a_given_boundary_keeps_its_order():
    """A reloaded mesh passes its facets so `boundary_curves` stay aligned with them."""
    given = [[2, 3], [0, 1], [1, 2], [3, 0]]
    mesh = Mesh([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2], [0, 2, 3]], given)
    assert mesh.boundary.tolist() == given


# --- sizes, extent, and element geometry ---

def test_sizes_and_dimensions():
    mesh = _two_triangle_square()
    assert (mesh.n_vertices, mesh.n_elements) == (4, 2)
    assert (mesh.spatial_dim, mesh.element_dim) == (2, 2)
    assert repr(mesh) == 'Mesh(4 vertices, 2 triangles, 4 boundary facets, 2D)'


def test_bounds():
    lower, upper = _two_triangle_square().bounds
    assert np.array_equal(lower, [0, 0]) and np.array_equal(upper, [1, 1])


def test_measures_in_each_dimension():
    line = Mesh([[0.0], [0.5], [2.0]], [[0, 1], [1, 2]])
    assert np.allclose(line.element_measures, [0.5, 1.5])
    assert line.measure == pytest.approx(2.0)

    square = _two_triangle_square()
    assert np.allclose(square.element_measures, [0.5, 0.5])
    assert square.area == pytest.approx(1.0)

    tet = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2, 3]])
    assert tet.measure == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        tet.area


def test_measure_of_a_triangle_embedded_in_3d():
    """The Gram formula does not need the element to span the space it sits in."""
    mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    assert mesh.measure == pytest.approx(0.5)


def test_centroids():
    mesh = _two_triangle_square()
    assert np.allclose(mesh.centroids, [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])


def test_min_angle_of_a_right_isoceles_split_is_45():
    assert _two_triangle_square().min_angle == pytest.approx(45.0)


# --- new meshes from this one ---

def test_displaced_moves_the_vertices_and_keeps_the_topology():
    mesh = _two_triangle_square()
    shift = np.tile([1.0, 2.0], (4, 1))
    moved = mesh.displaced(shift, scale=0.5)
    assert np.allclose(moved.vertices, mesh.vertices + [0.5, 1.0])
    assert np.array_equal(moved.elements, mesh.elements)
    assert np.array_equal(moved.boundary, mesh.boundary)
    assert np.array_equal(mesh.vertices[0], [0, 0]), 'the source is untouched'


def test_displaced_reads_the_vertex_block_of_a_longer_dof_vector():
    """A P2 vector lists the edge nodes after the vertices; the warp is its P1 part."""
    mesh = _two_triangle_square()
    dofs = np.concatenate([np.tile([1.0, 0.0], 4), np.full(2 * 5, 99.0)])
    assert np.allclose(mesh.displaced(dofs).vertices, mesh.vertices + [1.0, 0.0])
    with pytest.raises(ValueError):
        mesh.displaced(np.zeros((3, 2)))


def test_refined_splits_every_element_by_default():
    mesh = _two_triangle_square()
    assert mesh.refined().n_elements == 8
    assert mesh.refined([0]).n_elements > 2


# --- input validation ---
# A malformed array should fail here, at the entry point for user data, with a
# named error -- not much later inside element geometry with an opaque shape error.

def test_rejects_wrong_rank_vertices():
    with pytest.raises(ValueError, match='vertices must be a 2D'):
        Mesh(vertices=[0, 1, 2], elements=[[0, 1]], boundary=[[0]])


def test_rejects_out_of_range_element_index():
    with pytest.raises(ValueError, match='element node indices'):
        Mesh(
            vertices=[[0, 0], [1, 0], [1, 1]],
            elements=[[0, 1, 3]],  # node 3 does not exist
            boundary=[[0, 1]],
        )


def test_rejects_boundary_facet_of_wrong_width():
    with pytest.raises(ValueError, match='boundary facet'):
        Mesh(
            vertices=[[0, 0], [1, 0], [1, 1]],
            elements=[[0, 1, 2]],
            boundary=[[0, 1, 2]],  # a triangle's facet is an edge (2 nodes), not 3
        )
