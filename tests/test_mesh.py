"""Mesh geometry: topology queries over vertices, elements, and facets.

The element<->vertex field projections that used to live here moved to
`FunctionSpace` (see `tests/test_space.py`). They are discretization operations
rather than geometric ones, and the correct projection is weighted by element
measure -- which the space owns and the mesh does not.
"""
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
    """For a linear simplex the edge set is exactly the pairs of its nodes, so
    two triangles give five distinct edges -- four sides plus the shared diagonal."""
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
    """Pairing every node only spells out the edges for a linear simplex -- a
    quadratic element's midside nodes would invent edges that do not exist."""
    with pytest.raises(NotImplementedError, match='linear simplices'):
        Mesh(
            vertices=np.zeros((6, 2)),
            elements=[[0, 1, 2, 3, 4, 5]],
            boundary=[[0, 1]],
        )


def test_copy_is_independent_of_its_source():
    mesh = _two_triangle_square()
    duplicate = mesh.copy()
    duplicate.vertices[0] = [9.0, 9.0]
    assert np.allclose(mesh.vertices[0], [0.0, 0.0])


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
