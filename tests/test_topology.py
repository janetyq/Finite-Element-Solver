"""Tests for dimension-general mesh topology: edge and boundary-facet extraction.

The 1D and 3D paths had no correctness coverage, which is how the triangle-only
`range(3)` assumption survived in both `Mesh._get_all_edges` and
`get_boundary_from_vertices_elements`.
"""
import numpy as np
import pytest

from fem.geometry import get_boundary_from_vertices_elements
from fem.mesh.mesh import Mesh


# --- edges ---

def test_line_mesh_edges():
    """A 2-node line element is its own single edge. This used to IndexError:
    the old loop indexed element[2] on a 2-node element."""
    mesh = Mesh(
        vertices=[[0.0], [1.0], [2.0]],
        elements=[[0, 1], [1, 2]],
        boundary=[[0], [2]],
    )
    assert {tuple(e) for e in mesh.edges} == {(0, 1), (1, 2)}


def test_triangle_mesh_edges():
    """Two triangles sharing a diagonal: 5 distinct edges, not 6."""
    mesh = Mesh(
        vertices=[[0, 0], [1, 0], [1, 1], [0, 1]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [0, 3]],
    )
    assert {tuple(e) for e in mesh.edges} == {(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)}


def test_tet_mesh_edges():
    """A tet has 6 edges. The old cyclic `range(3)` found only 3, silently
    dropping every edge touching the fourth node."""
    mesh = Mesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        elements=[[0, 1, 2, 3]],
        boundary=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    )
    assert {tuple(e) for e in mesh.edges} == {
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    }


def test_non_simplex_elements_are_rejected():
    """Pairing every node is the edge set only for linear simplices, so a
    higher-node element must fail loudly rather than invent edges."""
    with pytest.raises(NotImplementedError):
        Mesh(
            vertices=np.zeros((5, 2)),
            elements=[[0, 1, 2, 3, 4]],
            boundary=[[0, 1]],
        )


# --- boundary facets ---

def test_boundary_of_two_triangles():
    """The shared diagonal is interior; the four outer edges are boundary."""
    elements = [[0, 1, 2], [0, 2, 3]]
    boundary = {tuple(f) for f in get_boundary_from_vertices_elements(elements)}
    assert boundary == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_boundary_of_single_tet():
    """Every face of a lone tet is a boundary face -- 4 triangles, not edges."""
    boundary = {tuple(f) for f in get_boundary_from_vertices_elements([[0, 1, 2, 3]])}
    assert boundary == {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}


def test_boundary_of_two_tets_drops_shared_face():
    """Two tets glued on face (0,1,2): that face is interior, 6 faces remain."""
    boundary = {
        tuple(f)
        for f in get_boundary_from_vertices_elements([[0, 1, 2, 3], [0, 1, 2, 4]])
    }
    assert (0, 1, 2) not in boundary
    assert len(boundary) == 6


def test_boundary_of_rect_mesh_is_the_perimeter(make_unit_square):
    """End-to-end on the real generator: every boundary vertex sits on the
    perimeter of the unit square, and none of the interior ones do."""
    femesh = make_unit_square(6)
    on_perimeter = np.isclose(femesh.vertices, 0) | np.isclose(femesh.vertices, 1)
    assert on_perimeter[femesh.boundary_idxs].any(axis=1).all()
    assert len(femesh.boundary) == 4 * (6 - 1)
