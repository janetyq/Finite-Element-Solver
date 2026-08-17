"""Structured simplex mesh generators for axis-aligned rectangles and boxes.

These produce conforming triangulations (2D) and tetrahedralizations (3D) on a
regular grid, the meshes convergence studies want: uniform refinement without the
element-quality variation an unstructured generator introduces. Distinct in intent
from `ruppert`, whose Delaunay refinement meshes arbitrary polygonal outlines.
"""
from collections.abc import Sequence

import numpy as np

from fem.geometry import get_boundary_from_vertices_elements
from fem.mesh.mesh import Mesh
from fem.typing import FloatArray


def create_rect_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured triangulation of the axis-aligned rectangle spanned by
    `corners` ((x0, y0), (x1, y1)), with `resolution` (nx, ny) nodes per axis.'''
    x_range = np.linspace(corners[0][0], corners[1][0], resolution[0])
    y_range = np.linspace(corners[0][1], corners[1][1], resolution[1])

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*resolution[0] + i

    for i in range(resolution[0]-1):
        for j in range(resolution[1]-1):
            if (i + j) % 2 == 0:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            else:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i, j+1)])
                elements.append([get_index(i+1, j), get_index(i+1, j+1), get_index(i, j+1)])

    boundary = get_boundary_from_vertices_elements(elements)
    return Mesh(vertices, elements, boundary)


def create_box_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured tetrahedralization of the axis-aligned box spanned by
    `corners` ((x0, y0, z0), (x1, y1, z1)), with `resolution` (nx, ny, nz) nodes
    per axis.

    Each grid cell is split into six tets by Kuhn's decomposition: every tet
    contains the cell's main diagonal (corner 000 to 111), one per permutation
    of the three axes. Because every cell splits along the same diagonal
    direction, neighbouring cells agree on their shared faces and the mesh is
    conforming, the property that makes this usable for convergence studies,
    where an unstructured generator would confound the refinement rate.
    '''
    nx, ny, nz = resolution
    x_range = np.linspace(corners[0][0], corners[1][0], nx)
    y_range = np.linspace(corners[0][1], corners[1][1], ny)
    z_range = np.linspace(corners[0][2], corners[1][2], nz)

    vertices = np.array([[x, y, z] for x in x_range for y in y_range for z in z_range])

    def get_index(i, j, k):
        return (i*ny + j)*nz + k

    # Corners of a cell, indexed by the bits of (di, dj, dk): corner 5 is
    # (1, 0, 1). The six tets below are written against that numbering.
    KUHN_TETS = [
        (0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
        (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7),
    ]

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner = [
                    get_index(i + (c >> 2 & 1), j + (c >> 1 & 1), k + (c & 1))
                    for c in range(8)
                ]
                elements.extend([[corner[c] for c in tet] for tet in KUHN_TETS])

    boundary = get_boundary_from_vertices_elements(elements)
    return Mesh(vertices, elements, boundary)
