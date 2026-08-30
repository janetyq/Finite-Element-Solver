"""Structured simplex meshes of simple shapes: an axis-aligned box in 1D, 2D, or 3D.

These are the meshes convergence studies want: uniform refinement without the
element-quality variation an unstructured generator introduces. Distinct in intent
from `ruppert`, whose Delaunay refinement meshes arbitrary outlines.
"""
from collections.abc import Sequence

import numpy as np

from fem.mesh.mesh import Mesh
from fem.typing import FloatArray

# The six tets of Kuhn's decomposition of a cube, over corners indexed by the bits of
# (di, dj, dk): corner 5 is (1, 0, 1). Every tet contains the main diagonal (corner
# 000 to 111), one per permutation of the three axes. Because every cell splits along
# the same diagonal direction, neighbouring cells agree on their shared faces and the
# mesh is conforming, which is what makes it usable for convergence studies.
_KUHN_TETS = [
    (0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
    (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7),
]


def box_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured simplex mesh of the axis-aligned box spanned by `corners`, with
    `resolution` nodes along each axis.

    The dimension is the corners': `((x0,), (x1,))` is a line of `LinearLineElement`s,
    `((x0, y0), (x1, y1))` a rectangle of triangles, `((x0, y0, z0), (x1, y1, z1))` a box
    of tets. Grid cells split into two triangles (alternating diagonals) or six tets
    (Kuhn's decomposition), so the mesh is conforming.
    '''
    lower, upper = np.asarray(corners, dtype=float)
    dim = len(lower)
    if dim != len(resolution):
        raise ValueError(f'{dim}D corners need {dim} resolutions, got {len(resolution)}')
    if dim == 1:
        return _line(lower, upper, resolution)
    if dim == 2:
        return _rect(lower, upper, resolution)
    if dim == 3:
        return _box(lower, upper, resolution)
    raise ValueError(f'box_mesh meshes 1D, 2D, or 3D boxes, got {dim}D corners')


def _line(lower, upper, resolution) -> Mesh:
    n = resolution[0]
    vertices = np.linspace(lower[0], upper[0], n)[:, None]
    elements = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    return Mesh(vertices, elements)


def _rect(lower, upper, resolution) -> Mesh:
    nx, ny = resolution
    x_range = np.linspace(lower[0], upper[0], nx)
    y_range = np.linspace(lower[1], upper[1], ny)
    # x varies fastest: vertex (i, j) is at j * nx + i.
    vertices = np.column_stack([np.tile(x_range, ny), np.repeat(y_range, nx)])

    # Every cell at once, i-major so the element order is the cell order.
    i, j = (axis.ravel() for axis in np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij'))
    n00, n10 = j * nx + i, j * nx + i + 1
    n01, n11 = (j + 1) * nx + i, (j + 1) * nx + i + 1
    even = ((i + j) % 2 == 0)[:, None]
    # Alternating diagonals: an even cell splits along 00-11, an odd one along 10-01.
    first = np.where(even, np.column_stack([n00, n10, n11]), np.column_stack([n00, n10, n01]))
    second = np.where(even, np.column_stack([n00, n11, n01]), np.column_stack([n10, n11, n01]))
    elements = np.stack([first, second], axis=1).reshape(-1, 3)
    return Mesh(vertices, elements)


def _box(lower, upper, resolution) -> Mesh:
    nx, ny, nz = resolution
    x_range = np.linspace(lower[0], upper[0], nx)
    y_range = np.linspace(lower[1], upper[1], ny)
    z_range = np.linspace(lower[2], upper[2], nz)
    # z varies fastest: vertex (i, j, k) is at (i * ny + j) * nz + k.
    vertices = np.column_stack([
        np.repeat(x_range, ny * nz),
        np.tile(np.repeat(y_range, nz), nx),
        np.tile(z_range, nx * ny),
    ])

    # Every cell at once, i-major, k fastest, so the element order is the cell order
    # with its six Kuhn tets in sequence.
    i, j, k = (axis.ravel() for axis in np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing='ij'))
    c = np.arange(8)
    # The cell's eight corners by the bits of c: (di, dj, dk) = (c >> 2 & 1, c >> 1 & 1, c & 1).
    corners = ((i[:, None] + (c >> 2 & 1)) * ny + (j[:, None] + (c >> 1 & 1))) * nz + (k[:, None] + (c & 1))
    elements = corners[:, np.array(_KUHN_TETS)].reshape(-1, 4)
    return Mesh(vertices, elements)
