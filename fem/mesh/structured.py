"""Structured simplex meshes of simple shapes: an axis-aligned box in 1D, 2D, or 3D.

These are the meshes convergence studies want: uniform refinement without the
element-quality variation an unstructured generator introduces. Distinct in intent
from `ruppert`, whose Delaunay refinement meshes arbitrary outlines.
"""
from collections.abc import Sequence
from typing import Literal

import numpy as np

from fem.mesh.mesh import Mesh
from fem.typing import FloatArray, IntArray

# Cube corners are indexed by the bits of (di, dj, dk): corner c has di = c >> 2, dj =
# c >> 1, dk = c & 1, so corner 5 is (1, 0, 1). Two corners are cube-edge-adjacent when
# their indices differ in one bit.

# Kuhn's decomposition: six tets sharing the main diagonal (corner 0 to corner 7), one
# per permutation of the three axes. Every cell splits the same way, so neighbours agree
# on shared faces with no orientation bookkeeping. All six tets are congruent, and the
# decomposition generalises to any dimension and refines through a bounded number of
# similarity classes (Bey's bisection), which is what makes it the right base for 3D
# adaptive refinement or multigrid. Its drawback is shape: the shared tet has edges of
# length 1, sqrt(2), and sqrt(3), far from regular, which inflates the error constant on
# a fixed mesh. Kept selectable (`tet_split='kuhn'`) for a future 3D adaptive path.
_KUHN_TETS = [
    (0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
    (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7),
]

# The five-tetrahedron decomposition: a central tet on the four corners of one parity
# (all its edges are face diagonals, so it is a regular tetrahedron) plus four corner
# tets. Every cube has one of two mirror forms, chosen so the diagonal each cuts on a
# shared face matches its neighbour's; the (i + j + k) checkerboard below alternates
# them. This is the default: on a fixed mesh its near-regular tets give a much smaller
# error constant (about 4x lower L2 error, and a clean second-order rate from the
# coarsest mesh, where Kuhn's is still pre-asymptotic). It has no clean uniform-
# refinement rule, so it is a static-mesh choice, not an adaptive one.
_TET5_EVEN = [
    (0, 3, 5, 6),   # the central regular tet, corners of even parity
    (0, 1, 3, 5), (0, 2, 3, 6), (0, 4, 5, 6), (3, 5, 6, 7),   # caps at 1, 2, 4, 7
]
_TET5_ODD = [
    (1, 2, 4, 7),   # the mirror form's central tet, corners of odd parity
    (0, 1, 2, 4), (1, 2, 3, 7), (1, 4, 5, 7), (2, 4, 6, 7),   # caps at 0, 3, 5, 6
]

TetSplit = Literal['regular', 'kuhn']


def box_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
    *,
    tet_split: TetSplit = 'regular',
) -> Mesh:
    '''A structured simplex mesh of the axis-aligned box spanned by `corners`, with
    `resolution` nodes along each axis.

    The dimension is the corners': `((x0,), (x1,))` is a line of `LinearLineElement`s,
    `((x0, y0), (x1, y1))` a rectangle of triangles, `((x0, y0, z0), (x1, y1, z1))` a box
    of tets. Grid cells split into two triangles (alternating diagonals) or, in 3D, into
    tets, so the mesh is conforming.

    `tet_split` (3D only) chooses the tetrahedral decomposition. `'regular'` (the
    default) is the five-tet split, whose near-regular elements give a smaller error
    constant and a clean second-order rate from a coarse mesh. `'kuhn'` is the six-tet
    Kuhn split, which is worse-shaped but refines cleanly, for an adaptive or multigrid
    3D path; see the module comments.
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
        return _box(lower, upper, resolution, tet_split)
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


def _box(lower, upper, resolution, tet_split: TetSplit) -> Mesh:
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
    # with its tets in sequence.
    i, j, k = (axis.ravel() for axis in np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing='ij'))
    c = np.arange(8)
    # The cell's eight corners by the bits of c: (di, dj, dk) = (c >> 2 & 1, c >> 1 & 1, c & 1).
    corners = ((i[:, None] + (c >> 2 & 1)) * ny + (j[:, None] + (c >> 1 & 1))) * nz + (k[:, None] + (c & 1))
    elements = _tets_per_cell(corners, i + j + k, tet_split)
    return Mesh(vertices, elements)


def _tets_per_cell(corners: IntArray, cell_sum: IntArray, tet_split: TetSplit) -> IntArray:
    '''Turn each cell's eight corner indices into its tets, `(n_cells * n_tets, 4)`.

    Kuhn splits every cell the same way; the regular five-tet split alternates its two
    mirror forms on the `(i + j + k)` checkerboard so neighbours agree on shared faces.
    '''
    if tet_split == 'kuhn':
        return corners[:, np.array(_KUHN_TETS)].reshape(-1, 4)
    if tet_split != 'regular':
        raise ValueError(f"tet_split must be 'regular' or 'kuhn', got {tet_split!r}")
    even = corners[:, np.array(_TET5_EVEN)]        # (n_cells, 5, 4)
    odd = corners[:, np.array(_TET5_ODD)]
    picked = np.where((cell_sum % 2 == 0)[:, None, None], even, odd)
    return picked.reshape(-1, 4)
