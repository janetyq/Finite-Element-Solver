"""Structured simplex meshes of simple shapes: an axis-aligned box in 1D, 2D, or 3D, and
an annulus.

These are the meshes convergence studies want: uniform refinement without the
element-quality variation an unstructured generator introduces. Distinct in intent
from `ruppert`, whose Delaunay refinement meshes arbitrary outlines.
"""
from collections.abc import Sequence

import numpy as np

from fem.mesh.curves import Circle
from fem.mesh.mesh import Mesh, boundary_facets
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
    vertices = np.array([[x, y] for y in y_range for x in x_range])

    def node(i, j):
        return j * nx + i

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if (i + j) % 2 == 0:
                elements.append([node(i, j), node(i+1, j), node(i+1, j+1)])
                elements.append([node(i, j), node(i+1, j+1), node(i, j+1)])
            else:
                elements.append([node(i, j), node(i+1, j), node(i, j+1)])
                elements.append([node(i+1, j), node(i+1, j+1), node(i, j+1)])
    return Mesh(vertices, elements)


def _box(lower, upper, resolution) -> Mesh:
    nx, ny, nz = resolution
    x_range = np.linspace(lower[0], upper[0], nx)
    y_range = np.linspace(lower[1], upper[1], ny)
    z_range = np.linspace(lower[2], upper[2], nz)
    vertices = np.array([[x, y, z] for x in x_range for y in y_range for z in z_range])

    def node(i, j, k):
        return (i * ny + j) * nz + k

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner = [
                    node(i + (c >> 2 & 1), j + (c >> 1 & 1), k + (c & 1))
                    for c in range(8)
                ]
                elements.extend([[corner[c] for c in tet] for tet in _KUHN_TETS])
    return Mesh(vertices, elements)


def annulus_mesh(
    inner_radius: float, outer_radius: float, n_radial: int, n_theta: int,
) -> Mesh:
    """Structured triangle mesh of the annulus about the origin, with its rims attached
    as `Circle`s.

    `n_radial` nodes across the radial direction and `n_theta` sectors around. The
    inner and outer boundary facets carry a `Circle`, so a curved space places their
    midside nodes on the true rim rather than at the chord midpoint.
    """
    rings = np.arange(n_radial)
    radii = inner_radius + (outer_radius - inner_radius) * (rings / (n_radial - 1))
    thetas = 2 * np.pi * np.arange(n_theta) / n_theta
    r, t = np.meshgrid(radii, thetas, indexing="ij")
    vertices = np.column_stack([(r * np.cos(t)).ravel(), (r * np.sin(t)).ravel()])

    def node(ring: int, sector: int) -> int:
        return ring * n_theta + sector % n_theta

    elements = []
    for ring in range(n_radial - 1):
        for sector in range(n_theta):
            a, b = node(ring, sector), node(ring, sector + 1)
            c, d = node(ring + 1, sector + 1), node(ring + 1, sector)
            elements.extend([[a, b, c], [a, c, d]])
    elements = np.array(elements)

    boundary = boundary_facets(elements)
    inner_curve = Circle([0.0, 0.0], inner_radius)
    outer_curve = Circle([0.0, 0.0], outer_radius)
    midradius = 0.5 * (inner_radius + outer_radius)
    boundary_curves = [
        inner_curve if float(np.hypot(*vertices[facet[0]])) < midradius else outer_curve
        for facet in boundary
    ]
    return Mesh(vertices, elements, boundary, boundary_curves)
