"""The domains the demos solve on, built rather than loaded.

The structured meshes several demos share. A demo whose shape is part of its story
(a heatsink, an L-bracket, a plate with a hole) draws its own `Outline` in its
`physics.py`. Sizes are chosen for what the figure needs to show within what the gallery
build can afford; `tests/test_demos.py` substitutes a tiny mesh for every domain.
"""
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh


def square(n: int = 40) -> Mesh:
    """The unit square at `n` divisions a side."""
    return box_mesh(corners=[[0.0, 0.0], [1.0, 1.0]], resolution=(n, n))


def beam(length: float = 4.0, height: float = 1.0, n: int = 80) -> Mesh:
    """A `length` x `height` rectangle divided `n` ways along its length.

    The cross-wise count follows the aspect ratio, so the triangles stay near-isotropic
    as the beam gets longer.
    """
    across = max(2, round(n * height / length))
    return box_mesh(corners=[[0.0, 0.0], [length, height]], resolution=(n, across))


def column(length: float = 24.0, height: float = 1.0,
           n_length: int = 48, n_across: int = 6) -> Mesh:
    """A slender column standing upright, meshed for a buckling solve.

    Length runs along y (ends at y = 0 and y = length) so the mode shapes draw as columns
    stand, with `height` the thin cross-dimension along x.

    Unlike `beam`, the through-thickness count is set independently of the aspect
    ratio: a buckling mode is bending, which needs several elements across the thin
    dimension. `n_across` is forced odd so a vertex lands on the neutral axis for a
    pinned end to anchor.
    """
    n_across += 1 - n_across % 2
    return box_mesh(corners=[[0.0, 0.0], [height, length]],
                            resolution=(n_across, n_length))

