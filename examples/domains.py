"""The domains the demos solve on, built rather than loaded.

A demo's shape is part of what it says -- a cantilever is a beam, and an SIMP result is
a truss spanning one -- so each demo names its domain here instead of solving on
whatever mesh the caller happened to pass. `files/` used to hold six meshes for this,
which were cached `create_rect_mesh` output: the same function reproduces any of them
vertex for vertex in about 40 ms, so they were fixtures that could only drift.

Sizes are chosen so a demo costs about what it did on the old 40x40 default, roughly
1600 vertices. That matters because the gallery workflow runs every demo on each push.
"""
from fem.mesh.mesh import Mesh
from fem.mesh.ruppert import create_rect_mesh


def square(n: int = 40) -> Mesh:
    """The unit square at `n` divisions a side."""
    return create_rect_mesh(corners=[[0.0, 0.0], [1.0, 1.0]], resolution=(n, n))


def beam(length: float = 4.0, height: float = 1.0, n: int = 80) -> Mesh:
    """A `length` x `height` rectangle divided `n` ways along its length.

    The cross-wise count follows the aspect ratio, so the triangles stay near-isotropic
    rather than becoming slivers as the beam gets longer -- element quality bounds the
    error, and a demo should not quietly hand the solver its worst case.
    """
    across = max(2, round(n * height / length))
    return create_rect_mesh(corners=[[0.0, 0.0], [length, height]], resolution=(n, across))
