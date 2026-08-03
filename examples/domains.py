"""The domains the demos solve on, built rather than loaded.

A demo's shape is part of what it says -- a cantilever is a beam, and an SIMP result is
a truss spanning one -- so each demo names its domain here instead of solving on
whatever mesh the caller happened to pass. `files/` used to hold six meshes for this,
which were cached `create_rect_mesh` output: the same function reproduces any of them
vertex for vertex in about 40 ms, so they were fixtures that could only drift.

Sizes are chosen for what the figure needs to show, within what the gallery workflow
can afford -- it renders every demo on each push. There is more room there than the
sizes suggest: a `tripcolor` is flat-shaded per element, so a coarse mesh is visibly
faceted, while the build's cost sits mostly in rasterizing animation frames rather
than in solving. `tests/test_demos.py` substitutes a tiny mesh for every domain, so
none of this reaches the per-commit gate either.
"""
import numpy as np

from fem.mesh.mesh import Mesh
from fem.mesh.ruppert import RuppertsAlgorithm, create_rect_mesh
from fem.mesh.svg import PSLG


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


def plate_with_hole_pslg(length: float = 6.0, height: float = 3.0, radius: float = 0.3,
                         segments: int = 48) -> PSLG:
    """A `length` x `height` plate with a circular hole at its centre, as a PSLG.

    Two loops: the outline and the hole. Under the even-odd rule the inner one is a
    hole rather than a second region, so the mesh covers the material and stops at
    the rim -- which is what makes the rim a boundary the solver can see.

    `segments` is how finely the circle is polygonalised. Too few and the "hole" is a
    visible polygon whose corners are stress concentrations of their own.
    """
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    hole = np.column_stack([length/2 + radius*np.cos(angles),
                            height/2 + radius*np.sin(angles)])
    return PSLG.from_loops([outline, hole])


def plate_with_hole(length: float = 6.0, height: float = 3.0, radius: float = 0.3,
                    min_angle: float = 25, max_area_fraction: float = 0.0005) -> Mesh:
    """The plate above, triangulated by Ruppert's algorithm.

    Unlike every other domain here this one is *generated* rather than laid out on a
    grid: there is no structured triangulation of a domain with a hole in it. The
    element size is set by `max_area_fraction` of the plate's area, since the angle
    bound constrains element shape but says nothing about size.
    """
    pslg = plate_with_hole_pslg(length, height, radius)
    pslg.validate()
    rupperts = RuppertsAlgorithm(pslg, min_angle=min_angle,
                                 max_area=max_area_fraction * pslg.area())
    return rupperts.refine()
