"""The domains the demos solve on, built rather than loaded.

A demo's shape is part of what it says (a cantilever is a beam, and an SIMP result is
a truss spanning one), so each demo names its domain here instead of solving on
whatever mesh the caller happened to pass. `files/` used to hold six meshes for this,
which were cached `create_rect_mesh` output: the same function reproduces any of them
vertex for vertex in about 40 ms, so they were fixtures that could only drift.

Sizes are chosen for what the figure needs to show, within what the gallery workflow
can afford: it renders every demo on each push. There is more room there than the
sizes suggest: a `tripcolor` is flat-shaded per element, so a coarse mesh is visibly
faceted, while the build's cost sits mostly in rasterizing animation frames rather
than in solving. `tests/test_demos.py` substitutes a tiny mesh for every domain, so
none of this reaches the per-commit gate either.
"""
import numpy as np

from fem.mesh.mesh import Mesh
from fem.mesh.ruppert import create_rect_mesh
from fem.mesh.svg import PSLG


def square(n: int = 40) -> Mesh:
    """The unit square at `n` divisions a side."""
    return create_rect_mesh(corners=[[0.0, 0.0], [1.0, 1.0]], resolution=(n, n))


def beam(length: float = 4.0, height: float = 1.0, n: int = 80) -> Mesh:
    """A `length` x `height` rectangle divided `n` ways along its length.

    The cross-wise count follows the aspect ratio, so the triangles stay near-isotropic
    rather than becoming slivers as the beam gets longer: element quality bounds the
    error, and a demo should not quietly hand the solver its worst case.
    """
    across = max(2, round(n * height / length))
    return create_rect_mesh(corners=[[0.0, 0.0], [length, height]], resolution=(n, across))


def column(length: float = 24.0, height: float = 1.0,
           n_length: int = 48, n_across: int = 6) -> Mesh:
    """A slender column, meshed for a buckling solve.

    Unlike `beam`, the through-thickness count is set independently of the aspect ratio
    rather than following it. A buckling mode is bending, which the elements have to
    resolve across the height; an isotropic-triangle slice of a long column would leave
    only two or three of them there, far too few for the mode to curve. `n_across` points
    evenly spaced across the height land on the neutral axis only when `n_across` is odd
    (an even number of intervals, one landing dead-center); `n_across` is forced odd so a
    pinned end has a vertex there to anchor.
    """
    n_across += 1 - n_across % 2
    return create_rect_mesh(corners=[[0.0, 0.0], [length, height]],
                            resolution=(n_length, n_across))


def plate_with_hole_pslg(length: float = 6.0, height: float = 3.0, radius: float = 0.3,
                         segments: int = 48) -> PSLG:
    """A `length` x `height` plate with a circular hole at its centre, as a PSLG.

    Two loops: the outline and the hole. Under the even-odd rule the inner one is a
    hole rather than a second region, so the mesh covers the material and stops at
    the rim, which makes the rim a boundary the solver can see.

    `segments` is how finely the circle is polygonalised. Too few and the "hole" is a
    visible polygon whose corners are stress concentrations of their own.
    """
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    hole = np.column_stack([length/2 + radius*np.cos(angles),
                            height/2 + radius*np.sin(angles)])
    return PSLG.from_loops([outline, hole])


def tuning_fork_pslg(tine_length: float = 0.088, tine_thickness: float = 0.004,
                     gap: float = 0.006, base_height: float = 0.012,
                     stem_length: float = 0.030, stem_width: float = 0.008,
                     n_fillet: int = 12) -> PSLG:
    """A two-tined tuning fork, upright with its tines pointing up, as a PSLG.

    One non-convex outline, not a shape with holes: a stem rises into a base that
    forks into two tines with a slot between them. Traced counter-clockwise from the
    bottom-left of the stem, up the outer edges, across each tip, and down the inner
    edges, with a rounded valley (radius `gap/2`) at the slot root in place of the two
    sharp reentrant corners a straight slot bottom would leave. The mesher resolves the
    straight edges; only the fillet is pre-sampled, into `n_fillet` points.

    Dimensions are in metres, sized so a steel fork's fundamental lands near concert A
    (see `demo_modal`): each tine is a clamped-free beam of length `tine_length` and
    thickness `tine_thickness`, and beam theory sets the tone from those two and the
    material. Centred on x = 0, with the stem base on y = 0 -- the line a modal solve
    clamps, the node a real fork is held at.
    """
    half_outer = gap / 2 + tine_thickness       # tine outer edge, |x| at the tips
    y_base_top = stem_length + base_height       # where the tines and the slot begin
    y_tip = y_base_top + tine_length

    # The slot root as a rounded valley joining the two reentrant corners at
    # (+-gap/2, y_base_top): an ellipse, x-radius gap/2 so its ends land exactly on the
    # corners, y-depth capped to stay inside the base. theta pi -> 2pi runs left corner
    # -> bottom -> right corner, so the valley's endpoints replace the corners rather
    # than duplicating them (which PSLG.validate would reject).
    depth = min(gap / 2, 0.8 * base_height)
    theta = np.linspace(np.pi, 2 * np.pi, n_fillet)
    valley = np.column_stack([(gap / 2) * np.cos(theta), y_base_top + depth * np.sin(theta)])

    outline = np.array([
        [-stem_width / 2, 0.0],                  # stem base, left
        [-stem_width / 2, stem_length],          # up the stem
        [-half_outer, stem_length],              # out to the base's left edge
        [-half_outer, y_tip],                    # up the left tine's outer edge
        [-gap / 2, y_tip],                       # across the left tip, then down the
        *valley.tolist(),                        # inner edge into the valley and up again
        [gap / 2, y_tip],                        # to the right tip
        [half_outer, y_tip],                     # across the right tip
        [half_outer, stem_length],               # down the right tine's outer edge
        [stem_width / 2, stem_length],           # in to the stem
        [stem_width / 2, 0.0],                   # down the stem to the base
    ])
    return PSLG.from_loops([outline])
