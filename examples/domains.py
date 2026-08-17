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

from fem.mesh.curves import Circle
from fem.mesh.mesh import Mesh
from fem.mesh.structured import create_rect_mesh
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
    """A slender column standing upright, meshed for a buckling solve.

    Length runs along y (ends at y = 0 and y = length) so the mode shapes draw as columns
    stand, with `height` the thin cross-dimension along x.

    Unlike `beam`, the through-thickness count is set independently of the aspect ratio
    rather than following it. A buckling mode is bending, which the elements have to
    resolve across the thin dimension; an isotropic-triangle slice of a long column would
    leave only two or three of them there, far too few for the mode to curve. `n_across`
    points evenly spaced across the thickness land on the neutral axis only when `n_across`
    is odd (an even number of intervals, one landing dead-center); `n_across` is forced odd
    so a pinned end has a vertex there to anchor.
    """
    n_across += 1 - n_across % 2
    return create_rect_mesh(corners=[[0.0, 0.0], [height, length]],
                            resolution=(n_across, n_length))


def plate_with_hole_pslg(length: float = 6.0, height: float = 3.0, radius: float = 0.3,
                         segments: int = 48) -> PSLG:
    """A `length` x `height` plate with a circular hole at its centre, as a PSLG.

    Two loops: the outline and the hole. Under the even-odd rule the inner one is a
    hole rather than a second region, so the mesh covers the material and stops at
    the rim, which makes the rim a boundary the solver can see.

    The hole loop carries a `Circle`, so refinement rounds it and an isoparametric
    solve places its boundary nodes on the true rim, instead of the polygon corners
    being stress concentrations of their own. `segments` still sets the initial
    polygonisation; with curved elements far fewer are needed.
    """
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    center = (length / 2, height / 2)
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    hole = np.column_stack([center[0] + radius*np.cos(angles),
                            center[1] + radius*np.sin(angles)])
    return PSLG.from_loops([outline, hole], curves=[None, Circle(list(center), radius)])


def disk_pslg(radius: float = 1.0, center: tuple[float, float] = (0.0, 0.0),
              segments: int = 24) -> PSLG:
    """A disk of `radius` about `center`, its rim carrying a `Circle`.

    A single curved loop: with isoparametric elements the rim follows the true circle,
    so `segments` only sets the coarsest sampling refinement starts from.
    """
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    rim = np.column_stack([center[0] + radius*np.cos(angles),
                           center[1] + radius*np.sin(angles)])
    return PSLG.from_loops([rim], curves=[Circle(list(center), radius)])


def annulus_pslg(inner_radius: float = 0.5, outer_radius: float = 1.0,
                 center: tuple[float, float] = (0.0, 0.0), segments: int = 24) -> PSLG:
    """An annulus between two concentric circles, each rim carrying a `Circle`.

    The outer loop is the material's outer edge and the inner loop the hole; under the
    even-odd rule the mesh covers the ring between them, and both rims are curved.
    """
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)

    def ring(r: float) -> np.ndarray:
        return np.column_stack([center[0] + r*np.cos(angles), center[1] + r*np.sin(angles)])

    return PSLG.from_loops(
        [ring(outer_radius), ring(inner_radius)],
        curves=[Circle(list(center), outer_radius), Circle(list(center), inner_radius)],
    )


def heatsink_pslg(width: float = 3.0, base_height: float = 0.5, fin_height: float = 1.4,
                  fin_width: float = 0.22, n_fins: int = 7, margin: float = 0.18) -> PSLG:
    """A finned heatsink cross-section (a comb) as a single-outline PSLG.

    A `width` x `base_height` base slab carries `n_fins` fins of `fin_width` x
    `fin_height` standing on top, evenly spaced and kept `margin` clear of the ends. The
    bottom edge is the heated face (a chip beneath it); every other edge is a surface
    that sheds heat, so a solver reads the whole top and sides as a convective film.
    """
    span = width - 2 * margin
    pitch = (span - fin_width) / (n_fins - 1) if n_fins > 1 else 0.0
    lefts = margin + pitch * np.arange(n_fins)

    # Traced counter-clockwise: the bottom edge, up the right side, then the top from
    # right to left, going up and over each fin, and finally down the left side.
    outline = [(0.0, 0.0), (width, 0.0), (width, base_height)]
    for x_l in lefts[::-1]:
        x_r = x_l + fin_width
        outline += [(x_r, base_height), (x_r, base_height + fin_height),
                    (x_l, base_height + fin_height), (x_l, base_height)]
    outline.append((0.0, base_height))
    return PSLG.from_loops([np.array(outline)])


def _naca4_outline(camber: float, camber_pos: float, thickness: float, n: int,
                   te_trim: float = 0.05) -> np.ndarray:
    """A NACA 4-digit airfoil as a closed loop of points, unit chord along +x.

    `camber` (m), `camber_pos` (p), and `thickness` (t) are the usual fractions: a NACA
    2412 is (0.02, 0.4, 0.12). Cosine node spacing clusters points at the leading and
    trailing edges, where the curvature is highest.

    `te_trim` cuts that fraction of the chord off the trailing edge, leaving a blunt edge
    of finite thickness in place of the near-cusp a full 4-digit section tapers to. The
    cusp is a sliver the mesher would chase with unboundedly many tiny triangles; a blunt
    edge meshes in a moment and barely changes the flow.
    """
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta)) * (1 - te_trim)    # cosine spacing, 0 (LE) to 1-te_trim (TE)
    yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                          + 0.2843 * x**3 - 0.1015 * x**4)
    if camber > 0 and 0 < camber_pos < 1:
        m, p = camber, camber_pos
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)                 # symmetric section (m = 0)
    theta = np.arctan(dyc)
    upper = np.column_stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)])
    lower = np.column_stack([x + yt * np.sin(theta), yc - yt * np.cos(theta)])
    # Trailing edge over the top to the leading edge, then back under; the shared leading
    # edge is dropped so it is not duplicated.
    return np.vstack([upper[::-1], lower[1:]])


def airfoil_channel_pslg(length: float = 7.0, height: float = 4.0, chord: float = 3.0,
                         angle_of_attack: float = 6.0, camber: float = 0.02,
                         camber_pos: float = 0.4, thickness: float = 0.12,
                         n_points: int = 100) -> PSLG:
    """A rectangular channel with a NACA 4-digit airfoil obstacle in it, as a PSLG.

    The airfoil is generated analytically (no data file needed), scaled to `chord`,
    pitched `angle_of_attack` degrees nose-up into a left-to-right flow, and placed in
    the channel. The default is a NACA 2412. Under the even-odd rule the airfoil loop is
    a hole, so a mesh covers the fluid and stops at the wing, making its surface a
    boundary the solver sees (and, taking no condition, a streamline).
    """
    foil = _naca4_outline(camber, camber_pos, thickness, n_points)
    foil = foil * chord - [0.35 * chord, 0.0]       # pivot near the quarter-chord
    a = np.deg2rad(angle_of_attack)
    c, s = np.cos(a), np.sin(a)
    foil = foil @ np.array([[c, -s], [s, c]])       # nose up into the +x flow
    foil = foil + [0.42 * length, 0.5 * height]
    outline = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    return PSLG.from_loops([outline, foil])


def l_bracket_pslg(arm: float = 4.0, width: float = 1.2, fillet_radius: float = 0.0,
                   n_fillet: int = 16) -> PSLG:
    """An L-shaped bracket as a PSLG, with an optional fillet at the inner corner.

    Two limbs of thickness `width` and length `arm`: the vertical one up the left edge,
    the horizontal one along the bottom, meeting at a re-entrant (inner) corner at
    `(width, width)`. A sharp corner there is a stress singularity; `fillet_radius > 0`
    rounds it with a concave arc of `n_fillet` points, the same way `tuning_fork_pslg`
    rounds its slot root, which is what turns the peak into a finite, mesh-converged
    value.

    Clamp the top of the vertical limb (`on_plane(1, arm)`) and load the tip of the
    horizontal one (`on_plane(0, arm)`); the concentration then sits at the inner corner.
    """
    outline = [(0.0, 0.0), (arm, 0.0), (arm, width)]
    if fillet_radius > 0:
        # Round the re-entrant corner: an arc of radius r centred at (width+r, width+r),
        # bulging into the notch to add material. It runs from A = (width+r, width) on the
        # bottom limb's top edge (theta = 3pi/2) to B = (width, width+r) on the vertical
        # limb's right edge (theta = pi), replacing the sharp point between them.
        r = fillet_radius
        theta = np.linspace(1.5 * np.pi, np.pi, n_fillet)
        arc = np.column_stack([width + r + r * np.cos(theta), width + r + r * np.sin(theta)])
        outline.extend(map(tuple, arc))
    else:
        outline.append((width, width))
    outline.extend([(width, arm), (0.0, arm)])
    return PSLG.from_loops([np.array(outline)])


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

    Dimensions are in metres; the defaults size a steel fork near concert A (see
    `demo_modal`). Centred on x = 0, with the stem base on y = 0, the line a modal solve
    clamps.
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
