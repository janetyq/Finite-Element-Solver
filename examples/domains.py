"""The domains the demos solve on, built rather than loaded.

A demo's shape is part of what it says (a cantilever is a beam, a SIMP result is a
truss spanning one), so each demo names its domain here. Sizes are chosen for what
the figure needs to show within what the gallery build can afford; `tests/test_demos.py`
substitutes a tiny mesh for every domain.
"""
import numpy as np

from fem.mesh.curves import Arc, Circle, Line
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.mesh.outline import Outline


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


def plate_with_hole_outline(length: float = 6.0, height: float = 3.0,
                            radius: float = 0.3) -> Outline:
    """A `length` x `height` plate with a circular hole at its centre.

    Two loops: the outline and the hole, which under the even-odd rule is a hole rather
    than a second region. The hole is a `Circle`, so refinement rounds it and an
    isoparametric solve places its boundary nodes on the true rim.
    """
    plate = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    return Outline([Outline.from_polygons([plate]).loops[0],
                    Circle([length / 2, height / 2], radius)])


def annulus_outline(inner_radius: float = 0.5, outer_radius: float = 1.0,
                    center: tuple[float, float] = (0.0, 0.0)) -> Outline:
    """An annulus between two concentric circles.

    The outer loop is the material's outer edge and the inner loop the hole; under the
    even-odd rule the mesh covers the ring between them, and both rims are curved.
    """
    return Outline([Circle(list(center), outer_radius), Circle(list(center), inner_radius)])


def heatsink_outline(width: float = 3.0, base_height: float = 0.5, fin_height: float = 1.4,
                     fin_width: float = 0.22, n_fins: int = 7, margin: float = 0.18) -> Outline:
    """A finned heatsink cross-section (a comb) as a single loop.

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
    return Outline.from_polygons([np.array(outline)])


def _naca4_outline(camber: float, camber_pos: float, thickness: float, n: int,
                   te_trim: float = 0.05) -> np.ndarray:
    """A NACA 4-digit airfoil as a closed loop of points, unit chord along +x.

    `camber` (m), `camber_pos` (p), and `thickness` (t) are the usual fractions: a NACA
    2412 is (0.02, 0.4, 0.12). Cosine node spacing clusters points at the leading and
    trailing edges, where the curvature is highest.

    `te_trim` cuts that fraction of the chord off the trailing edge, leaving a blunt
    edge in place of the near-cusp a full 4-digit section tapers to, which the mesher
    would chase with unboundedly many tiny triangles.
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


def airfoil_channel_outline(length: float = 7.0, height: float = 4.0, chord: float = 3.0,
                         angle_of_attack: float = 6.0, camber: float = 0.02,
                         camber_pos: float = 0.4, thickness: float = 0.12,
                            n_points: int = 100) -> Outline:
    """A rectangular channel with a NACA 4-digit airfoil obstacle in it.

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
    channel = np.array([[0.0, 0.0], [length, 0.0], [length, height], [0.0, height]])
    return Outline.from_polygons([channel, foil])


def l_bracket_outline(arm: float = 4.0, width: float = 1.2,
                      fillet_radius: float = 0.0) -> Outline:
    """An L-shaped bracket, with an optional fillet at the inner corner.

    Two limbs of thickness `width` and length `arm`: the vertical one up the left edge,
    the horizontal one along the bottom, meeting at a re-entrant (inner) corner at
    `(width, width)`. A sharp corner there is a stress singularity; `fillet_radius > 0`
    rounds it with a concave `Arc`, so an isoparametric solve reads a true circular
    fillet.

    Clamp the top of the vertical limb (`on_plane(1, arm)`) and load the tip of the
    horizontal one (`on_plane(0, arm)`); the concentration then sits at the inner corner.
    """
    corners = [np.array(p) for p in [(0.0, 0.0), (arm, 0.0), (arm, width)]]
    pieces = [Line(corners[0], corners[1]), Line(corners[1], corners[2])]
    if fillet_radius > 0:
        # Round the re-entrant corner: an arc of radius r centred at (width+r, width+r),
        # bulging into the notch to add material. It runs from A = (width+r, width) on the
        # bottom limb's top edge (theta = 3pi/2) to B = (width, width+r) on the vertical
        # limb's right edge (theta = pi), replacing the sharp point between them: the
        # arc reversed, since the outline is traced clockwise through it.
        r = fillet_radius
        fillet = Arc([width + r, width + r], r, np.pi, 1.5 * np.pi).reversed()
        pieces += [Line(corners[2], fillet.start), fillet]
        inner_end = fillet.end
    else:
        pieces.append(Line(corners[2], [width, width]))
        inner_end = np.array([width, width])
    top = [np.array(p) for p in [(width, arm), (0.0, arm)]]
    pieces += [Line(inner_end, top[0]), Line(top[0], top[1]), Line(top[1], corners[0])]
    return Outline([pieces])


def tuning_fork_outline(tine_length: float = 0.088, tine_thickness: float = 0.004,
                     gap: float = 0.006, base_height: float = 0.012,
                     stem_length: float = 0.030, stem_width: float = 0.008,
                        n_fillet: int = 12) -> Outline:
    """A two-tined tuning fork, upright with its tines pointing up.

    One non-convex outline: a stem rises into a base that forks into two tines with a
    slot between them. Traced counter-clockwise from the bottom-left of the stem, with
    a rounded valley (radius `gap/2`, `n_fillet` points) at the slot root in place of
    two sharp reentrant corners.

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
    # than duplicating them (which validation would reject).
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
    return Outline.from_polygons([outline])


# The generated shapes the outline demo meshes alongside the traced `files/*.svg`
# outlines: a star for sharp reentrant corners, a gear for teeth around a circular bore.


def star_outline(points: int = 5, outer_radius: float = 1.0, inner_radius: float = 0.42,
                 center: tuple[float, float] = (0.0, 0.0)) -> Outline:
    """A `points`-pointed star as a single straight-line loop.

    Radii alternate between `outer_radius` at the tips and `inner_radius` at the notches,
    so the reentrant notches are the sharp corners Ruppert's meets at the input angle
    rather than refining away.
    """
    angles = np.pi / 2 + np.linspace(0, 2 * np.pi, 2 * points, endpoint=False)
    radii = np.where(np.arange(2 * points) % 2 == 0, outer_radius, inner_radius)
    outline = np.column_stack([center[0] + radii * np.cos(angles),
                               center[1] + radii * np.sin(angles)])
    return Outline.from_polygons([outline])


def gear_outline(teeth: int = 12, root_radius: float = 0.7, tooth_height: float = 0.22,
                 tooth_fraction: float = 0.5, bore_radius: float = 0.28,
                 center: tuple[float, float] = (0.0, 0.0)) -> Outline:
    """A spur gear with a circular bore, as two loops (rim and hole).

    Each of `teeth` sectors carries one tooth: the radius steps from `root_radius` out to
    `root_radius + tooth_height` over the middle `tooth_fraction` of the sector and back,
    with radial flanks. The bore is a `Circle`, so an isoparametric solve reads
    a true round hole and refinement rounds it; under the even-odd rule it is a hole in
    the gear rather than a second part.
    """
    tip_radius = root_radius + tooth_height
    pitch = 2 * np.pi / teeth
    gap = 0.5 * (1 - tooth_fraction) * pitch     # root arc either side of each tooth
    outline = []
    for i in range(teeth):
        base = i * pitch
        # (root, base) -> (root, base+gap): the valley; then radial flank up, the tip
        # land, and the next flank down is the following sector's opening edge.
        for radius, angle in ((root_radius, base), (root_radius, base + gap),
                              (tip_radius, base + gap), (tip_radius, base + pitch - gap)):
            outline.append((center[0] + radius * np.cos(angle),
                            center[1] + radius * np.sin(angle)))

    return Outline([Outline.from_polygons([np.array(outline)]).loops[0],
                    Circle(list(center), bore_radius)])


def harbor_outline(length: float = 6.0, width: float = 4.0, wall_x: float = 2.5,
                   wall_thickness: float = 0.15, gap: float = 1.2) -> Outline:
    """A rectangular basin crossed by a breakwater with one gap, as a single loop.

    Open water lies left of the wall at `wall_x`, the sheltered harbor to its right. The
    two wall arms grow inward from the top and bottom edges, leaving `gap` open at
    mid-width.
    """
    x0, x1 = wall_x, wall_x + wall_thickness
    y0, y1 = (width - gap) / 2, (width + gap) / 2
    outline = np.array([
        [0.0, 0.0], [x0, 0.0], [x0, y0], [x1, y0], [x1, 0.0], [length, 0.0],
        [length, width], [x1, width], [x1, y1], [x0, y1], [x0, width], [0.0, width],
    ])
    return Outline.from_polygons([outline])
