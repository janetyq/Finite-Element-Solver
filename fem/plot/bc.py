"""Drawing a problem's boundary conditions: what each one says, and where it applies.

A condition is drawn with two independent axes of meaning:

* Colour is the weak-form type: Dirichlet blue (held at a value), Neumann red (an
  applied flux), Robin orange (a blend), and the unwritten natural condition grey.
* Shape is the mechanical role, as specific as the components allow. A vector
  Dirichlet edge is a drafting symbol: a hatched wall for a clamp, triangles on a
  ground line for a roller, a lone triangle for a pin, and an arrow off a wall or pin
  for an imposed displacement. A scalar field keeps dots and runs. Neumann is arrows
  (a vector traction) or a run (a scalar flux); Robin is a run.

`plot_bc` renders the standalone conditions panel: a filled body, the marks, and a
legend of values. `overlay_supports` renders the same marks over a result already on
the axes. Both go through one classifier (`_classify`) and one set of glyph drawers.
"""
import warnings
from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from fem.plot.helpers import plot_boundary


# One colour per condition type, held across the shapes a condition can take, so a
# Neumann flux reads as the same thing whether it came out as arrows or as a run of
# boundary. Grey is for the condition nobody wrote, and is meant to recede.
BC_COLORS = {'dirichlet': 'tab:blue', 'neumann': 'red', 'robin': 'tab:orange'}
FREE_COLOR = '0.6'

# The domain drawn as a body rather than left as white space. Without it the panel is an
# empty box with marks on its edges, which at gallery-card size reads as nothing at all.
DOMAIN_FILL = '#e9e9e9'

# Arrows per condition. One per boundary vertex makes the density a fact about the mesh
# rather than about the load, and on a fine edge they overlap into a solid band.
MAX_BC_ARROWS = 12

# Arrow length, as a fraction of the domain's larger extent. Fixed, because the
# magnitude is quoted in the legend and all an arrow has to say here is which way.
BC_ARROW_LENGTH = 0.10

# The drafting glyphs (wall, triangles) sized as a fraction of the domain's long extent.
DRAFTING_SIZE = 0.055
GLYPH_ZORDER = 6


# -- formatting a condition's value for the legend ------------------------------------


def _format_component(c):
    """One component: a number, or 'free' for the NaN a roller leaves unpinned."""
    return 'free' if np.isnan(c) else f'{c:g}'


def _format_vector(value):
    """A condition's value: bare if scalar, in parentheses if it has components."""
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if len(value) == 1:
        return _format_component(value[0])
    return '(' + ', '.join(_format_component(c) for c in value) + ')'


def _format_values(values):
    """What a condition is set to over its region: one value, or the range it spans."""
    if len(values) == 0:
        return '?'
    # equal_nan: a roller's free component is NaN at every vertex, and that
    # shared NaN is itself the single value being reported, not a range.
    if np.allclose(values, values[0], equal_nan=True):
        return _format_vector(values[0])
    # nanmin/nanmax warn on a component that is NaN at every vertex (a free
    # component spanning a callable value that varies in its pinned
    # component), which is expected here, and `_format_component` already turns
    # the resulting NaN back into 'free'.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (f'{_format_vector(np.nanmin(values, axis=0))} to '
                f'{_format_vector(np.nanmax(values, axis=0))}')


def _robin_label(kappa, values):
    """A Robin condition as the equation it is, rather than as the word "Robin".

    Written around the ambient value g/kappa where there is one, since that is what the
    boundary is exchanging with and the number a reader is looking for.
    """
    g = _format_values(values)
    if kappa and len(values) and np.allclose(values, values[0]):
        ambient = np.atleast_1d(values[0])[0] / kappa
        return f'Robin: du/dn + {kappa:g}(u - {ambient:g}) = 0'
    return f'Robin: du/dn + {kappa:g} u = {g}'


# -- classification: a condition -> a coloured, shaped mark ----------------------------


@dataclass
class _Mark:
    """One condition, ready to draw: its colour (weak-form type), its shape (mechanical
    role), and the vertices, values, and covered facets it applies to."""
    color: str
    shape: str          # 'dot' | 'run' | 'arrow' | 'wall' | 'roller' | 'pin' | 'imposed'
    idxs: np.ndarray
    values: np.ndarray
    covered: np.ndarray
    label: str


def _covered_facets(mesh, idxs):
    """Which boundary facets have every vertex in `idxs`.

    A region covers the run of boundary it encloses, while a single named vertex covers
    no facet at all, which keeps a pointwise condition looking pointwise.
    """
    facets = np.asarray(mesh.boundary)
    if len(facets) == 0 or len(idxs) == 0:
        return np.zeros(len(facets), dtype=bool)
    return np.asarray(np.isin(facets, np.asarray(idxs, dtype=int)).all(axis=1), dtype=bool)


def _dirichlet_shape(values, is_edge):
    """The drafting shape a Dirichlet region takes. A scalar field has no mechanical role
    to draw, so it stays a dot regardless of shape. A vector field's region becomes the
    symbol its geometry and components imply: a single point is a pin (it has no extent to
    resist rotation with, so it can never be a clamp, whatever its components); an edge is
    a roller (some component free, so it can still rotate and slide the free way), an
    imposed displacement (every component fixed, some nonzero), or a clamp (every
    component fixed to zero)."""
    if values.shape[1] < 2:
        return 'dot'
    if not is_edge:
        return 'pin'
    if np.isnan(values).any():
        return 'roller'
    if np.any(np.abs(values) > 1e-12):
        return 'imposed'
    return 'wall'


def _classify(bc, mesh):
    """Turn a spec's conditions into drawable marks, and report the field's component
    count. The natural condition is left to the caller, which alone knows which facets
    no condition claimed."""
    from fem.boundary import BCType
    from fem.regions import TimeDependent

    entries = bc.entries(mesh)
    # entries() shows a time-dependent value at t = 0; say so in the label. Same order as
    # entries(): the conditions, then the Robin ones.
    varies = [isinstance(v, TimeDependent) for _, _, v in bc.conditions] + \
             [isinstance(g, TimeDependent) for _, _, g in bc.robin_conditions]
    components = max((values.shape[1] for _, _, values in entries if len(values)), default=1)
    # entries() emits the Robin conditions last and in `robin_conditions` order; only the
    # coefficient is read from there, the half of a Robin condition entries() omits.
    kappas = iter([kappa for _, kappa, _ in bc.robin_conditions])

    marks = []
    for (bc_type, idxs, values), time_dependent in zip(entries, varies):
        if len(idxs) == 0:
            continue
        covered = _covered_facets(mesh, idxs)
        color = BC_COLORS[bc_type.value]
        if bc_type is BCType.DIRICHLET:
            shape = _dirichlet_shape(values, bool(covered.any()))
            label = f'Dirichlet: u = {_format_values(values)}'
        elif bc_type is BCType.ROBIN:
            shape = 'run'
            label = _robin_label(next(kappas, 0.0), values)
        else:
            shape = 'arrow' if values.shape[1] >= 2 else 'run'
            label = (f'Neumann: t = {_format_values(values)}' if components >= 2
                     else f'Neumann: du/dn = {_format_values(values)}')
        if time_dependent:
            label = f'{label} at t = 0 (varies in time)'
        marks.append(_Mark(color, shape, np.asarray(idxs), values, covered, label))
    return marks, components


# -- geometry: the segment and outward normal of an edge region ------------------------


def _spread_along(points, target):
    """Up to `target` of `points`, spread evenly along the run they lie on.

    Sorted by whichever axis they vary most in, so a clamped edge is sampled down its
    length.
    """
    if len(points) <= target:
        return np.arange(len(points))
    span = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(points[:, int(np.argmax(span))])
    return order[np.linspace(0, len(order) - 1, target).astype(int)]


def _edge_geometry(pts):
    """The segment a region's boundary vertices lie on, with a placeholder normal.

    Returns `(p0, p1, normal)` with `p0 -> p1` spanning the edge along the axis it varies
    most in; `_outward` then orients `normal` out of the body.
    """
    center = pts.mean(axis=0)
    extent = np.ptp(pts, axis=0)
    along = int(np.argmax(extent))
    normal_axis = 1 - along
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    p0, p1 = center.copy(), center.copy()
    p0[along], p1[along] = lo[along], hi[along]
    normal = np.zeros(2)
    normal[normal_axis] = 1.0
    return p0, p1, normal


def _outward(normal, edge_center, domain_center):
    """Flip `normal` to point away from `domain_center` (out of the body)."""
    normal = normal.copy()
    axis = int(np.argmax(np.abs(normal)))
    if edge_center[axis] < domain_center[axis]:
        normal[axis] *= -1.0
    return normal


# -- glyph drawers: one shape apiece, coloured by type --------------------------------


def _legend_proxy(ax, color, marker, label):
    """A legend entry for a glyph the legend cannot show directly: a marker hinting the
    shape, drawn nowhere, carrying the value label."""
    if label is not None:
        ax.plot([], [], color=color, marker=marker, linestyle='none', label=label)


def _fill_domain(ax, mesh):
    """Flat fill of the elements, so the panel reads as a body with conditions on it.

    Drawn from the elements rather than the outline, so a domain with a hole in it comes
    out with a hole in it.
    """
    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements)
    ax.tripcolor(triangulation, facecolors=np.zeros(len(mesh.elements)),
                 cmap=ListedColormap([DOMAIN_FILL]), rasterized=True)


def _draw_dots(ax, pts, color, label):
    """Dots at each vertex: the scalar Dirichlet mark. A scalar field (temperature, a
    potential) has no mechanical role to draw as a shape, so this is the only mark it
    ever takes."""
    ax.plot(pts[:, 0], pts[:, 1], 'o', color=color, markersize=4, label=label,
            zorder=GLYPH_ZORDER)


def _point_outward(point, domain_center):
    """The direction from the domain's center to `point`, so a point mark can be oriented
    the same way an edge mark is oriented off its normal. Falls back to +y on the
    degenerate case of a point sitting exactly at the center."""
    direction = np.asarray(point[:2], dtype=float) - domain_center
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-12 else np.array([0.0, 1.0])


def _draw_pin(ax, point, normal, unit, color, label):
    """A lone hatched triangle at a point: a pin. Hatched like a wall (translation is
    fixed there, the same statement a wall makes), but a single point has no extent to
    resist rotation with, so unlike a wall it does not clamp the end. The hatching (not a
    ground line) is what tells it apart from a roller when the two sit on the same edge,
    which they typically do: a roller by itself can still slide as a rigid body, and a
    pin is usually what a spec adds at one of its points to stop that."""
    tri = 0.6 * unit
    tangent = np.array([-normal[1], normal[0]])
    apex = np.asarray(point[:2], dtype=float)
    base = apex + normal * tri
    poly = np.array([apex, base + tangent * 0.5 * tri, base - tangent * 0.5 * tri])
    ax.add_patch(Polygon(poly, closed=True, facecolor='none', edgecolor=color,
                         hatch='////', lw=1.1, zorder=GLYPH_ZORDER))
    _legend_proxy(ax, color, '^', label)


def _draw_run(ax, mesh, covered, color, label, linewidth=3.0, linestyle='solid'):
    """The covered boundary facets as a run: a scalar flux, a Robin edge, or the natural
    boundary."""
    if not covered.any():
        return
    segments = mesh.vertices[np.asarray(mesh.boundary)[covered]]
    ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth,
                                     linestyles=linestyle, zorder=GLYPH_ZORDER))
    ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def _draw_arrows(ax, mesh, points, values, color, label):
    """`values` as fixed-length arrows at a sample of `points`, pointing the way the
    traction (or imposed displacement) acts."""
    keep = _spread_along(points, MAX_BC_ARROWS)
    direction = np.asarray(values)[keep, :2]
    magnitude = np.linalg.norm(direction, axis=1, keepdims=True)
    hat = np.divide(direction, magnitude, out=np.zeros_like(direction), where=magnitude > 0)

    span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    length = BC_ARROW_LENGTH * float(np.max(span[:2]))
    # Heads set explicitly: quiver sizes them off the shaft width, which is a fraction of
    # the axes width, so on a long flat panel the default is a barely visible point.
    ax.quiver(points[keep, 0], points[keep, 1], hat[:, 0], hat[:, 1], color=color,
              angles='xy', scale_units='xy', scale=1/length, width=0.008,
              headwidth=4, headlength=5, headaxislength=4.5, zorder=GLYPH_ZORDER)
    _legend_proxy(ax, color, '>', label)


def _draw_wall(ax, p0, p1, normal, unit, color, label):
    """A hatched wall along the edge: a built-in (clamped) end."""
    thickness = 0.55 * unit
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=2.0, solid_capstyle='round',
            zorder=GLYPH_ZORDER)
    band = np.array([p0, p1, p1 + normal * thickness, p0 + normal * thickness])
    ax.add_patch(Polygon(band, closed=True, facecolor='none', edgecolor=color,
                         hatch='////', lw=0.0, zorder=GLYPH_ZORDER))
    _legend_proxy(ax, color, 's', label)


def _draw_roller(ax, p0, p1, normal, unit, color, label):
    """A row of triangles on a ground line: a roller, held one way, free to rotate and to
    slide the other. The ground line tells it apart from a pin's lone triangle."""
    tri = 0.5 * unit
    tangent = np.array([-normal[1], normal[0]])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.5, solid_capstyle='round',
            zorder=GLYPH_ZORDER)
    edge_len = float(np.linalg.norm(p1 - p0))
    n_tri = int(np.clip(round(edge_len / tri), 2, 4))
    for t in np.linspace(0.18, 0.82, n_tri):
        apex = p0 + t * (p1 - p0)
        base = apex + normal * tri
        poly = np.array([apex, base + tangent * 0.5 * tri, base - tangent * 0.5 * tri])
        ax.add_patch(Polygon(poly, closed=True, facecolor='white', edgecolor=color,
                             lw=1.1, zorder=GLYPH_ZORDER))
    g0, g1 = p0 + normal * tri, p1 + normal * tri
    ax.plot([g0[0], g1[0]], [g0[1], g1[1]], color=color, lw=1.0, zorder=GLYPH_ZORDER)
    _legend_proxy(ax, color, '^', label)


# -- rendering: the shared core, and the panel / overlay it drives ---------------------


def _has_arrow(values):
    """Whether a Dirichlet value carries an imposed-displacement arrow (a nonzero vector)."""
    return values.shape[1] >= 2 and bool(np.any(np.nan_to_num(values)))


def _draw_marks(ax, mesh, coords, marks, *, with_labels):
    """Draw each mark at `coords` (mesh vertices for the panel, deformed positions for an
    overlay). Widens the view where a glyph juts past the body (a wall outboard, an
    arrow beyond the edge) so it is not clipped to a stub."""
    coords = np.asarray(coords)
    place = coords[:, :2]
    lo, hi = place.min(axis=0), place.max(axis=0)
    domain_center = (lo + hi) / 2
    reference = float(np.max(hi - lo))
    unit = DRAFTING_SIZE * reference

    labelled = set()
    juts = False
    for mark in marks:
        label = mark.label if (with_labels and mark.label not in labelled) else None
        labelled.add(mark.label)
        pts = coords[mark.idxs][:, :2]

        if mark.shape == 'dot':
            _draw_dots(ax, pts, mark.color, label)
        elif mark.shape == 'pin':
            normal = _point_outward(pts[0], domain_center)
            _draw_pin(ax, pts[0], normal, unit, mark.color, label)
            if _has_arrow(mark.values):
                _draw_arrows(ax, mesh, pts, mark.values, mark.color, None)
            juts = True
        elif mark.shape == 'run':
            _draw_run(ax, mesh, mark.covered, mark.color, label)
        elif mark.shape == 'arrow':
            _draw_arrows(ax, mesh, pts, mark.values, mark.color, label)
            juts = True
        else:  # wall / imposed / roller
            p0, p1, normal = _edge_geometry(pts)
            normal = _outward(normal, pts.mean(axis=0), domain_center)
            if mark.shape == 'roller':
                _draw_roller(ax, p0, p1, normal, unit, mark.color, label)
            else:
                _draw_wall(ax, p0, p1, normal, unit, mark.color, label)
                if mark.shape == 'imposed':
                    _draw_arrows(ax, mesh, pts, mark.values, mark.color, None)
            juts = True

    if juts:
        pad = 1.3 * max(unit, BC_ARROW_LENGTH * reference)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)


def plot_bc(ax, mesh, bc):
    """Draw what each condition says and where it applies, and where none applies.

    Every mark is labelled with its value, not just its type: a clamp and a prescribed
    displacement are different physics, and drawn only as position they are the same row
    of red dots. Colour is the weak-form type, shape the mechanical role (see the module
    docstring); the legend under the panel pairs each with the value it is set to.

    Whatever is left over carries the natural condition: traction-free, insulated,
    whichever name the equation gives it. That is not the absence of a boundary
    condition: the weak form drops the boundary integral there, which is precisely the
    statement that the flux through that edge is zero. So it is drawn like the rest.
    """
    _fill_domain(ax, mesh)
    # The outline, not the triangulation: this panel is about where the conditions sit,
    # and a fine mesh drawn under them is a grey field that hides the markers.
    plot_boundary(ax, mesh)

    marks, components = _classify(bc, mesh)
    _draw_marks(ax, mesh, mesh.vertices, marks, with_labels=True)

    constrained = np.zeros(len(np.asarray(mesh.boundary)), dtype=bool)
    for mark in marks:
        constrained |= mark.covered
    # "Natural", not "free": free is the elasticity word, and on a heat problem it reads
    # as "nothing here" when the statement is the opposite: du/dn = 0 is an insulated
    # edge. Each demo gives the physical name.
    natural = 'Natural: t = 0' if components >= 2 else 'Natural: du/dn = 0'
    _draw_run(ax, mesh, ~constrained, FREE_COLOR, natural, linewidth=2.5, linestyle='--')

    # Under the panel rather than in it: the legend is the key to a picture whose whole
    # content sits on its edges, and every default position inside the axes covers some
    # of them. The offset is scaled by the domain's aspect (which an equal-aspect axes
    # takes as the shape of its box), so a 4:1 beam does not put the legend through its
    # own tick labels while a square leaves a gap.
    if any(ax.get_legend_handles_labels()[1]):
        span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        aspect = float(np.clip(span[1] / span[0], 0.4, 4.0))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -(0.06 + 0.10/aspect)),
                  ncol=2, frameon=False, fontsize='small')


def overlay_supports(ax, mesh, bc, coords=None):
    """Overlay the condition marks on a panel already showing `mesh`'s shape.

    The same marks `plot_bc` draws, minus the filled body and the legend, so an end
    condition can be read off a result at a glance. Conditions are read off the
    undeformed `mesh` (region selection is by position, and a compressed end no longer
    sits where `on_plane` put it), then drawn at `coords` if given: the deformed vertex
    positions, so a load follows the material point it acts on while a support, whose
    point does not move, stays put.
    """
    place = mesh.vertices if coords is None else np.asarray(coords)
    marks, _ = _classify(bc, mesh)
    _draw_marks(ax, mesh, place, marks, with_labels=False)
