"""Drawing a problem's boundary conditions: what each one says, and where it applies.

Split out of `helpers.py`, which is otherwise a module of one-function drawing
primitives. This is a small picture with its own vocabulary -- who is drawn as dots and
who as arrows, how a value is written, what an unconstrained edge means -- and that
vocabulary is what the conditions panel of every solver demo is read through.
"""
import warnings

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation

from fem.plot.helpers import plot_boundary


# One colour per condition type, held across the ways a condition can be drawn -- so a
# Neumann flux reads as the same thing whether it came out as arrows or as a run of
# boundary. Grey is for the condition nobody wrote, and is meant to recede.
BC_COLORS = {'dirichlet': 'red', 'neumann': 'tab:blue', 'robin': 'tab:orange'}
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
    # component) -- expected here, and `_format_component` already turns the
    # resulting NaN back into 'free'.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (f'{_format_vector(np.nanmin(values, axis=0))} to '
                f'{_format_vector(np.nanmax(values, axis=0))}')


def _spread_along(points, target):
    """Up to `target` of `points`, spread evenly along the run they lie on.

    Sorted by whichever axis they vary most in, so a clamped edge is sampled down its
    length. `_spread_sample`'s 2D binning is no use here: the points of one condition
    usually lie on a line, where a grid of bins collapses to a handful of cells.
    """
    if len(points) <= target:
        return np.arange(len(points))
    span = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(points[:, int(np.argmax(span))])
    return order[np.linspace(0, len(order) - 1, target).astype(int)]


def _fill_domain(ax, mesh):
    """Flat fill of the elements, so the panel reads as a body with conditions on it.

    Drawn from the elements rather than the outline, so a domain with a hole in it comes
    out with a hole in it.
    """
    triangulation = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements)
    ax.tripcolor(triangulation, facecolors=np.zeros(len(mesh.elements)),
                 cmap=ListedColormap([DOMAIN_FILL]), rasterized=True)


def _draw_arrows(ax, mesh, points, values, color, label):
    """Draw `values` as fixed-length arrows at a sample of `points`.

    Which kind of arrow it is -- a traction applied, or a displacement imposed -- is
    said by the colour and spelled out in the label, since the two are different physics
    and were previously the same picture.
    """
    keep = _spread_along(points, MAX_BC_ARROWS)
    direction = np.asarray(values)[keep, :2]
    magnitude = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = np.divide(direction, magnitude, out=np.zeros_like(direction), where=magnitude > 0)

    span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    length = BC_ARROW_LENGTH * float(np.max(span[:2]))
    # Heads set explicitly: quiver sizes them off the shaft width, which is a fraction of
    # the axes width, so on a long flat panel the default is a barely visible point.
    ax.quiver(points[keep, 0], points[keep, 1], unit[:, 0], unit[:, 1], color=color,
              angles='xy', scale_units='xy', scale=1/length, width=0.008,
              headwidth=4, headlength=5, headaxislength=4.5)
    ax.plot([], [], color=color, marker='>', linestyle='none', label=label)
    # An arrow on the boundary points out of the domain, and the axes limits are the
    # domain -- so without this it is clipped to a stub at the edge, which is what a
    # traction pulling on a plate looked like.
    ax.margins(*(length / np.maximum(span[:2], length)))


def _covered_facets(mesh, idxs):
    """Which boundary facets have every vertex in `idxs`.

    A region covers the run of boundary it encloses, while a single named vertex covers
    no facet at all -- which is what keeps a pointwise condition looking pointwise.
    """
    facets = np.asarray(mesh.boundary)
    if len(facets) == 0 or len(idxs) == 0:
        return np.zeros(len(facets), dtype=bool)
    return np.isin(facets, np.asarray(idxs, dtype=int)).all(axis=1)


def _draw_run(ax, mesh, covered, color, label, linewidth=3.0, linestyle='solid'):
    """Draw the covered boundary facets as a run, labelled once."""
    if not covered.any():
        return
    segments = mesh.vertices[np.asarray(mesh.boundary)[covered]]
    ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth,
                                     linestyles=linestyle))
    # The collection itself carries no legend handle worth showing, so the entry is a
    # proxy line drawn nowhere.
    ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


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


def plot_bc(ax, mesh, bc):
    """Draw what each condition says and where it applies, and where none applies.

    Every mark is labelled with its value, not just its type: a clamp and a prescribed
    displacement are different physics, and drawn only as position they are the same row
    of red dots.

    Dirichlet is per-vertex, so it is drawn as dots -- plus arrows where it pins a vector
    somewhere other than zero, which is a displacement being imposed rather than held.
    Neumann is arrows where its value is a vector, since then the direction is part of
    what was said; a scalar flux acts along a normal this panel does not show, so it is
    drawn as a run of boundary rather than an arrow pointing somewhere misleading. Robin
    is always a run: it is a condition on facets, not on nodes.

    Whatever is left over carries the *natural* condition -- traction-free, insulated,
    whichever name the equation gives it. That is not the absence of a boundary
    condition: the weak form drops the boundary integral there, which is precisely the
    statement that the flux through that edge is zero. So it is drawn like the rest.
    """
    from fem.boundary import BCType

    _fill_domain(ax, mesh)
    # The outline, not the triangulation: this panel is about where the conditions sit,
    # and a fine mesh drawn under them is a grey field that hides the markers. The mesh
    # itself has demos of its own.
    plot_boundary(ax, mesh)

    entries = bc.entries(mesh)
    constrained = np.zeros(len(np.asarray(mesh.boundary)), dtype=bool)
    labelled = set()
    # entries() emits the Robin conditions last and in `robin_conditions` order. Only
    # the coefficient is read from there: it is half of what a Robin condition says and
    # the only half entries() does not carry.
    kappas = iter([kappa for _, kappa, _ in bc.robin_conditions])
    # How many components the unknown has, taken from the conditions themselves, so the
    # free boundary can be described in the same terms they are.
    components = max((values.shape[1] for _, _, values in entries if len(values)), default=1)

    for bc_type, idxs, values in entries:
        color = BC_COLORS[bc_type.value]
        covered = _covered_facets(mesh, idxs)
        constrained |= covered
        points = mesh.vertices[idxs]

        if bc_type is BCType.DIRICHLET:
            label = f'Dirichlet: u = {_format_values(values)}'
            ax.plot(points[:, 0], points[:, 1], 'o', color=color, markersize=4,
                    label=None if label in labelled else label)
            # A nonzero vector pin is a displacement being applied; say which way. No
            # label of its own -- the dots underneath it already carry one. A roller's
            # free (NaN) component is not a displacement being imposed, so it must not
            # itself trigger an arrow -- nan_to_num zeroes it before the check.
            if values.shape[1] >= 2 and np.any(np.nan_to_num(values)):
                _draw_arrows(ax, mesh, points, values, color, None)
            labelled.add(label)
            continue

        if bc_type is BCType.ROBIN:
            label = _robin_label(next(kappas, 0.0), values)
        elif components >= 2:
            label = f'Neumann: t = {_format_values(values)}'
        else:
            label = f'Neumann: du/dn = {_format_values(values)}'

        shown = None if label in labelled else label
        labelled.add(label)
        if bc_type is BCType.NEUMANN and values.shape[1] >= 2:
            _draw_arrows(ax, mesh, points, values, color, shown)
        else:
            _draw_run(ax, mesh, covered, color, shown)

    # "Natural", not "free": free is the elasticity word, and on a heat problem it reads
    # as "nothing here" when the statement is the opposite -- du/dn = 0 is an insulated
    # edge, which is why an unconstrained plate levels off at the mean of where it
    # started rather than cooling towards anything. Each demo gives the physical name.
    natural = 'Natural: t = 0' if components >= 2 else 'Natural: du/dn = 0'
    _draw_run(ax, mesh, ~constrained, FREE_COLOR, natural, linewidth=2.5, linestyle='--')

    # Under the panel rather than in it: the legend is the key to a picture whose whole
    # content sits on its edges, and every default position inside the axes covers some
    # of them. Placed here rather than left to `Plotter.format_axs`, so every panel of
    # conditions in the gallery reads the same way.
    #
    # The offset is a fraction of the axes *height*, while the tick labels it has to
    # clear are a fixed size -- so it is scaled by the domain's aspect, which an
    # equal-aspect axes takes as the shape of its own box. Without that a 4:1 beam puts
    # the legend through its own tick labels while a square leaves a gap.
    if any(ax.get_legend_handles_labels()[1]):
        span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        aspect = float(np.clip(span[1] / span[0], 0.4, 4.0))
        # Two columns, not four: the labels now carry values, and four of them across a
        # narrow panel runs off both sides of it.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -(0.06 + 0.10/aspect)),
                  ncol=2, frameon=False, fontsize='small')


# -- support/load glyphs overlaid on a deformed shape ---------------------------------
#
# `plot_bc` above draws a standalone conditions panel -- filled body, per-vertex dots,
# a legend of values. `overlay_supports` draws the *other* view: only the drafting
# symbols (a clamp's hatched wall, a pin's triangles, a load's arrows), laid over a
# panel that already shows a shape, so a buckled mode can be read for its end conditions
# without a legend. The two are complementary, and the buckling demo shows both.

GLYPH_COLOR = 'black'
GLYPH_ZORDER = 6


def _dirichlet_kind(values):
    """Classify a Dirichlet region: 'pin' (a component left free), 'driven' (every
    component fixed, some to a nonzero value -- an imposed displacement), or 'clamp'
    (every component fixed to zero)."""
    if values.size == 0 or np.isnan(values).any():
        return 'pin'
    return 'driven' if np.any(np.abs(values) > 1e-12) else 'clamp'


def _edge_geometry(pts):
    """The segment a region's boundary vertices lie on, and its outward unit normal.

    Returns `(p0, p1, normal)` with `p0 -> p1` spanning the edge along the axis it
    varies most in, and `normal` the perpendicular pointing away from the region's own
    centre of the domain -- which for an end edge of a column is straight out of the end.
    """
    center = pts.mean(axis=0)
    extent = np.ptp(pts, axis=0)
    along = int(np.argmax(extent))
    normal_axis = 1 - along
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    p0, p1 = center.copy(), center.copy()
    p0[along], p1[along] = lo[along], hi[along]
    normal = np.zeros(2)
    # Point out of the domain: away from the domain centroid is unknown from one edge, so
    # use the axis the edge sits at an extreme of. A caller passes the outward reference.
    normal[normal_axis] = 1.0
    return p0, p1, normal


def _outward(normal, edge_center, domain_center):
    """Flip `normal` to point away from `domain_center` (out of the body)."""
    normal = normal.copy()
    axis = int(np.argmax(np.abs(normal)))
    if edge_center[axis] < domain_center[axis]:
        normal[axis] *= -1.0
    return normal


def _draw_wall(ax, p0, p1, normal, unit):
    """A hatched wall along the edge: a built-in (clamped) end."""
    thickness = 0.55 * unit
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GLYPH_COLOR, lw=2.0,
            solid_capstyle='round', zorder=GLYPH_ZORDER)
    band = np.array([p0, p1, p1 + normal * thickness, p0 + normal * thickness])
    ax.add_patch(Polygon(band, closed=True, facecolor='none', edgecolor=GLYPH_COLOR,
                         hatch='////', lw=0.0, zorder=GLYPH_ZORDER))


def _draw_pin_run(ax, p0, p1, normal, unit):
    """A row of triangles along the edge, on a ground line: a pin/roller support -- held
    transversely but free to rotate (and, on a roller, to slide along the edge)."""
    tri = 0.5 * unit
    tangent = np.array([-normal[1], normal[0]])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GLYPH_COLOR, lw=1.5,
            solid_capstyle='round', zorder=GLYPH_ZORDER)
    edge_len = float(np.linalg.norm(p1 - p0))
    n_tri = int(np.clip(round(edge_len / tri), 2, 4))
    for t in np.linspace(0.18, 0.82, n_tri):
        apex = p0 + t * (p1 - p0)
        base = apex + normal * tri
        poly = np.array([apex, base + tangent * 0.5 * tri, base - tangent * 0.5 * tri])
        ax.add_patch(Polygon(poly, closed=True, facecolor='white', edgecolor=GLYPH_COLOR,
                             lw=1.1, zorder=GLYPH_ZORDER))
    g0, g1 = p0 + normal * tri, p1 + normal * tri
    ax.plot([g0[0], g1[0]], [g0[1], g1[1]], color=GLYPH_COLOR, lw=1.0, zorder=GLYPH_ZORDER)


def _draw_anchor(ax, point):
    """A small dot at one anchored vertex -- the single point that ties off a pinned
    column's rigid axial slide. A dot, not a support symbol: it is numerical scaffolding,
    not a wall or roller the reader should weigh against the ones on the ends."""
    ax.plot(point[0], point[1], marker='o', markersize=4, markerfacecolor='white',
            markeredgecolor=GLYPH_COLOR, markeredgewidth=1.0, zorder=GLYPH_ZORDER + 1)


def _draw_load(ax, p0, p1, direction, length):
    """A few arrows along the edge, heads on it, pointing the traction's way -- an applied
    force (on a free edge) or the push of an imposed displacement (off a wall)."""
    magnitude = float(np.linalg.norm(direction[:2]))
    if magnitude == 0.0:
        return
    unit_dir = np.asarray(direction[:2], dtype=float) / magnitude
    heads = [p0 + t * (p1 - p0) for t in np.linspace(0.25, 0.75, 3)]
    xs = [h[0] for h in heads]
    ys = [h[1] for h in heads]
    ax.quiver(xs, ys, [unit_dir[0] * length] * 3, [unit_dir[1] * length] * 3,
              angles='xy', scale_units='xy', scale=1.0, color=GLYPH_COLOR, width=0.006,
              headwidth=4, headlength=5, headaxislength=4.5, pivot='tip', zorder=GLYPH_ZORDER)


def overlay_supports(ax, mesh, bc, coords=None):
    """Overlay support and load glyphs on a panel already showing `mesh`'s shape.

    Reads the conditions off the *undeformed* `mesh` (region selection is by position, and
    a compressed end no longer sits where `on_plane` put it), then draws each at `coords`
    if given -- the deformed vertex positions, so a load follows the material point it acts
    on while a support, whose point does not move, stays put. Adds only the symbols, no
    legend; `plot_bc` is the panel that spells the values out.
    """
    from fem.boundary import BCType

    place = mesh.vertices if coords is None else np.asarray(coords)
    span = place[:, :2].max(axis=0) - place[:, :2].min(axis=0)
    domain_center = (place[:, :2].max(axis=0) + place[:, :2].min(axis=0)) / 2
    # Sized off the long dimension (the column's length), which barely moves, rather than
    # the slender one -- a mode's transverse bow swells the short extent by several times,
    # and glyphs sized to it would shrink and grow from panel to panel.
    reference = float(np.max(span))
    unit = 0.055 * reference       # support-glyph size
    load_len = 0.07 * reference    # load-arrow length

    for bc_type, idxs, values in bc.entries(mesh):
        if len(idxs) == 0:
            continue
        pts = place[idxs][:, :2]

        if bc_type is BCType.DIRICHLET:
            kind = _dirichlet_kind(values)
            if len(idxs) == 1:
                _draw_anchor(ax, pts[0])
                continue
            p0, p1, normal = _edge_geometry(pts)
            normal = _outward(normal, pts.mean(axis=0), domain_center)
            if kind == 'pin':
                _draw_pin_run(ax, p0, p1, normal, unit)
            else:
                _draw_wall(ax, p0, p1, normal, unit)
                if kind == 'driven':
                    _draw_load(ax, p0, p1, np.nanmean(values[:, :2], axis=0), load_len)
        else:
            p0, p1, _ = _edge_geometry(pts)
            _draw_load(ax, p0, p1, np.nanmean(values[:, :2], axis=0), load_len)

    # The glyphs jut past the domain (a wall outboard, an arrow's tail beyond the edge);
    # widen the view so they are not clipped to stubs at the panel's edge.
    pad = 1.3 * max(unit, load_len)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - 0.2 * pad, y1 + 0.2 * pad)
