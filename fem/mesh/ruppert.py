import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, KDTree, QhullError

from fem.mesh.mesh import Mesh, boundary_facets, triangle_angles

logger = logging.getLogger(__name__)


def circumcenter(triangle_points):
    '''Centre of the circle through a triangle's three vertices.

    Takes a single `(3, 2)` triangle or a stacked `(..., 3, 2)` array. Solved
    against the first vertex as origin, which keeps the accuracy of a very
    flat triangle (the shape mesh refinement asks about most) rather than
    intersecting bisectors by slope, where a near-horizontal edge loses most of
    the available precision.
    '''
    points = np.asarray(triangle_points, dtype=float)
    origin = points[..., 0, :]
    b = points[..., 1, :] - origin
    c = points[..., 2, :] - origin

    twice_area = 2 * (b[..., 0]*c[..., 1] - b[..., 1]*c[..., 0])
    if np.any(twice_area == 0):
        raise ValueError('a degenerate triangle has no circumcenter')

    b_sq = np.sum(b**2, axis=-1)
    c_sq = np.sum(c**2, axis=-1)
    return origin + np.stack([
        (c[..., 1]*b_sq - b[..., 1]*c_sq) / twice_area,
        (b[..., 0]*c_sq - c[..., 0]*b_sq) / twice_area,
    ], axis=-1)


def segment_angles(vertices, segments):
    '''The smallest angle, in degrees, between two segments meeting at each
    vertex, as `{vertex index: angle}`. Vertices where fewer than two segments
    meet are left out.

    Delaunay refinement is only guaranteed to terminate for inputs whose
    segments meet at 60 degrees or more; below that a corner has to be treated
    specially or refinement will chase it forever.
    '''
    vertices = np.asarray(vertices, dtype=float)
    incident = {}
    for start, end in np.asarray(segments):
        incident.setdefault(int(start), []).append(int(end))
        incident.setdefault(int(end), []).append(int(start))

    angles = {}
    for vertex, neighbours in incident.items():
        if len(neighbours) < 2:
            continue
        directions = vertices[neighbours] - vertices[vertex]
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(lengths == 0, 1, lengths)
        cosines = np.clip(directions @ directions.T, -1.0, 1.0)
        # The diagonal is each direction against itself; only distinct pairs count.
        pairs = np.triu_indices(len(directions), k=1)
        angles[vertex] = float(np.degrees(np.arccos(cosines[pairs])).min())
    return angles

# The sharpest corner (two input segments meeting at a shared vertex) for which
# Ruppert's is proven to finish. Refinement fixes one problem at a time (a segment with
# a vertex inside its diametral circle, or a triangle failing the angle or area test) by
# inserting a point, and stops once none are left. Below 60 degrees two segments sharing
# a corner would each re-create the other's problem, so such a corner is meshed at the
# angle it comes in at rather than refined towards a bound it cannot reach: see
# `_split_point` and `_spans_a_sharp_corner`.
SAFE_INPUT_ANGLE = 60.0

# The largest `min_angle` accepted. Ruppert's proof covers bounds to about 20.7 degrees
# and refinement terminates in practice to roughly 30; past the mid thirties it can
# run forever on any input, so the bound is refused rather than tried.
MAX_MIN_ANGLE = 33.0

# `refine` is capped at this many insertions per input vertex (plus a floor), a
# backstop for a run that will not terminate inside the accepted angle range.
INSERTIONS_PER_INPUT_VERTEX = 500
MIN_INSERTION_CAP = 20_000

# Encroachment is tested in the Thales form (a - p).(b - p) < 0, which is exactly zero
# for a segment's own endpoints and so never reports one inside its own circle (see
# `_circles_containing`). This relative tolerance shrinks the test slightly so a vertex
# a hair outside the circle is not counted in either.
ENCROACHMENT_TOLERANCE = 1e-12

# Twice the area over the longest edge squared: a scale-free flatness measure, ~1 for
# a well-shaped triangle and ~1e-14 for one whose corners lie on a line. Below this a
# triangle counts as collinear to floating-point precision; real triangles, however
# skinny, stay above ~1e-3. See `_is_degenerate`.
DEGENERACY_TOLERANCE = 1e-9

# Vertex indices are packed three-to-an-integer to give a triangle a name that
# survives an insertion; see `_triangle_keys`. Three indices below this bound
# stay inside a signed 64-bit integer, and a mesh of a million vertices is far
# past anything this refinement finishes in reasonable time.
_KEY_STRIDE = np.int64(1) << 20


def _triangle_keys(simplices):
    '''One integer naming each triangle, from its corners in sorted order.

    A triangle has to be named by its corners because qhull renumbers
    `simplices` freely across an insertion; a row index means nothing from one
    pass to the next. Reducing the three corners to a single integer turns
    "does this triangle still exist" into a lookup in a sorted array.

    Takes one `(3,)` triangle or a stacked `(n, 3)` array.
    '''
    corners = np.sort(np.asarray(simplices), axis=-1).astype(np.int64)
    return (corners[..., 0]*_KEY_STRIDE + corners[..., 1])*_KEY_STRIDE + corners[..., 2]


class RuppertsAlgorithm:
    '''Ruppert's Delaunay refinement of a PSLG.

    Starting from a Delaunay triangulation of the input vertices, loop until
    no encroached segments and no skinny triangles remain:

    1. Split encroached segments (priority). A segment is encroached
       when a mesh vertex other than its own endpoints falls inside its
       diametral circle (the circle whose diameter is the segment).  Replace
       it with two halves at its midpoint.  Fixing encroachment can also fix
       nearby skinny triangles.

    2. Insert circumcenters of skinny triangles. A triangle whose
       smallest angle is below `min_angle` is refined by inserting its
       circumcenter.  If that new point would encroach a segment, split the
       segment instead and re-examine the triangle later.

    Each step adds one vertex, which the triangulation absorbs incrementally
    rather than being rebuilt around.

    Why segments are respected: the Delaunay triangulation only knows about points,
    but a segment whose diametral circle contains no other vertex is guaranteed to
    appear as a Delaunay edge, so splitting encroached segments makes the
    unconstrained Delaunay conform to the boundary.

    What is returned: the result keeps only what the PSLG encloses: a
    Delaunay triangulation spans the convex hull, so a non-convex outline also
    produces triangles outside it. Interior/exterior alternates by the
    even-odd rule: a region with an odd number of segment crossings to
    infinity is inside, so a loop inside another is a hole. The returned mesh's
    `boundary_tags` record which input loop each boundary facet came from.

    Ruppert proved termination for inputs whose segments meet at 60 degrees or
    more (`SAFE_INPUT_ANGLE`).  Segments meeting below it are the one case where
    `min_angle` does not hold: the triangle across such a corner is meshed at the
    input's own angle, since no refinement improves it.  Everything away from
    those corners still meets the bound.

    Cost grows steeply in the number of input points: the triangulation itself is
    grown a point at a time, but everything around it rescans the whole mesh per
    insertion.  Simplify a densely sampled outline
    (`fem.mesh.svg.douglas_peucker`) before handing it over.
    '''

    def __init__(self, pslg, min_angle: float = 30, max_area: float | None = None,
                 max_insertions: int | None = None):
        '''Refine `pslg` until every triangle clears both bounds.

        `min_angle` is in degrees: the smallest interior angle any output triangle
        may have, bar the triangles across input corners already sharper than
        `SAFE_INPUT_ANGLE`, which no refinement can improve. Bounds above
        `MAX_MIN_ANGLE` are refused: past it refinement can fail to terminate
        however blunt the input.

        `max_area` is an absolute area, not a fraction of the region; callers
        wanting a fraction scale it themselves. None leaves element size
        unbounded, so a large region comes back as a handful of big triangles.

        `max_insertions` caps the points `refine` may insert before giving up with a
        `RuntimeError`; None is `INSERTIONS_PER_INPUT_VERTEX` per input vertex, at
        least `MIN_INSERTION_CAP`, generous for any run that terminates.
        '''
        if not 0 <= min_angle <= MAX_MIN_ANGLE:
            raise ValueError(
                f'min_angle must be between 0 and {MAX_MIN_ANGLE} degrees, got {min_angle}; '
                f'Ruppert refinement is not guaranteed to terminate above it'
            )
        self.vertices = np.array(pslg.vertices)
        self.n_input_vertices = len(self.vertices)
        self.max_insertions = (
            max(MIN_INSERTION_CAP, INSERTIONS_PER_INPUT_VERTEX * self.n_input_vertices)
            if max_insertions is None else max_insertions
        )
        self.segments = np.array([sorted(seg) for seg in pslg.segments])
        self.segment_loops = np.array(pslg.loop_ids)
        # Per-segment analytic curve, aligned with `self.segments`. A split point on a
        # curved segment is projected onto the curve rather than left at the chord
        # midpoint, so refinement rounds the outline; halves inherit their parent's
        # curve, and it is carried onto the matching boundary facet of the output mesh.
        self.segment_curves = list(pslg.segment_curves)
        self.triangulation = Delaunay(self.vertices)
        self._incremental = False
        self.min_angle = min_angle
        self.max_area = max_area

        # A fixed seed so a mesh is reproducible: the only randomness is the direction of
        # the tiny nudge `_perturb` gives each inserted circumcenter (see there), and
        # insertions happen in a deterministic order, so the whole run is deterministic.
        self._rng = np.random.default_rng(0)

        # Diametral circles, and which segments a vertex sits inside. Both are
        # maintained as segments and vertices are added; see `_diametral_circles`
        # and `get_encroached_segments`. The one full scan is here, where a KD-tree
        # beats testing every circle against every vertex.
        self._circles = None
        # Bad triangles still to refine, newest last; see `refine`. `_live_keys` is
        # how a queued one is checked to still exist, dropped on every insertion.
        self._bad_queue = []
        self._live_keys = None
        centers, radii_sq = self._diametral_circles()
        distances, nearest = KDTree(self.vertices).query(centers)
        # The vertex nearest a diametral centre decides encroachment: anything strictly
        # inside is nearer than the endpoints, which sit on the circle. A segment's own
        # endpoint is excluded so floating-point error at the centre cannot place it a
        # hair inside its own circle; `_circles_containing` keeps that exclusion exact
        # from here on.
        is_own_endpoint = ((nearest == self.segments[:, 0])
                           | (nearest == self.segments[:, 1]))
        self._encroached = (~is_own_endpoint
                            & (distances**2 < radii_sq * (1 - ENCROACHMENT_TOLERANCE)))

        corner_angles = segment_angles(self.vertices, self.segments)
        self.input_angle = min(corner_angles.values(), default=180.0)
        # Corners the angle bound cannot be met at. Their segments split on shells
        # rather than at midpoints, and the triangle across them is accepted as it
        # comes; otherwise refinement chases them forever.
        self.sharp_vertices = {v for v, angle in corner_angles.items()
                               if angle < SAFE_INPUT_ANGLE}
        if self.sharp_vertices:
            logger.warning(
                'segments meet at %.1f degrees somewhere, below the %.0f Delaunay '
                'refinement can hold; the %d corner(s) under it keep their own angle '
                'in the mesh, and cost extra elements around them',
                self.input_angle, SAFE_INPUT_ANGLE, len(self.sharp_vertices),
            )

    def _diametral_circles(self):
        '''Centres and squared radii of every segment's diametral circle.

        Cached: a segment's endpoints never move, so this changes only when the
        segment list does. `del_segment` and `add_segment` drop the cache.
        '''
        if self._circles is None:
            ends = self.vertices[self.segments]
            centers = ends.mean(axis=1)
            radii_sq = np.sum((ends[:, 1] - ends[:, 0])**2, axis=-1) / 4
            self._circles = centers, radii_sq
        return self._circles

    def _circles_containing(self, vertex):
        '''Mask over segments whose diametral circle strictly contains `vertex`.

        Tested in the Thales form: `vertex` is inside the circle on diameter (a, b)
        exactly when angle a-vertex-b is obtuse, i.e. (a - vertex).(b - vertex) < 0.
        That dot product equals |center - vertex|^2 - radius^2 algebraically, but
        unlike the two terms computed apart it is exactly zero when `vertex` is an
        endpoint, so a segment is never judged to contain its own endpoint however
        small it is relative to the coordinate magnitude. The center/radius form lost
        that to cancellation and split the segment forever.
        '''
        ends = self.vertices[self.segments]
        _, radii_sq = self._diametral_circles()
        signed = np.einsum('ij,ij->i', ends[:, 0] - vertex, ends[:, 1] - vertex)
        return signed < -ENCROACHMENT_TOLERANCE * radii_sq

    def _is_encroached(self, segment):
        '''Whether any vertex placed so far falls strictly inside `segment`'s circle.

        For a segment that has just appeared, where the incremental mask has
        nothing to carry forward. Uses the same endpoint-exact dot-product test as
        `_circles_containing`.
        '''
        start, end = self.vertices[segment[0]], self.vertices[segment[1]]
        radius_sq = np.sum((end - start)**2) / 4
        signed = np.einsum('ij,ij->i', start - self.vertices, end - self.vertices)
        return bool(np.any(signed < -ENCROACHMENT_TOLERANCE * radius_sq))

    def get_encroached_segments(self):
        '''Segments with a mesh vertex strictly inside their diametral circle.

        Read off a mask kept up to date as vertices and segments are added,
        rather than rescanned: vertices are only ever appended, so a segment
        becomes encroached exactly when a new vertex lands in its circle, and a
        segment that has just been created is the only one needing a full scan.
        '''
        return self.segments[self._encroached]

    def get_segments_encroached_by(self, vertex):
        '''Segments whose diametral circle would strictly contain `vertex`.'''
        return self.segments[self._circles_containing(vertex)]

    def get_triangle_areas(self, simplices=None):
        '''The area of each triangle. Defaults to every triangle; pass a subset
        to ask about only those.'''
        if simplices is None:
            simplices = self.triangulation.simplices
        corners = self.vertices[simplices]
        edge_a, edge_b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        return 0.5 * np.abs(edge_a[:, 0]*edge_b[:, 1] - edge_a[:, 1]*edge_b[:, 0])

    def _fails_a_bound(self, simplices, segment_mask=None):
        '''Which of `simplices` are too skinny or too large, enclosure aside.

        Takes the triangles to judge rather than reading them off the
        triangulation, so the refinement loop can ask about the handful an
        insertion just created instead of about all of them.
        '''
        angles = triangle_angles(self.vertices[simplices])
        bad = angles.min(axis=-1) < self.min_angle
        if self.sharp_vertices:
            if segment_mask is None:
                segment_mask = self._segment_edges(simplices)
            bad &= ~self._spans_a_sharp_corner(simplices, angles, segment_mask)
        # The area cap survives that exemption: a corner triangle cannot be made
        # less sharp, but it can be made smaller.
        if self.max_area is not None:
            bad |= self.get_triangle_areas(simplices) > self.max_area
        # A degenerate sliver has no usable circumcenter, so it is never bad
        # however small its angle; see `_is_degenerate`.
        bad &= ~self._is_degenerate(simplices)
        return bad

    def _is_degenerate(self, simplices):
        '''Triangles whose corners are collinear to floating-point precision.

        A segment split lands a midpoint exactly on the line through the segment's
        endpoints, and qhull can fan that triple into a zero-area sliver. Its
        circumcenter lands ~1e12 away, wrecking the next incremental insertion, so
        such a sliver must be neither refined nor returned as an element.
        '''
        corners = self.vertices[simplices]
        edges = corners - corners[:, [1, 2, 0]]
        longest_sq = np.sum(edges**2, axis=-1).max(axis=-1)
        # Multiplied through rather than divided, so coincident corners
        # (longest_sq == 0) report degenerate instead of dividing by zero.
        return 2 * self.get_triangle_areas(simplices) <= DEGENERACY_TOLERANCE * longest_sq

    def get_bad_triangles(self):
        '''Interior triangles that violate the angle bound or area cap, in index order.

        Exterior triangles are excluded: refining them would insert
        circumcenters that enlarge the convex hull, creating more exterior
        triangles and never terminating.  They go through the region labelling
        rather than an even-odd test per triangle: a non-convex outline leaves
        hundreds of skinny triangles outside the hull, all of them failing the
        angle bound; one test per region is far cheaper.
        '''
        simplices = self.triangulation.simplices
        segment_mask = self._segment_edges()
        bad = self._fails_a_bound(simplices, segment_mask)
        # Regions are only meaningful once no segment is encroached, which the
        # refinement loop always resolves first.
        bad &= ~self.get_exterior_triangles(segment_mask)
        return simplices[bad]

    def _bad_triangles_created_by(self, vertex_index):
        '''The bad triangles among those that appeared when `vertex_index` went in.

        This is everything the standing queue of bad triangles is missing, which
        rests on two facts.

        A triangle cannot go bad after it is created. Its angles and area come
        from its corners, and corners never move; splitting a segment only
        subdivides it, so the even-odd boundary does not shift either and a
        triangle's enclosure is settled too. So a triangle is bad from the
        moment it exists or fine for as long as it exists, and nothing already
        in the mesh can turn bad unnoticed.

        Every triangle an insertion creates has the inserted vertex as a corner,
        so the ones to examine are found by an integer comparison instead of the
        trigonometry a full rescan costs.

        Enclosure is settled per triangle here rather than by labelling regions.
        That is the wrong trade over a whole mesh (measured at 1.9x slower) and
        the right one over the handful of triangles one insertion makes.
        '''
        simplices = self.triangulation.simplices
        created = simplices[(simplices == vertex_index).any(axis=1)]
        candidates = created[self._fails_a_bound(created)]
        centroids = self.vertices[candidates].mean(axis=1)
        return candidates[self._crossing_counts(centroids) % 2 == 1]

    def _live_triangle_keys(self):
        '''Sorted keys of every triangle that currently exists.

        Rebuilt after an insertion rather than maintained. That is a couple of
        vectorised passes, and it lets a queued triangle be checked without
        asking qhull to locate a point; point location rebuilds its own search
        structure on every insertion, which costs a thousand times more in a
        loop that inserts and then immediately asks.
        '''
        if self._live_keys is None:
            self._live_keys = np.sort(_triangle_keys(self.triangulation.simplices))
        return self._live_keys

    def _refill_bad_queue(self, triangles, replace=True):
        '''Queue `triangles` for refinement, as corner triples rather than rows.

        Rows are qhull's numbering and do not survive an insertion; the corners do.
        '''
        queued = [tuple(sorted(triangle)) for triangle in triangles]
        self._bad_queue = queued if replace else self._bad_queue + queued

    def _next_queued_bad_triangle(self):
        '''The newest queued triangle that still exists, or None if none does.

        Entries go stale as later insertions re-fan the triangles around them,
        which is cheaper to discover here than to track as it happens.
        '''
        keys = self._live_triangle_keys()
        while self._bad_queue:
            triangle = self._bad_queue.pop()
            key = _triangle_keys(triangle)
            position = int(np.searchsorted(keys, key))
            if position < len(keys) and keys[position] == key:
                return triangle
        return None

    def _spans_a_sharp_corner(self, simplices, angles, segment_mask):
        '''Triangles whose smallest angle is one the input already has.

        Where two segments meet below `SAFE_INPUT_ANGLE`, the triangle between
        them holds their angle however small the elements around it get, so
        refining it never ends. Taking it as it stands is what lets a sharp
        outline mesh at all, at the price of the bound holding everywhere else.
        '''
        corner = np.argmin(angles, axis=-1)
        rows = np.arange(len(simplices))
        # The two edges meeting at a vertex are those opposite the other two.
        between_segments = (segment_mask[rows, (corner + 1) % 3]
                            & segment_mask[rows, (corner + 2) % 3])
        return between_segments & np.isin(simplices[rows, corner], list(self.sharp_vertices))

    def _segment_edges(self, simplices=None):
        '''(n_tri, 3) bool mask: which of each triangle's edges is a PSLG segment.

        Column `j` corresponds to the edge opposite vertex `j`, matching the
        layout of `Delaunay.neighbors`, so `neighbors[i, j]` is the triangle
        across a segment when `_segment_edges()[i, j]` is True. Defaults to
        every triangle; pass a subset to ask about only those.
        '''
        if simplices is None:
            simplices = self.triangulation.simplices
        opposite = np.stack([
            np.sort(simplices[:, [1, 2]], axis=1),
            np.sort(simplices[:, [2, 0]], axis=1),
            np.sort(simplices[:, [0, 1]], axis=1),
        ], axis=1)
        # Each vertex pair packed into one integer, so the lookup is a single isin.
        stride = len(self.vertices)
        segments = np.sort(self.segments, axis=1)
        return np.isin(opposite[..., 0]*stride + opposite[..., 1],
                       segments[:, 0]*stride + segments[:, 1])

    def get_regions(self, segment_mask=None):
        '''Label each triangle with an integer region ID.

        Two triangles are in the same region if they can reach each other
        through shared edges without crossing a segment.  Implemented as
        connected components of the triangle adjacency graph with segment
        edges removed.
        '''
        neighbors = self.triangulation.neighbors
        if segment_mask is None:
            segment_mask = self._segment_edges()
        interior_edge = (neighbors != -1) & ~segment_mask
        triangle, edge = np.nonzero(interior_edge)
        dual = coo_matrix(
            (np.ones(len(triangle), dtype=bool), (triangle, neighbors[triangle, edge])),
            shape=(len(neighbors), len(neighbors)),
        )
        _, labels = connected_components(dual, directed=False)
        return labels

    def _crossing_counts(self, points):
        '''How many segments a ray from each point to +x passes through.'''
        starts = self.vertices[self.segments[:, 0]][None, :, :]
        ends = self.vertices[self.segments[:, 1]][None, :, :]
        points = points[:, None, :]

        rises = (starts[..., 1] > points[..., 1]) != (ends[..., 1] > points[..., 1])
        # Only the segments the ray's height crosses can be hit; the rest would
        # divide by a zero rise, so keep them out of the arithmetic entirely.
        height = np.where(rises, ends[..., 1] - starts[..., 1], 1.0)
        crossing_x = starts[..., 0] + (points[..., 1] - starts[..., 1]) * (
            ends[..., 0] - starts[..., 0]) / height
        return (rises & (crossing_x > points[..., 0])).sum(axis=1)

    def get_exterior_triangles(self, segment_mask=None):
        '''Bool mask: True for triangles outside the PSLG boundary.

        Picks one triangle per region, counts how many segments a ray from
        its centroid crosses to reach infinity, and applies the even-odd rule:
        odd crossings = inside, even = outside.  A loop inside another is
        therefore a hole without anyone having to declare it.
        '''
        labels = self.get_regions(segment_mask)
        # A region's representative centroid must be one the even-odd ray cast can
        # trust: a degenerate triangle's sits on the segment its collinear corners
        # straddle, so prefer a non-degenerate representative. An all-degenerate
        # region covers no area and drops out of the mesh, so its label is moot.
        degenerate = self._is_degenerate(self.triangulation.simplices)
        order = np.lexsort((np.arange(len(labels)), degenerate))
        representatives = order[np.unique(labels[order], return_index=True)[1]]
        corners = self.vertices[self.triangulation.simplices[representatives]]
        outside = self._crossing_counts(corners.mean(axis=1)) % 2 == 0
        return outside[labels]

    def _enclosed_mesh(self):
        '''The enclosed triangles as a Mesh, renumbered onto the vertices it uses.'''
        simplices = self.triangulation.simplices
        # A degenerate sliver covers no area and is a qhull artifact along a
        # segment, not an element; drop it whatever its region's label.
        kept = ~self.get_exterior_triangles() & ~self._is_degenerate(simplices)
        elements = simplices[kept]
        if len(elements) == 0:
            raise ValueError(
                'the PSLG encloses no region: every triangle can be reached from '
                'outside without crossing a segment, so there is nothing to mesh'
            )

        used = np.unique(elements)
        renumbered = np.zeros(len(self.vertices), dtype=np.intp)
        renumbered[used] = np.arange(len(used))
        elements = renumbered[elements]

        boundary = boundary_facets(elements)
        boundary_tags = self._trace_boundary(boundary, used, renumbered, self.segment_loops, -1)
        boundary_curves = None
        if any(curve is not None for curve in self.segment_curves):
            boundary_curves = self._trace_boundary(
                boundary, used, renumbered, self.segment_curves, None)
        return Mesh(self.vertices[used], elements, boundary, boundary_curves, boundary_tags)

    def _trace_boundary(self, boundary, used, renumbered, per_segment, missing):
        '''`per_segment[i]` carried onto the boundary facet that matches segment `i`'s
        endpoints, in `boundary` order; `missing` where no segment matches.

        A boundary facet inherits the (possibly split) input segment it lies on: its
        loop id, so an obstacle's rim can be told from the outer wall around it, and its
        curve, so an isoparametric space can place its boundary nodes on the true curve.
        Neither can be recovered from the finished mesh; a boundary is just edges by then.
        '''
        is_used = np.zeros(len(self.vertices), dtype=bool)
        is_used[used] = True
        of_edge = {}
        for (start, end), value in zip(self.segments, per_segment):
            if is_used[start] and is_used[end]:
                of_edge[tuple(sorted((int(renumbered[start]), int(renumbered[end]))))] = value
        return [of_edge.get(tuple(sorted(int(v) for v in facet)), missing) for facet in boundary]

    def _retriangulate(self):
        '''Fold the vertices added since the last call into the triangulation.

        Inserting a point only invalidates the triangles whose circumcircle
        contains it, so qhull re-fans that cavity and leaves the rest standing.
        Rebuilding from scratch instead costs a pass over every vertex already
        placed, once per insertion, and that is half of a refinement run.

        Incremental mode cannot start from a point set with no non-degenerate
        initial simplex, and that is not an exotic input: any four cocircular
        points are one, a square included. It also rules out the `Qz` option that
        would otherwise handle them. So a run rebuilds until qhull will take the
        point set, which the first inserted vertex almost always settles.

        An incremental insertion can also fail partway through a run, with a
        precision error ("wide merge" on nearly-cocircular points, which an
        axis-aligned outline and the circumcenters inserted along a re-entrant
        corner readily produce). A batch rebuild settles the same point set,
        because that path applies qhull's `Qbb`/`Qz` paraboloid scaling that
        incremental mode cannot; the next insertion resumes incrementally.
        '''
        added = self.vertices[len(self.triangulation.points):]
        if not len(added):
            return
        self._live_keys = None
        if self._incremental:
            try:
                self.triangulation.add_points(added)
                return
            except QhullError:
                # Fall through to a rebuild, dropping the incremental state the
                # failed insertion left the triangulation in.
                self._incremental = False
        try:
            self.triangulation = Delaunay(self.vertices, incremental=True)
            self._incremental = True
        except QhullError:
            self.triangulation = Delaunay(self.vertices)

    def refine(self):
        '''Refine until nothing is left to fix, and return the enclosed mesh.

        One problem is fixed per pass, by inserting one point. Encroached
        segments outrank bad triangles: clearing a diametral circle is what
        keeps a segment in the triangulation at all, and it often settles the
        skinny triangles beside it on the way.

        Bad triangles are held in a queue that each insertion tops up with the
        triangles it created, rather than found by rescanning the mesh every
        pass. The queue is only a shortcut: an empty one is confirmed against a
        full scan before returning, so anything it fails to notice costs a
        rescan rather than a mesh with skinny elements left in it.
        '''
        encroached_segments = list(self.get_encroached_segments())
        self._refill_bad_queue(self.get_bad_triangles())

        while True:
            would_encroach = []
            new_vertex = None
            if encroached_segments:
                self.split_segment(encroached_segments.pop())
                new_vertex = len(self.vertices) - 1
            else:
                triangle = self._next_queued_bad_triangle()
                if triangle is None:
                    # The queue is spent, which is the shortcut's answer and not
                    # the algorithm's. Only a full scan finding nothing ends this.
                    remaining = self.get_bad_triangles()
                    if len(remaining) == 0:
                        break
                    self._refill_bad_queue(remaining)
                    continue
                centre = self._perturb(
                    circumcenter(self.vertices[list(triangle)]), list(triangle))
                # Inserting a point inside a segment's diametral circle would cut
                # the mesh off from the outline, so split those segments instead
                # and put the triangle back to be reconsidered once they are gone.
                would_encroach = list(self.get_segments_encroached_by(centre))
                if would_encroach:
                    self._bad_queue.append(triangle)
                else:
                    self.add_vertex(centre)
                    new_vertex = len(self.vertices) - 1
            if new_vertex is not None:
                if len(self.vertices) - self.n_input_vertices > self.max_insertions:
                    raise RuntimeError(
                        f'Ruppert refinement inserted {self.max_insertions} points without '
                        f'clearing the bounds (min_angle={self.min_angle}, '
                        f'max_area={self.max_area}); lower the angle bound, or raise '
                        f'max_insertions if the mesh is meant to be this fine'
                    )
                self._retriangulate()
                self._refill_bad_queue(self._bad_triangles_created_by(new_vertex),
                                       replace=False)
            # `would_encroach` is not in the mask: no vertex was placed to put it
            # there, since the circumcenter that would have was refused.
            encroached_segments = list(self.get_encroached_segments()) + would_encroach

        logger.debug('refined to %d triangles over %d vertices',
                     len(self.triangulation.simplices), len(self.vertices))
        return self._enclosed_mesh()

    def del_segment(self, segment):
        '''Remove `segment`, returning the `(loop_id, curve)` it belonged to.'''
        segment_idx = np.where((self.segments == segment).all(axis=1))[0][0]
        loop_id = int(self.segment_loops[segment_idx])
        curve = self.segment_curves[segment_idx]
        self.segments = np.delete(self.segments, segment_idx, axis=0)
        self.segment_loops = np.delete(self.segment_loops, segment_idx)
        del self.segment_curves[segment_idx]
        self._encroached = np.delete(self._encroached, segment_idx)
        self._circles = None
        return loop_id, curve

    def _perturb(self, circumcenter, triangle):
        '''Nudge an inserted circumcenter a hair off its exact position.

        A circumcenter lies exactly on its triangle's circumcircle, which is the
        cocircular worst case for the incremental Delaunay underneath: on the lifted
        paraboloid the new point is coplanar with the ones it joins, and qhull's facet
        merge can fail with a precision error. One re-entrant corner trips it rarely, but
        a finely sampled smooth outline (an airfoil) packs enough near-cocircular points
        that it trips on most insertions there, and the batch-rebuild recovery in
        `_retriangulate` then dominates the run. Moving the point a fixed tiny fraction of
        its circumradius, in a deterministic direction, breaks the degeneracy at the
        source while leaving it where refinement meant to put it: the offset is far above
        qhull's precision floor and far below any mesh feature. Only interior circumcenters
        are perturbed; a segment split point has to stay on its segment.
        '''
        radius = float(np.linalg.norm(circumcenter - self.vertices[triangle[0]]))
        angle = float(self._rng.uniform(0, 2 * np.pi))
        return circumcenter + 1e-4 * radius * np.array([np.cos(angle), np.sin(angle)])

    def add_vertex(self, vertex):
        # The one place vertices appear, so the one place a segment can newly
        # become encroached by an existing circle.
        self._encroached |= self._circles_containing(vertex)
        self.vertices = np.append(self.vertices, [vertex], axis=0)

    def add_segment(self, segment, loop_id=0, curve=None):
        # A new circle has no history to carry forward, so it is scanned against
        # every vertex placed so far. Its own endpoints lie on it, not inside.
        encroached = self._is_encroached(segment)
        self.segments = np.append(self.segments, [segment], axis=0)
        self.segment_loops = np.append(self.segment_loops, loop_id)
        self.segment_curves.append(curve)
        self._encroached = np.append(self._encroached, encroached)
        self._circles = None

    def _split_point(self, segment):
        '''Where to cut `segment` in two: its midpoint, unless it runs from a
        corner too sharp to mesh, in which case a point on a shell around that
        corner.

        Midpoints do not converge there. Splitting one segment drops a vertex
        inside its neighbour's diametral circle, splitting the neighbour drops
        one back inside the first's, and the two walk into the corner without
        ever clearing each other. Cutting at a power-of-two distance from the
        corner instead puts both splits on the same ladder of radii, and once
        two of them land on one shell they are equidistant from the corner and
        stop encroaching, and the cascade ends after a bounded number of rounds.

        A segment sharp at both ends is laddered from one of them per split,
        and still ends up on shells at each. The midpoint is always the newest
        vertex and so the highest index, which leaves the other corner at index
        0 of the half kept beside it, so that half ladders from there when it
        splits in turn. Renumber vertices and this stops being true.
        '''
        start, end = self.vertices[segment[0]], self.vertices[segment[1]]
        if int(segment[0]) in self.sharp_vertices:
            corner, far = start, end
        elif int(segment[1]) in self.sharp_vertices:
            corner, far = end, start
        else:
            return 0.5 * (start + end)

        length = float(np.linalg.norm(far - corner))
        # The one power of two in [length/3, 2*length/3], so the cut stays near
        # the middle while the radius comes off a ladder shared with every other
        # segment at this corner.
        radius = 2.0 ** np.floor(np.log2(2 * length / 3))
        return corner + (far - corner) * (radius / length)

    def split_segment(self, segment):
        new_vertex_idx = len(self.vertices)
        new_segments = [[segment[0], new_vertex_idx], [segment[1], new_vertex_idx]]
        # Halves inherit the loop and the curve, so a boundary facet can still be traced
        # back to the outline it came from however many times it has been split.
        loop_id, curve = self.del_segment(segment)
        split = self._split_point(segment)
        # On a curved segment the midpoint is projected onto the true curve, so each
        # split moves the outline toward the curve rather than only halving a chord. A
        # smooth curve has no sharp corner, so this never fights the shell splitting.
        if curve is not None:
            split = np.asarray(curve.project(split))
        self.add_vertex(split)
        self.add_segment(new_segments[0], loop_id, curve)
        self.add_segment(new_segments[1], loop_id, curve)
        return new_segments
