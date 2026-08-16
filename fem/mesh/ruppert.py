import logging
from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, KDTree, QhullError

from fem.mesh.mesh import Mesh
from fem.typing import FloatArray
from fem.geometry import (
    calculate_segment_angles,
    calculate_triangle_angles,
    calculate_circumcenter,
    get_boundary_from_vertices_elements,
)

logger = logging.getLogger(__name__)

# The sharpest corner (two input segments meeting at a shared vertex) for which
# Ruppert's is proven to finish. Refinement fixes one problem at a time (a segment with
# a vertex inside its diametral circle, or a triangle failing the angle or area test) by
# inserting a point, and stops once none are left. Below 60 degrees two segments sharing
# a corner would each re-create the other's problem, so such a corner is meshed at the
# angle it comes in at rather than refined towards a bound it cannot reach: see
# `_split_point` and `_spans_a_sharp_corner`.
SAFE_INPUT_ANGLE = 60.0

# A segment's own endpoints sit exactly on its diametral circle, so floating-point
# noise can push them fractionally inside.  This relative tolerance shrinks the test
# circle slightly to prevent a segment from appearing encroached by its own endpoint
# (which would split it forever).
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

    1. **Split encroached segments** (priority).  A segment is *encroached*
       when a mesh vertex other than its own endpoints falls inside its
       diametral circle (the circle whose diameter is the segment).  Replace
       it with two halves at its midpoint.  Fixing encroachment can also fix
       nearby skinny triangles.

    2. **Insert circumcenters of skinny triangles.**  A triangle whose
       smallest angle is below `min_angle` is refined by inserting its
       circumcenter.  If that new point would encroach a segment, split the
       segment instead and re-examine the triangle later.

    Each step adds one vertex, which the triangulation absorbs incrementally
    rather than being rebuilt around.

    **Why segments are respected.**  The Delaunay triangulation only knows
    about points, not segments.  But if a segment's diametral circle contains
    no other vertex, it is guaranteed to appear as a Delaunay edge. Splitting
    encroached segments clears their diametral circles, which makes the
    unconstrained Delaunay conform to the boundary.

    **What is returned.** The result keeps only what the PSLG encloses: a
    Delaunay triangulation spans the convex hull, so a non-convex outline also
    produces triangles outside it. Interior/exterior alternates by the
    even-odd rule: a region with an odd number of segment crossings to
    infinity is inside, so a loop inside another is a hole. After `refine`,
    `boundary_loops` records which input loop each boundary facet came from.

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

    def __init__(self, pslg, min_angle=30, max_area=None):
        '''Refine `pslg` until every triangle clears both bounds.

        `min_angle` is in degrees: the smallest interior angle any output triangle
        may have, bar the triangles across input corners already sharper than
        `SAFE_INPUT_ANGLE`, which no refinement can improve. Ruppert's proof covers
        bounds up to about 20.7 degrees and it holds in practice to roughly 30;
        above that refinement can fail to terminate however blunt the input.

        `max_area` is an absolute area, not a fraction of the region; callers
        wanting a fraction scale it themselves. None leaves element size
        unbounded, so a large region comes back as a handful of big triangles.
        '''
        self.vertices = np.array(pslg.vertices)
        self.segments = np.array([sorted(seg) for seg in pslg.segments])
        self.segment_loops = np.array(getattr(pslg, 'loop_ids',
                                              np.zeros(len(self.segments), dtype=int)))
        self.triangulation = Delaunay(self.vertices)
        self._incremental = False
        self.min_angle = min_angle
        self.max_area = max_area
        # Which loop each boundary facet of the returned mesh came from; set by refine().
        self.boundary_loops = np.zeros(0, dtype=int)

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
        distances, _ = KDTree(self.vertices).query(centers)
        self._encroached = distances**2 < radii_sq * (1 - ENCROACHMENT_TOLERANCE)

        corner_angles = calculate_segment_angles(self.vertices, self.segments)
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
        '''Mask over segments whose diametral circle strictly contains `vertex`.'''
        centers, radii_sq = self._diametral_circles()
        offsets = np.asarray(vertex) - centers
        return np.sum(offsets**2, axis=-1) < radii_sq * (1 - ENCROACHMENT_TOLERANCE)

    def _is_encroached(self, segment):
        '''Whether any vertex placed so far falls strictly inside `segment`'s circle.

        For a segment that has just appeared, where the incremental mask has
        nothing to carry forward.
        '''
        ends = self.vertices[segment]
        center = ends.mean(axis=0)
        radius_sq = np.sum((ends[1] - ends[0])**2) / 4
        offsets = self.vertices - center
        return bool(np.any(np.sum(offsets**2, axis=-1)
                           < radius_sq * (1 - ENCROACHMENT_TOLERANCE)))

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
        angles = calculate_triangle_angles(self.vertices[simplices])
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
        angle bound, and collapsing those to one test per region is the point.
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

        Rows are qhull's numbering and do not survive an insertion; the corners
        do, which is what a queue outliving the pass that filled it needs.
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

        boundary = get_boundary_from_vertices_elements(elements)
        self.boundary_loops = self._trace_boundary_to_loops(boundary, used, renumbered)
        return Mesh(self.vertices[used], elements, boundary)

    def _trace_boundary_to_loops(self, boundary, used, renumbered):
        '''The input loop each boundary facet came from, or -1 if none did.

        This tells an obstacle's rim from the outer wall around it, and it
        cannot be recovered from the finished mesh; a boundary is just edges
        by then.
        '''
        is_used = np.zeros(len(self.vertices), dtype=bool)
        is_used[used] = True
        loop_of_edge = {}
        for (start, end), loop_id in zip(self.segments, self.segment_loops):
            if is_used[start] and is_used[end]:
                loop_of_edge[tuple(sorted((renumbered[start], renumbered[end])))] = int(loop_id)
        return np.array([loop_of_edge.get(tuple(sorted(facet)), -1) for facet in boundary],
                        dtype=int)

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
        '''
        added = self.vertices[len(self.triangulation.points):]
        if not len(added):
            return
        self._live_keys = None
        if self._incremental:
            self.triangulation.add_points(added)
            return
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
                circumcenter = calculate_circumcenter(self.vertices[list(triangle)])
                # Inserting a point inside a segment's diametral circle would cut
                # the mesh off from the outline, so split those segments instead
                # and put the triangle back to be reconsidered once they are gone.
                would_encroach = list(self.get_segments_encroached_by(circumcenter))
                if would_encroach:
                    self._bad_queue.append(triangle)
                else:
                    self.add_vertex(circumcenter)
                    new_vertex = len(self.vertices) - 1
            if new_vertex is not None:
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
        '''Remove `segment`, returning the loop it belonged to.'''
        segment_idx = np.where((self.segments == segment).all(axis=1))[0][0]
        loop_id = int(self.segment_loops[segment_idx])
        self.segments = np.delete(self.segments, segment_idx, axis=0)
        self.segment_loops = np.delete(self.segment_loops, segment_idx)
        self._encroached = np.delete(self._encroached, segment_idx)
        self._circles = None
        return loop_id

    def add_vertex(self, vertex):
        # The one place vertices appear, so the one place a segment can newly
        # become encroached by an existing circle.
        self._encroached |= self._circles_containing(vertex)
        self.vertices = np.append(self.vertices, [vertex], axis=0)

    def add_segment(self, segment, loop_id=0):
        # A new circle has no history to carry forward, so it is scanned against
        # every vertex placed so far. Its own endpoints lie on it, not inside.
        encroached = self._is_encroached(segment)
        self.segments = np.append(self.segments, [segment], axis=0)
        self.segment_loops = np.append(self.segment_loops, loop_id)
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
        # Halves inherit the loop, so a boundary facet can still be traced back to
        # the outline it came from however many times it has been split.
        loop_id = self.del_segment(segment)
        self.add_vertex(self._split_point(segment))
        self.add_segment(new_segments[0], loop_id)
        self.add_segment(new_segments[1], loop_id)
        return new_segments

# Simple meshing functions
def create_rect_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured triangulation of the axis-aligned rectangle spanned by
    `corners` ((x0, y0), (x1, y1)), with `resolution` (nx, ny) nodes per axis.'''
    x_range = np.linspace(corners[0][0], corners[1][0], resolution[0])
    y_range = np.linspace(corners[0][1], corners[1][1], resolution[1])

    vertices = np.array([[x, y] for y in y_range for x in x_range])
    elements = []

    def get_index(i, j):
        return j*resolution[0] + i

    for i in range(resolution[0]-1):
        for j in range(resolution[1]-1):
            if (i + j) % 2 == 0:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i+1, j+1)])
                elements.append([get_index(i, j), get_index(i+1, j+1), get_index(i, j+1)])
            else:
                elements.append([get_index(i, j), get_index(i+1, j), get_index(i, j+1)])
                elements.append([get_index(i+1, j), get_index(i+1, j+1), get_index(i, j+1)])

    boundary = get_boundary_from_vertices_elements(elements)
    mesh = Mesh(vertices, elements, boundary)

    return mesh

def create_box_mesh(
    corners: Sequence[Sequence[float]] | FloatArray,
    resolution: Sequence[int],
) -> Mesh:
    '''A structured tetrahedralization of the axis-aligned box spanned by
    `corners` ((x0, y0, z0), (x1, y1, z1)), with `resolution` (nx, ny, nz) nodes
    per axis.

    Each grid cell is split into six tets by Kuhn's decomposition: every tet
    contains the cell's main diagonal (corner 000 to 111), one per permutation
    of the three axes. Because every cell splits along the same diagonal
    direction, neighbouring cells agree on their shared faces and the mesh is
    conforming, the property that makes this usable for convergence studies,
    where an unstructured generator would confound the refinement rate.
    '''
    nx, ny, nz = resolution
    x_range = np.linspace(corners[0][0], corners[1][0], nx)
    y_range = np.linspace(corners[0][1], corners[1][1], ny)
    z_range = np.linspace(corners[0][2], corners[1][2], nz)

    vertices = np.array([[x, y, z] for x in x_range for y in y_range for z in z_range])

    def get_index(i, j, k):
        return (i*ny + j)*nz + k

    # Corners of a cell, indexed by the bits of (di, dj, dk): corner 5 is
    # (1, 0, 1). The six tets below are written against that numbering.
    KUHN_TETS = [
        (0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
        (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7),
    ]

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner = [
                    get_index(i + (c >> 2 & 1), j + (c >> 1 & 1), k + (c & 1))
                    for c in range(8)
                ]
                elements.extend([[corner[c] for c in tet] for tet in KUHN_TETS])

    boundary = get_boundary_from_vertices_elements(elements)
    return Mesh(vertices, elements, boundary)
