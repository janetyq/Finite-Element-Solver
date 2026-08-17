import re
import xml.etree.ElementTree as ET

import numpy as np
import svg.path  # pyright: ignore[reportMissingImports]

from fem.geometry import calculate_polygon_area, point_in_polygon
from fem.mesh.curves import CubicBezier

# A cubic is subdivided until its control points lie within this fraction of its own
# control-box extent of its chord. Curvature-adaptive and scale-free: a gentle cubic
# becomes one chord, a sharp one a few. Fine enough that Douglas-Peucker, not this, sets
# the final density; the retained curve makes that reduction lossless anyway.
_CUBIC_FLATNESS = 0.01


def _document_height(root):
    '''Height of the SVG user-space box, or None if the file does not say.

    Needed to mirror the artwork: SVG's y axis points down the page, so a path
    read literally arrives upside down in any y-up frame.
    '''
    height = root.get('height')
    if height is not None:
        # Lengths may carry a unit ('737.6px'); the number is what matters here.
        number = re.match(r'\s*([0-9.eE+-]+)', height)
        if number:
            return float(number.group(1))

    view_box = root.get('viewBox')
    if view_box is not None:
        bounds = view_box.replace(',', ' ').split()
        if len(bounds) == 4:
            return float(bounds[1]) + float(bounds[3])
    return None


def _subdivide_cubic(control):
    '''De Casteljau split of a cubic at t=0.5 into its left and right halves.'''
    p0, p1, p2, p3 = control
    p01, p12, p23 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    p012, p123 = (p01 + p12) / 2, (p12 + p23) / 2
    mid = (p012 + p123) / 2
    return np.array([p0, p01, p012, mid]), np.array([mid, p123, p23, p3])


def _perp_distance(start, end, point):
    '''Distance from `point` to the line through `start` and `end`.'''
    se = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = np.hypot(se[0], se[1])
    if length == 0:
        return float(np.hypot(*(np.asarray(point, dtype=float) - np.asarray(start, dtype=float))))
    return float(abs(se[0] * (start[1] - point[1]) - se[1] * (start[0] - point[0])) / length)


def _flatten_cubic(control, flatness, depth=0, max_depth=16):
    '''Points on the cubic (its end, not its start), subdivided until flat to `flatness`.

    Curvature-adaptive: a gentle cubic returns just its endpoint, a sharp one is split
    until each piece's control points lie within `flatness` of its chord. The start point
    is owned by the previous path segment, so joining two segments never doubles a vertex.
    '''
    p0, p1, p2, p3 = control
    if (depth >= max_depth
            or max(_perp_distance(p0, p3, p1), _perp_distance(p0, p3, p2)) <= flatness):
        return [p3]
    left, right = _subdivide_cubic(control)
    return (_flatten_cubic(left, flatness, depth + 1, max_depth)
            + _flatten_cubic(right, flatness, depth + 1, max_depth))


def _read_svg_loops(svg_file):
    '''Closed outlines of an SVG, as `(points, segment_curves)` per loop.

    `points` is the loop's ring, without a repeated closing vertex (`PSLG.from_loops`
    supplies the wraparound edge). `segment_curves[i]` is the analytic `Curve` that
    segment `points[i] -> points[i+1]` (wrapping) lies on: one shared `CubicBezier` for the
    pieces a cubic path segment was sampled into, `None` for a straight one. Per segment,
    not per point, so a cubic's first piece stays on its curve at the joint with a
    neighbour rather than being read as a straight chord.

    Returned in a y-up frame mirrored about the document height, curves mirrored with it,
    so the artwork plots the way it looks in a browser rather than flipped.
    '''
    root = ET.parse(svg_file).getroot()
    raw_loops = []   # (points, seg_curves) in the SVG's own y-down frame
    for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
        d = path.get("d")
        if d is None:
            continue
        points, seg_curves = [], []
        for segment in svg.path.parse_path(d):
            start = np.array([segment.start.real, segment.start.imag])
            end = np.array([segment.end.real, segment.end.imag])
            if isinstance(segment, svg.path.path.Move):
                assert len(points) == 0
                points.append(start)
            elif isinstance(segment, svg.path.path.Line):
                points.append(end)
                seg_curves.append(None)
            elif isinstance(segment, svg.path.path.CubicBezier):
                control = np.array([
                    start,
                    [segment.control1.real, segment.control1.imag],
                    [segment.control2.real, segment.control2.imag],
                    end,
                ])
                curve = CubicBezier(*control)
                extent = float(np.max(control.max(axis=0) - control.min(axis=0)))
                # One curve object shared by every piece, so the pieces read as one arc.
                for piece_end in _flatten_cubic(control, _CUBIC_FLATNESS * extent):
                    points.append(piece_end)
                    seg_curves.append(curve)
            elif isinstance(segment, svg.path.path.Close):
                # A cubic sampled through its true endpoint can land exactly back on the
                # Move point when the artwork closes there; that duplicate would be a
                # zero-length wrap edge, so drop it and let its curve become the loop's
                # closing segment. A path that does not close exactly gets a straight
                # closing segment, the wraparound `from_loops` supplies.
                if len(points) >= 2 and np.allclose(points[-1], points[0]):
                    points.pop()
                    wrap_curve = seg_curves.pop() if seg_curves else None
                else:
                    wrap_curve = None
                if points:
                    seg_curves.append(wrap_curve)
                    raw_loops.append((points, seg_curves))
                points, seg_curves = [], []

    if not raw_loops:
        return []

    # Fall back to the artwork's own extent when the file declares no size: the mirror
    # line only shifts the result, and shape is what callers use.
    height = _document_height(root)
    if height is None:
        height = max(pt[1] for points, _ in raw_loops for pt in points)

    def mirror(pt):
        return [float(pt[0]), height - float(pt[1])]

    # A cubic's pieces share one curve object; mirror each distinct curve once so the
    # mirrored pieces stay shared too (which the segment folding below reads as identity).
    mirrored_curves = {}
    result = []
    for points, seg_curves in raw_loops:
        mirrored_seg_curves = []
        for curve in seg_curves:
            if curve is None:
                mirrored_seg_curves.append(None)
                continue
            mirrored = mirrored_curves.get(id(curve))
            if mirrored is None:
                mirrored = CubicBezier(*[mirror(c) for c in curve.controls])
                mirrored_curves[id(curve)] = mirrored
            mirrored_seg_curves.append(mirrored)
        result.append(([mirror(pt) for pt in points], mirrored_seg_curves))
    return result


def read_svg_to_list_of_path_points(svg_file):
    '''Reads an SVG file and returns a list of closed loop paths, each a list of points.

    Points come back in a y-up frame, mirrored about the document height, so the artwork
    plots the way it looks in a browser rather than flipped. This drops the per-segment
    curve tags `read_svg_to_pslg` uses; it is the plain point view its callers want.
    '''
    return [points for points, _ in _read_svg_loops(svg_file)]

def douglas_peucker(points, epsilon):
    '''
    Simplifies a curve by reducing the number of points while preserving the overall shape.

    Input:
        points - Nx2 array of points describing the path of a curve (in order)
        epsilon - the distance from a line between two points at which a point will be kept
    
    Output:
        a nx2 array of points describing a simplified curve using the Douglas-Peucker algorithm, n < N

    The algorithm recursively keeps points that are furthest away from the line segment between pairs of points
    and stops when the furthest points are less than epsilon distance away
    '''
    if len(points) <= 2:
        return points

    def perp_distance(start, end, point):
        se_vector = start - end
        sp_vector = start - point
        # The 2D cross product written out: np.cross on 2-vectors is deprecated in
        # NumPy 2 and slated for removal.
        cross = se_vector[0]*sp_vector[1] - se_vector[1]*sp_vector[0]
        return np.abs(cross) / np.linalg.norm(se_vector)

    furthest_dist = 0
    furthest_p_idx = None
    start, end = points[0], points[-1]
    for p_idx, point in enumerate(points[1:-1], start=1):
        dist = perp_distance(start, end, point)
        if dist > furthest_dist:
            furthest_dist = dist
            furthest_p_idx = p_idx
    
    # furthest_p_idx stays None when no interior point beat furthest_dist, which
    # a collinear run does; the recursion below would slice with None+1.
    if furthest_p_idx is None or furthest_dist < epsilon:
        return np.array([start, end])
    else:
        return np.concatenate([douglas_peucker(points[:furthest_p_idx+1], epsilon)[:-1], douglas_peucker(points[furthest_p_idx:], epsilon)])


def _simplify_indices(points, epsilon):
    '''Douglas-Peucker on an open polyline, returning the sorted indices it keeps.

    The index-returning form of `douglas_peucker`, so a caller can carry per-point or
    per-segment data through the simplification: which point survives, not just where it
    lands. The endpoints are always kept.
    '''
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= 2:
        return list(range(n))

    def recurse(lo, hi):
        start, end = points[lo], points[hi]
        furthest_dist, furthest = 0.0, None
        for k in range(lo + 1, hi):
            dist = _perp_distance(start, end, points[k])
            if dist > furthest_dist:
                furthest_dist, furthest = dist, k
        if furthest is None or furthest_dist < epsilon:
            return [lo, hi]
        return recurse(lo, furthest)[:-1] + recurse(furthest, hi)

    return recurse(0, n - 1)


def _fold_segment_curves(seg_curves, keep, n):
    '''The curve of each simplified segment, folded from the originals it spans.

    A simplified segment from kept point `keep[j]` to `keep[j+1]` replaces the original
    segments between them; it keeps their common curve, or `None` where they disagree (a
    chord that merged across a curve/line junction is straight). The last simplified
    segment wraps back to the first kept point, spanning the original closing segment.
    '''
    folded = []
    m = len(keep)
    for j in range(m):
        a, b = keep[j], keep[(j + 1) % m]
        span = list(range(a, b)) if b > a else list(range(a, n)) + list(range(0, b))
        spanned = [seg_curves[s] for s in span]
        common = spanned[0] if spanned and all(s is spanned[0] for s in spanned) else None
        folded.append(common)
    return folded


def read_svg_to_pslg(svg_file, tolerance=0.005):
    '''Read an SVG file and return a PSLG of its closed outlines.

    Each outline is simplified with Douglas-Peucker against its own bounding-box extent,
    so small features survive even when the drawing spans a wide range of scales.
    `tolerance` is the fraction of each loop's extent below which points are dropped.

    Cubic Bezier segments keep their `CubicBezier` curve: simplification only moves the
    outline's vertices, while the curve carries the true shape onto the mesh, so meshing
    and refinement round the outline to the curve rather than freezing the sampled polygon.
    '''
    loops, segment_curves = [], []
    for points, seg_curves in _read_svg_loops(svg_file):
        loop = np.asarray(points, dtype=float)
        if len(loop) < 3:
            continue
        extent = float(np.max(np.max(loop, axis=0) - np.min(loop, axis=0)))
        keep = _simplify_indices(loop, tolerance * extent)
        if len(keep) < 3:
            continue
        loops.append(loop[keep])
        segment_curves.append(_fold_segment_curves(seg_curves, keep, len(loop)))
    return PSLG.from_loops(loops, segment_curves=segment_curves)


def _find_crossing_segments(vertices, segments):
    '''The first pair of segments that properly cross, or None.

    Pairs sharing an endpoint are allowed to touch there, so they are skipped.
    Compares every pair, which is quadratic in the input size (the outline, not
    the mesh refined from it).
    '''
    starts, ends = vertices[segments[:, 0]], vertices[segments[:, 1]]

    def side_of(line_start, line_end, point):
        '''Sign of which side of a directed line each point falls on.'''
        return ((line_end[..., 0] - line_start[..., 0]) * (point[..., 1] - line_start[..., 1])
                - (line_end[..., 1] - line_start[..., 1]) * (point[..., 0] - line_start[..., 0]))

    rows, cols = np.asarray(starts[:, None]), np.asarray(starts[None, :])
    row_ends, col_ends = ends[:, None], ends[None, :]
    # A proper crossing puts each segment's endpoints on opposite sides of the other.
    straddles_row = (side_of(rows, row_ends, cols) > 0) != (side_of(rows, row_ends, col_ends) > 0)
    straddles_col = (side_of(cols, col_ends, rows) > 0) != (side_of(cols, col_ends, row_ends) > 0)
    crossing = straddles_row & straddles_col

    shares_endpoint = (segments[:, None, :, None] == segments[None, :, None, :]).any(axis=(2, 3))
    crossing &= ~shares_endpoint
    crossing &= np.triu(np.ones_like(crossing), k=1).astype(bool)

    found = np.argwhere(crossing)
    return tuple(found[0]) if len(found) else None


def _loop_point_curves(spec, n_points):
    '''Normalize a loop's curve spec to one entry per point.

    `None` is a straight loop; a single `Curve` puts the whole loop on it (a circular
    hole); a sequence gives one curve, or `None`, per point, for an outline that is part
    straight and part arc (a filleted corner).
    '''
    if spec is None:
        return [None] * n_points
    if isinstance(spec, (list, tuple, np.ndarray)):
        if len(spec) != n_points:
            raise ValueError(f'per-point curves must have {n_points} entries, got {len(spec)}')
        return list(spec)
    return [spec] * n_points


class PSLG:
    '''A planar straight-line graph: vertices, plus the segments a mesh must respect.

    `loop_ids` says which closed outline each segment came from, so a caller can
    tell an obstacle's boundary from an enclosing box's after meshing. It is all
    zeros unless the graph was built by `from_loops`.

    `segment_curves` is the analytic `Curve` (or `None`) each segment samples, aligned
    with `segments`. A segment on a curve has its split points projected onto it during
    meshing, and the curve is carried onto the output mesh's matching boundary facet, so
    refinement rounds the outline instead of only subdividing its chords. All `None` for
    a straight-line outline, the default. Per segment rather than per loop, so an outline
    that is part straight and part arc (a filleted corner) can curve only its arc.
    '''

    def __init__(self, vertices, segments=None, loop_ids=None, segment_curves=None):
        self.vertices = vertices
        if segments is None:
            self.segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])
        else:
            self.segments = np.asarray(segments)
        self.loop_ids = (np.zeros(len(self.segments), dtype=int) if loop_ids is None
                         else np.asarray(loop_ids))
        self.segment_curves = (list(segment_curves) if segment_curves is not None
                               else [None] * len(self.segments))

    @classmethod
    def from_loops(cls, loops, curves=None, segment_curves=None):
        '''A PSLG spanning several closed outlines.

        What each loop *means* is decided when meshing, by the even-odd rule: a
        loop inside another is a hole, a loop beside it is a separate piece. So
        the caller draws the outlines and does not also have to label them.

        Curves may be given one of two ways, at most one of them:

        `curves` is a per-loop entry, each either `None` (a straight outline), a single
        `Curve` (the whole loop samples it, as a circular hole does), or a per-point
        sequence tagging which `Curve` each point came from (for an outline that is part
        straight, part arc). A segment lies on a curve only when both its endpoints came
        from that same curve, so the chord joining a straight run to an arc stays straight.

        `segment_curves` is a per-loop sequence giving the `Curve` (or `None`) of each
        segment directly, one entry per point (segment `i` is point `i -> i+1`, wrapping).
        Used where the curve is known per segment rather than per point, as a traced SVG
        path is: a cubic's own first segment stays on its curve at a joint, which the
        per-point rule above would read as straight. A per-point and a per-segment sequence
        are both length `n`, so they cannot be told apart; hence the separate parameter.
        '''
        vertices, segments, loop_ids, out_curves = [], [], [], []
        for loop_id, loop in enumerate(loops):
            points = np.asarray(loop, dtype=float)
            n = len(points)
            offset = len(vertices)
            vertices.extend(points.tolist())
            if segment_curves is not None:
                loop_curves = list(segment_curves[loop_id])
                if len(loop_curves) != n:
                    raise ValueError(
                        f'per-segment curves must have {n} entries, got {len(loop_curves)}')
                for i in range(n):
                    segments.append([offset + i, offset + (i + 1) % n])
                    loop_ids.append(loop_id)
                    out_curves.append(loop_curves[i])
            else:
                point_curves = _loop_point_curves(
                    curves[loop_id] if curves is not None else None, n)
                for i in range(n):
                    segments.append([offset + i, offset + (i + 1) % n])
                    loop_ids.append(loop_id)
                    a, b = point_curves[i], point_curves[(i + 1) % n]
                    # A segment curves only where both endpoints came from the same curve.
                    out_curves.append(a if (a is not None and a is b) else None)
        return cls(np.array(vertices), np.array(segments), np.array(loop_ids), out_curves)

    def loops(self):
        '''The vertices of each closed outline, in order.'''
        return [self.vertices[self.segments[self.loop_ids == loop_id, 0]]
                for loop_id in np.unique(self.loop_ids)]

    def area(self):
        '''Area of the region these loops enclose, which a mesh of them covers.

        Holes subtract, by the same even-odd rule meshing applies: a loop nested
        inside an odd number of others encloses nothing. Summing the loops instead
        would count a hole twice over: once as the plate it is cut from, once as
        itself.
        '''
        loops = self.loops()
        total = 0.0
        for i, loop in enumerate(loops):
            depth = sum(point_in_polygon(loop[0], other)
                        for j, other in enumerate(loops) if j != i)
            total += -calculate_polygon_area(loop) if depth % 2 else calculate_polygon_area(loop)
        return total

    def __repr__(self):
        return f'PSLG(vertices={self.vertices}, segments={self.segments})'

    def validate(self):
        '''Raise if these segments do not describe a planar straight-line graph.

        Segments may share endpoints and may not otherwise touch. Meshing an
        input that breaks this does not fail, it quietly produces a mesh of the
        wrong region, so it is worth refusing up front.
        '''
        vertices = np.asarray(self.vertices, dtype=float)
        segments = np.asarray(self.segments)

        starts, ends = vertices[segments[:, 0]], vertices[segments[:, 1]]
        degenerate = np.flatnonzero(np.all(starts == ends, axis=1))
        if len(degenerate):
            raise ValueError(
                f'segment {segments[degenerate[0]].tolist()} has zero length')

        _, first, counts = np.unique(vertices, axis=0, return_index=True, return_counts=True)
        if np.any(counts > 1):
            duplicate = vertices[first[counts > 1][0]]
            raise ValueError(f'vertex {duplicate.tolist()} appears more than once')

        crossing = _find_crossing_segments(vertices, segments)
        if crossing is not None:
            first_seg, second = crossing
            raise ValueError(
                f'segments {segments[first_seg].tolist()} and {segments[second].tolist()} '
                'cross away from a shared endpoint'
            )

    def add_bounding_box(self, buffer=0.2):
        '''Enclose the graph in a box, as its own loop.

        Under the even-odd rule this makes whatever was already here a hole, so
        an outline plus a box is a plate with that outline cut out of it.
        '''
        x_min, y_min = np.min(self.vertices, axis=0)
        x_max, y_max = np.max(self.vertices, axis=0)
        width = x_max - x_min
        height = y_max - y_min

        corner_vertices = [
            [x_min - buffer*width, y_min - buffer*height],
            [x_min - buffer*width, y_max + buffer*height],
            [x_max + buffer*width, y_max + buffer*height],
            [x_max + buffer*width, y_min - buffer*height]
        ]
        num_vertices = len(self.vertices)

        box_loop = int(self.loop_ids.max()) + 1 if len(self.loop_ids) else 0
        self.vertices = np.append(self.vertices, corner_vertices, axis=0)
        for i in range(4):
            self.segments = np.append(self.segments, [[num_vertices + i, num_vertices + (i + 1) % 4]], axis=0)
            self.loop_ids = np.append(self.loop_ids, box_loop)
            self.segment_curves.append(None)   # the box edges are straight

