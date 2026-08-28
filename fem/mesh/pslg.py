"""A planar straight-line graph: the outline a 2D mesh is built to respect.

A `PSLG` is drawn by hand from loops (`PSLG.from_loops`, `PSLG.circle`), or read from an
SVG (`fem.mesh.svg.read_svg_to_pslg`), and meshed with `pslg.mesh(...)`, which runs
Ruppert's Delaunay refinement. The polygon helpers it needs (area, point-in-polygon,
segment crossing) live here beside it.

`PSLG.mesh` imports `fem.mesh.ruppert` lazily: the mesher consumes a `PSLG`, so the edge
points up and stays function-local.
"""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fem.mesh.curves import Circle, Curve
from fem.mesh.mesh import frozen_array
from fem.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh


def polygon_area(polygon: FloatArray) -> float:
    '''Area of a simple 2D polygon (shoelace), or of a triangle in 3D.'''
    polygon = np.asarray(polygon, dtype=float)
    if polygon.shape[1] == 2:
        x, y = polygon.T
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if polygon.shape[1] == 3 and len(polygon) == 3:
        a, b = polygon[1] - polygon[0], polygon[2] - polygon[0]
        return 0.5 * float(np.linalg.norm(np.cross(a, b)))
    # A general planar polygon in 3D needs Newell's method to recover the normal;
    # nothing asks for one yet, so refuse rather than return a wrong number.
    raise NotImplementedError(
        f'polygon area is defined for 2D polygons and 3D triangles, '
        f'got {len(polygon)} points in {polygon.shape[1]}D'
    )


def point_in_polygon(point: FloatArray, polygon: FloatArray) -> bool:
    '''Whether `point` is inside the 2D `polygon`, by the even-odd (ray-crossing) rule.'''
    x, y = point
    x_coords, y_coords = np.asarray(polygon, dtype=float).T
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[(i+1) % n], y_coords[(i+1) % n]
        if y1 < y <= y2 or y2 < y <= y1:
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) < x:
                inside = not inside
    return inside


def _candidate_segment_pairs(lo, hi):
    '''Segment index pairs `(i, j)` with `i < j` whose bounding boxes may overlap.

    A uniform grid keyed on a segment's bounding-box cells: two segments are a
    candidate only where they share a cell. Two crossing segments have overlapping
    boxes and so always share a cell, meaning no crossing is dropped, so the grid is a
    speed knob only. The cell size is the larger of twice the median segment box (each
    short segment then lands in about one cell) and the extent over sqrt(n) (so one
    long segment cannot cover unboundedly many cells).
    '''
    n = len(lo)
    diag = np.hypot(hi[:, 0] - lo[:, 0], hi[:, 1] - lo[:, 1])
    nonzero = diag[diag > 0]
    extent = float(np.max(hi.max(axis=0) - lo.min(axis=0)))
    cell = max(2.0 * float(np.median(nonzero)) if len(nonzero) else 0.0,
               extent / max(1.0, np.sqrt(n)))
    if cell <= 0.0:
        cell = 1.0   # every box is a point at one location; one cell holds them all

    origin = lo.min(axis=0)
    cell_lo = np.floor((lo - origin) / cell).astype(np.int64)
    cell_hi = np.floor((hi - origin) / cell).astype(np.int64)

    buckets = defaultdict(list)
    for i in range(n):
        for cx in range(cell_lo[i, 0], cell_hi[i, 0] + 1):
            for cy in range(cell_lo[i, 1], cell_hi[i, 1] + 1):
                buckets[(cx, cy)].append(i)

    pairs = set()
    for members in buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.add((members[a], members[b]))   # members are ascending, so i < j
    return pairs


def _find_crossing_segments(vertices, segments):
    '''The lexicographically first pair of segments that properly cross, or None.

    A proper crossing puts each segment's endpoints strictly on opposite sides of the
    other; a pair sharing an endpoint touches there and is skipped. Only pairs a
    spatial grid finds sharing a bounding-box cell are tested (`_candidate_segment_pairs`),
    so a spread-out outline costs about linear time rather than comparing every pair.
    '''
    vertices = np.asarray(vertices, dtype=float)
    segments = np.asarray(segments)
    if len(segments) < 2:
        return None

    starts, ends = vertices[segments[:, 0]], vertices[segments[:, 1]]
    lo = np.minimum(starts, ends)
    hi = np.maximum(starts, ends)

    candidates = _candidate_segment_pairs(lo, hi)
    if not candidates:
        return None
    pairs = np.array(sorted(candidates))
    i, j = pairs[:, 0], pairs[:, 1]

    def side_of(line_start, line_end, point):
        '''Sign of which side of a directed line each point falls on.'''
        return ((line_end[:, 0] - line_start[:, 0]) * (point[:, 1] - line_start[:, 1])
                - (line_end[:, 1] - line_start[:, 1]) * (point[:, 0] - line_start[:, 0]))

    # A proper crossing puts each segment's endpoints on opposite sides of the other.
    straddles_i = ((side_of(starts[i], ends[i], starts[j]) > 0)
                   != (side_of(starts[i], ends[i], ends[j]) > 0))
    straddles_j = ((side_of(starts[j], ends[j], starts[i]) > 0)
                   != (side_of(starts[j], ends[j], ends[i]) > 0))
    shares_endpoint = (segments[i][:, :, None] == segments[j][:, None, :]).any(axis=(1, 2))
    crossing = straddles_i & straddles_j & ~shares_endpoint

    hits = np.flatnonzero(crossing)
    if not len(hits):
        return None
    # `pairs` is lexicographically sorted, so the first crossing is the lex-min pair.
    first = hits[0]
    return int(i[first]), int(j[first])


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


@dataclass(frozen=True, eq=False)
class PSLG:
    '''A planar straight-line graph: vertices, plus the segments a mesh must respect.

    `loop_ids` says which closed outline each segment came from, so a caller can
    tell an obstacle's boundary from an enclosing box's after meshing: the mesh's
    `boundary_tags` carry them. All zeros unless the graph was built by `from_loops`.

    `segment_curves` is the analytic `Curve` (or `None`) each segment samples, aligned
    with `segments`. A segment on a curve has its split points projected onto it during
    meshing, and the curve is carried onto the output mesh's matching boundary facet, so
    refinement rounds the outline instead of only subdividing its chords. All `None` for
    a straight-line outline, the default. Per segment rather than per loop, so an outline
    that is part straight and part arc (a filleted corner) can curve only its arc.

    Immutable: `with_bounding_box` returns a new graph.
    '''
    vertices: FloatArray
    segments: IntArray
    loop_ids: IntArray
    segment_curves: tuple[Curve | None, ...]

    def __init__(self, vertices, segments=None, loop_ids=None, segment_curves=None):
        vertices = frozen_array(np.array(vertices, dtype=float))
        if segments is None:
            n = len(vertices)
            segments = np.array([[i, (i + 1) % n] for i in range(n)])
        segments = frozen_array(np.array(segments, dtype=int).reshape(-1, 2))
        loop_ids = (np.zeros(len(segments), dtype=int) if loop_ids is None
                    else np.array(loop_ids, dtype=int))
        curves = (tuple(segment_curves) if segment_curves is not None
                  else (None,) * len(segments))
        if len(loop_ids) != len(segments) or len(curves) != len(segments):
            raise ValueError('loop_ids and segment_curves must have one entry per segment')
        object.__setattr__(self, 'vertices', vertices)
        object.__setattr__(self, 'segments', segments)
        object.__setattr__(self, 'loop_ids', frozen_array(loop_ids))
        object.__setattr__(self, 'segment_curves', curves)

    # -- building ----------------------------------------------------------------------

    @classmethod
    def from_loops(cls, loops, curves=None, segment_curves=None) -> 'PSLG':
        '''A PSLG spanning several closed outlines.

        What each loop means is decided when meshing, by the even-odd rule: a
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

    @classmethod
    def circle(cls, center: Sequence[float], radius: float, segments: int) -> 'PSLG':
        '''A disk: one loop polygonising the circle, carrying it as its `Curve`, so
        meshing and refinement round it however coarse `segments` is.'''
        circle = Circle(list(center), radius)
        return cls.from_loops([circle.polygon(segments)], curves=[circle])

    def with_bounding_box(self, buffer: float = 0.2) -> 'PSLG':
        '''This graph enclosed in a box, as a new loop.

        Under the even-odd rule this makes whatever was already here a hole, so
        an outline plus a box is a plate with that outline cut out of it. `buffer` is
        the margin on each side as a fraction of the graph's extent.
        '''
        x_min, y_min = np.min(self.vertices, axis=0)
        x_max, y_max = np.max(self.vertices, axis=0)
        width, height = x_max - x_min, y_max - y_min
        corners = [
            [x_min - buffer*width, y_min - buffer*height],
            [x_min - buffer*width, y_max + buffer*height],
            [x_max + buffer*width, y_max + buffer*height],
            [x_max + buffer*width, y_min - buffer*height],
        ]
        n = len(self.vertices)
        box_loop = int(self.loop_ids.max()) + 1 if len(self.loop_ids) else 0
        box_segments = [[n + i, n + (i + 1) % 4] for i in range(4)]
        return PSLG(
            np.append(self.vertices, corners, axis=0),
            np.append(self.segments, box_segments, axis=0),
            np.append(self.loop_ids, [box_loop] * 4),
            self.segment_curves + (None,) * 4,     # the box edges are straight
        )

    # -- queries -----------------------------------------------------------------------

    def loops(self) -> list[FloatArray]:
        '''The vertices of each closed outline, in order.'''
        return [self.vertices[self.segments[self.loop_ids == loop_id, 0]]
                for loop_id in np.unique(self.loop_ids)]

    def area(self) -> float:
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
            total += -polygon_area(loop) if depth % 2 else polygon_area(loop)
        return total

    def validate(self) -> None:
        '''Raise if these segments do not describe a planar straight-line graph.

        Segments may share endpoints and may not otherwise touch. Meshing an
        input that breaks this does not fail, it quietly produces a mesh of the
        wrong region, so it is worth refusing up front.
        '''
        vertices, segments = self.vertices, self.segments

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

    # -- meshing -----------------------------------------------------------------------

    def mesh(self, min_angle: float = 30, max_area: float | None = None,
             max_area_fraction: float | None = None) -> 'Mesh':
        '''Triangulate by Ruppert's algorithm to a minimum-angle bound and, optionally,
        a maximum triangle area, absolute (`max_area`) or as a fraction of the
        enclosed area (`max_area_fraction`); give at most one.

        Validates first. The mesh's `boundary_tags` name the loop each boundary facet
        came from, and its `boundary_curves` the curve, so the hole in a plate can be
        addressed by tag and stays round under refinement. Use `RuppertsAlgorithm`
        directly to keep the refinement state (the split segments, say).
        '''
        from fem.mesh.ruppert import RuppertsAlgorithm
        if max_area is not None and max_area_fraction is not None:
            raise ValueError('give max_area or max_area_fraction, not both')
        if max_area_fraction is not None:
            max_area = max_area_fraction * self.area()
        self.validate()
        return RuppertsAlgorithm(self, min_angle=min_angle, max_area=max_area).refine()

    def __repr__(self) -> str:
        return (f'PSLG({len(self.vertices)} vertices, {len(self.segments)} segments, '
                f'{len(np.unique(self.loop_ids))} loops)')
