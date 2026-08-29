"""A planar straight-line graph: the chords a 2D mesh is built to respect.

A `PSLG` is what `Outline.sample` produces and `RuppertsAlgorithm` consumes: vertices,
the segments between them, which loop each came from, and the `Curve` (if any) each
segment is a chord of. Describe a domain as an `Outline` rather than building one of
these; `pslg.mesh(...)` runs Ruppert's Delaunay refinement on it. The polygon helpers
meshing needs (area, point-in-polygon, segment crossing) live here beside it.

`PSLG.mesh` imports `fem.mesh.ruppert` lazily: the mesher consumes a `PSLG`, so the edge
points up and stays function-local.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fem.mesh.curves import Curve
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


@dataclass(frozen=True, eq=False)
class PSLG:
    '''A planar straight-line graph: vertices, plus the segments a mesh must respect.

    `loop_ids` says which closed outline each segment came from, so a caller can
    tell an obstacle's boundary from an enclosing box's after meshing: the mesh's
    `boundary_tags` carry them. All zeros unless given.

    `segment_curves` is the analytic `Curve` (or `None`) each segment is a chord of,
    aligned with `segments`. A segment on a curve has its split points projected onto it
    during meshing, and the curve is carried onto the output mesh's matching boundary
    facet, so refinement rounds the outline instead of only subdividing its chords. All
    `None` for a straight-line outline, the default.

    Immutable.
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
