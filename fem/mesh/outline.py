"""An `Outline`: closed loops of pieces, sampled into chords only when meshed.

A 2D domain is drawn as loops of `Piece`s (`Line`, `Arc`, `CubicBezier`, or a lone
`Circle`) joined end to end. Nothing here is sampled: `sample(resolution)` turns the
loops into the `PSLG` (a straight-line graph of chords) that Ruppert's algorithm
refines, and `mesh(...)` does both. Each chord of a curved piece carries the piece as
its `Curve`, so the mesher, red-green refinement, and a curved element all project onto
the true shape; `resolution` is only the coarsest sampling they start from.

`from_polygons` draws straight loops, `from_svg` reads a traced drawing, and
`simplified(tolerance)` is Douglas-Peucker over the straight runs of a loop, leaving
the curved pieces alone. What each loop means is decided when meshing, by the even-odd
rule: a loop inside another is a hole, a loop beside it a separate piece of the domain.

`Outline.from_svg` imports `fem.mesh.svg` lazily: the reader produces an outline, so
the edge points up and stays function-local.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fem.mesh.curves import Curve, Line, Piece
from fem.mesh.pslg import PSLG
from fem.typing import FloatArray

if TYPE_CHECKING:
    from fem.mesh.mesh import Mesh

__all__ = ['Loop', 'Outline', 'douglas_peucker']

Loop = tuple[Piece, ...]

# The chords a piece is sampled into at least, whatever the resolution: a line is never
# subdivided (Ruppert splits it where the mesh needs it), a curve gets enough to read as
# one, and a circle enough to be a polygon at all.
_MIN_CHORDS = {'line': 1, 'curve': 2, 'circle': 8}


@dataclass(frozen=True)
class Outline:
    '''Closed loops of pieces; see the module docstring.'''
    loops: tuple[Loop, ...]

    def __init__(self, loops: Iterable[Iterable[Piece] | Piece]) -> None:
        normalized = tuple(
            (loop,) if isinstance(loop, Piece) else tuple(loop) for loop in loops)
        if not normalized:
            raise ValueError('an outline needs at least one loop')
        object.__setattr__(self, 'loops', normalized)
        self._validate()

    def _validate(self) -> None:
        tol = 1e-9 * max(self.extent, 1e-300)
        for k, loop in enumerate(self.loops):
            if not loop:
                raise ValueError(f'loop {k} has no pieces')
            if any(piece.closed for piece in loop):
                if len(loop) > 1:
                    raise ValueError(f'loop {k}: a closed piece (a Circle) is a loop of its own')
                continue
            for i, piece in enumerate(loop):
                following = loop[(i + 1) % len(loop)]
                if not np.allclose(piece.end, following.start, atol=tol, rtol=0.0):
                    raise ValueError(
                        f'loop {k}: piece {i} ends at {piece.end.tolist()} but piece '
                        f'{(i + 1) % len(loop)} starts at {following.start.tolist()}'
                    )

    # -- building ----------------------------------------------------------------------

    @classmethod
    def from_polygons(cls, polygons: Iterable[FloatArray | Sequence[Sequence[float]]]) -> Outline:
        '''Straight loops, one per polygon: `(n, 2)` points without a closing repeat,
        each becoming `n` `Line`s.'''
        loops = []
        for polygon in polygons:
            points = np.asarray(polygon, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
                raise ValueError(f'a polygon needs at least three 2D points, got {points.shape}')
            loops.append(tuple(Line(points[i], points[(i + 1) % len(points)])
                               for i in range(len(points))))
        return cls(loops)

    @classmethod
    def from_svg(cls, svg_file: str, ) -> Outline:
        '''The closed paths of an SVG file, `L` commands as `Line`s and `C` commands as
        `CubicBezier`s, in a y-up frame. Simplify a dense trace with `simplified`.'''
        from fem.mesh.svg import read_svg_outline
        return read_svg_outline(svg_file)

    def with_bounding_box(self, buffer: float = 0.2) -> Outline:
        '''This outline enclosed in a box, as a new loop.

        Under the even-odd rule this makes whatever was already here a hole, so an
        outline plus a box is a plate with that outline cut out of it. `buffer` is the
        margin on each side as a fraction of the outline's extent.
        '''
        lower, upper = self._bounds()
        width, height = upper - lower
        x0, y0 = lower[0] - buffer * width, lower[1] - buffer * height
        x1, y1 = upper[0] + buffer * width, upper[1] + buffer * height
        corners = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        box = tuple(Line(corners[i], corners[(i + 1) % 4]) for i in range(4))
        return Outline(self.loops + (box,))

    def simplified(self, tolerance: float) -> Outline:
        '''Douglas-Peucker over each run of consecutive `Line`s, dropping vertices within
        `tolerance` times the loop's extent of the kept chords. A run's first and last
        points are kept, so its joins to curved pieces hold; curved pieces are untouched.
        A loop that is all lines is the open polyline from its first vertex.
        '''
        loops = []
        for loop in self.loops:
            if any(piece.closed for piece in loop):
                loops.append(loop)
                continue
            epsilon = tolerance * _extent(_loop_points(loop))
            loops.append(_simplify_loop(loop, epsilon))
        return Outline(loops)

    # -- sampling and meshing ----------------------------------------------------------

    def sample(self, resolution: float = 0.02) -> PSLG:
        '''The straight-line graph of this outline: every piece sampled into chords no
        longer than `resolution` times the outline's extent (a line is one chord), the
        chords of a curved piece carrying it as their `Curve`, `loop_ids` the loop index.
        '''
        if resolution <= 0:
            raise ValueError(f'resolution must be positive, got {resolution}')
        target = resolution * self.extent
        vertices: list[FloatArray] = []
        segments: list[list[int]] = []
        loop_ids: list[int] = []
        curves: list[Curve | None] = []
        for k, loop in enumerate(self.loops):
            first = len(vertices)
            for piece in loop:
                n = self._chords(piece, target)
                points = piece.sample(n)
                if not piece.closed:
                    points = points[:-1]      # the end is the next piece's start
                start = len(vertices)
                vertices.extend(points)
                curve = None if isinstance(piece, Line) else piece
                for i in range(len(points)):
                    segments.append([start + i, start + i + 1])
                    curves.append(curve)
                    loop_ids.append(k)
            segments[-1][1] = first           # the loop closes on its first vertex
        return PSLG(np.array(vertices), np.array(segments), np.array(loop_ids), tuple(curves))

    @staticmethod
    def _chords(piece: Piece, target: float) -> int:
        if isinstance(piece, Line):
            return _MIN_CHORDS['line']
        least = _MIN_CHORDS['circle'] if piece.closed else _MIN_CHORDS['curve']
        return max(least, math.ceil(piece.length() / target))

    def mesh(self, min_angle: float = 30, max_area: float | None = None,
             max_area_fraction: float | None = None, resolution: float = 0.02) -> Mesh:
        '''Sample at `resolution` and triangulate by Ruppert's algorithm; see `PSLG.mesh`
        for the angle and area bounds. The mesh's boundary facets carry the loop each
        came from (`boundary_tags`, for `on_tag`) and the piece (`boundary_curves`).'''
        return self.sample(resolution).mesh(min_angle=min_angle, max_area=max_area,
                                            max_area_fraction=max_area_fraction)

    def area(self, resolution: float = 0.02) -> float:
        '''Area of the sampled outline, holes subtracted by the even-odd rule.'''
        return self.sample(resolution).area()

    # -- queries -----------------------------------------------------------------------

    @property
    def extent(self) -> float:
        '''The longer side of the bounding box of every piece.'''
        lower, upper = self._bounds()
        return float(np.max(upper - lower))

    def _bounds(self) -> tuple[FloatArray, FloatArray]:
        points = np.concatenate([_loop_points(loop) for loop in self.loops])
        return points.min(axis=0), points.max(axis=0)

    @property
    def n_pieces(self) -> int:
        return sum(len(loop) for loop in self.loops)

    def __repr__(self) -> str:
        return f'Outline({len(self.loops)} loops, {self.n_pieces} pieces)'


def _loop_points(loop: Loop) -> FloatArray:
    '''A coarse sampling of a loop, for its extent.'''
    return np.concatenate([piece.sample(8) for piece in loop])


def _extent(points: FloatArray) -> float:
    return float(np.max(np.max(points, axis=0) - np.min(points, axis=0)))


def _simplify_loop(loop: Loop, epsilon: float) -> Loop:
    '''Douglas-Peucker on each maximal run of `Line`s in `loop`.'''
    n = len(loop)
    if all(isinstance(piece, Line) for piece in loop):
        points = np.array([piece.start for piece in loop])
        keep = _simplify_indices(points, epsilon)
        kept = points[keep]
        if len(kept) < 3:
            return loop
        return tuple(Line(kept[i], kept[(i + 1) % len(kept)]) for i in range(len(kept)))

    # Rotate so the loop starts on a curved piece; every run of lines is then interior.
    first_curved = next(i for i, piece in enumerate(loop) if not isinstance(piece, Line))
    rotated = loop[first_curved:] + loop[:first_curved]
    out: list[Piece] = []
    i = 0
    while i < n:
        piece = rotated[i]
        if not isinstance(piece, Line):
            out.append(piece)
            i += 1
            continue
        j = i
        while j < n and isinstance(rotated[j], Line):
            j += 1
        run = rotated[i:j]
        points = np.array([line.start for line in run] + [run[-1].end])
        kept = points[_simplify_indices(points, epsilon)]
        out.extend(Line(kept[k], kept[k + 1]) for k in range(len(kept) - 1))
        i = j
    return tuple(out)


def douglas_peucker(points: FloatArray, epsilon: float) -> FloatArray:
    '''Simplify a polyline, dropping points within `epsilon` of the kept chords.

    Returns an Nx2 array (n < N) in the input order, endpoints kept. Keeps the point
    furthest from the chord between the current endpoints and recurses on each side
    until every dropped point lies within `epsilon`. A thin wrapper over
    `_simplify_indices`, which returns the surviving indices instead.
    '''
    points = np.asarray(points, dtype=float)
    return points[_simplify_indices(points, epsilon)]


def _simplify_indices(points: FloatArray, epsilon: float) -> list[int]:
    '''Douglas-Peucker on an open polyline, returning the sorted indices it keeps.

    The endpoints are always kept. Iterative, over an explicit stack rather than
    recursion, so a densely sampled outline cannot overflow the interpreter's recursion
    limit however deep the splits go. Each split's furthest-point search is vectorised
    over the span it examines.
    '''
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= 2:
        return list(range(n))

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        start, end = points[lo], points[hi]
        interior = points[lo + 1:hi]
        edge = end - start
        length = float(np.hypot(edge[0], edge[1]))
        if length == 0.0:
            # A zero-length chord (start == end): fall back to distance from start, so
            # a closed sub-span still simplifies.
            dists = np.hypot(interior[:, 0] - start[0], interior[:, 1] - start[1])
        else:
            dists = np.abs(edge[0] * (start[1] - interior[:, 1])
                           - edge[1] * (start[0] - interior[:, 0])) / length
        k = int(np.argmax(dists))
        # Split only on a point strictly off the chord and at least epsilon away, so a
        # collinear run (every distance zero) keeps just its endpoints, even at
        # epsilon 0, where dividing the recursion at it would loop.
        if dists[k] > 0.0 and dists[k] >= epsilon:
            furthest = lo + 1 + k
            keep[furthest] = True
            stack.append((lo, furthest))
            stack.append((furthest, hi))
    return np.flatnonzero(keep).tolist()
