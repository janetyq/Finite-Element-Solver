'''An incremental 2D Delaunay triangulation, grown a point at a time.

`IncrementalDelaunay` starts from a batch triangulation of its initial points and takes
one point at a time after that by Bowyer-Watson: find the triangles whose circumcircle
contains the point (the cavity), remove them, and fan the point to the cavity's rim.
Everything is keyed by vertex index in dicts, so an insertion costs the size of its
cavity, not of the mesh, and the caller learns which triangles it created.

The convex hull is closed with ghost triangles, one per hull edge, whose third corner is
`GHOST`. A ghost's "circumcircle" is the open half-plane beyond its hull edge, plus the
edge itself, so a point on or beyond the hull is absorbed by the same cavity rule as an
interior one and the hull stays convex without a special case.
'''
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from fem.typing import FloatArray, IntArray

GHOST = -1

# The orientation determinant is trusted only past its floating-point error bound
# (Shewchuk's filter for `orient2d`); within it three points are taken as collinear. A
# hull sampled along a straight line is the case that matters: a point on the line must
# split the hull edge it lies on and leave the ghosts of the collinear edges beside it
# alone, which a raw sign of round-off does not guarantee.
_ORIENT_ERROR_BOUND = 4 * np.finfo(float).eps

Triangle = tuple[int, int, int]


class IncrementalDelaunay:
    '''A Delaunay triangulation of the points given, with `insert` to add one more.

    `points` is the (n, 2) array of every point so far; a new one appends. Triangles
    are the real ones (no `GHOST` corner) unless said otherwise. `simplices` and
    `neighbors` present the current triangulation as arrays in `scipy.spatial.Delaunay`'s
    layout, rebuilt lazily after an insertion, for the whole-mesh scans; the per-insertion
    path never touches them.
    '''

    def __init__(self, points: FloatArray) -> None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f'points must be (n, 2), got {points.shape}')
        self._buffer = np.empty((max(2 * len(points), 16), 2))
        self._buffer[:len(points)] = points
        self.n_points = len(points)
        # The same coordinates as Python floats, for the predicates: a few scalar
        # multiplies per call are many times faster on floats than on array scalars.
        self._xy: list[tuple[float, float]] = [(float(x), float(y)) for x, y in points]

        self._triangles: dict[int, Triangle] = {}
        # Directed edge (a, b) -> the id of the triangle that walks a -> b
        # counter-clockwise; the neighbour across it is the triangle at (b, a).
        self._across: dict[tuple[int, int], int] = {}
        self._next_id = 0
        self._arrays: tuple[IntArray, IntArray] | None = None

        for a, b, c in Delaunay(points).simplices:
            a, b, c = int(a), int(b), int(c)
            if self._orient(a, b, c) < 0:
                b, c = c, b
            self._add((a, b, c))
        for (a, b), _ in list(self._across.items()):
            if (b, a) not in self._across:
                self._add((b, a, GHOST))

    # -- reading -------------------------------------------------------------------

    @property
    def points(self) -> FloatArray:
        return self._buffer[:self.n_points]

    def contains(self, triangle: Triangle) -> bool:
        '''Whether a triangle with these corners, in any order, currently exists.'''
        return self.find(triangle) is not None

    def triangle_on(self, a: int, b: int) -> int | None:
        '''The id of a real triangle with edge (a, b), or None if it is not an edge.'''
        for edge in ((a, b), (b, a)):
            idx = self._across.get(edge)
            if idx is not None and GHOST not in self._triangles[idx]:
                return idx
        return None

    def find(self, triangle: Triangle) -> int | None:
        '''The id of the triangle with these corners, in any order, or None.'''
        a, b, c = triangle
        for edge in ((a, b), (b, a)):
            idx = self._across.get(edge)
            if idx is not None and c in self._triangles[idx]:
                return idx
        return None

    @property
    def simplices(self) -> IntArray:
        '''(n_triangles, 3) corners of every real triangle, counter-clockwise.'''
        return self._as_arrays()[0]

    @property
    def neighbors(self) -> IntArray:
        '''(n_triangles, 3): `neighbors[i, j]` is the row of the triangle across the
        edge opposite corner j of row i, or -1 across a hull edge.'''
        return self._as_arrays()[1]

    def _as_arrays(self) -> tuple[IntArray, IntArray]:
        if self._arrays is None:
            ids = [idx for idx, t in self._triangles.items() if GHOST not in t]
            row_of = {idx: row for row, idx in enumerate(ids)}
            simplices = np.array([self._triangles[idx] for idx in ids], dtype=int).reshape(-1, 3)
            neighbors = np.full(simplices.shape, -1, dtype=int)
            for row, (a, b, c) in enumerate(simplices):
                for j, (u, v) in enumerate(((b, c), (c, a), (a, b))):
                    neighbors[row, j] = row_of.get(self._across[(int(v), int(u))], -1)
            self._arrays = simplices, neighbors
        return self._arrays

    # -- growing -------------------------------------------------------------------

    def insert(self, point: FloatArray, near: int | None = None) -> tuple[int, list[Triangle]]:
        '''Add `point`, returning its index and the real triangles the insertion created.

        `near` is the id of a triangle to start the walk to `point` from; the closer,
        the shorter the walk. None starts from an arbitrary triangle.
        '''
        idx = self._append(point)
        start = self._locate(idx, near)
        cavity = self._cavity(idx, start)
        rim = [(a, b) for t in cavity for a, b in self._edges(self._triangles[t])
               if self._across[(b, a)] not in cavity]
        for t in cavity:
            self._remove(t)
        # Each rim edge a -> b, counter-clockwise from the cavity's side, fans to the
        # triangle (a, b, p). A rim edge leaving a ghost carries `GHOST` as one end,
        # and the fan there is the ghost of a new hull edge, rotated to keep `GHOST` last.
        created: list[Triangle] = []
        for a, b in rim:
            if a == GHOST:
                self._add((b, idx, GHOST))
            elif b == GHOST:
                self._add((idx, a, GHOST))
            else:
                self._add((a, b, idx))
                created.append((a, b, idx))
        self._arrays = None
        return idx, created

    def _append(self, point: FloatArray) -> int:
        if self.n_points == len(self._buffer):
            grown = np.empty((2 * len(self._buffer), 2))
            grown[:self.n_points] = self._buffer[:self.n_points]
            self._buffer = grown
        self._buffer[self.n_points] = point
        self._xy.append((float(point[0]), float(point[1])))
        self.n_points += 1
        return self.n_points - 1

    def _locate(self, p: int, near: int | None) -> int:
        '''A triangle whose cavity `p` opens: the real one holding it, or the ghost
        beyond the hull edge it lies past. Walks from `near` across whichever edge
        `p` is outside of; a walk that cycles (collinear degeneracies) falls back to
        checking every triangle.'''
        t = near if near is not None and near in self._triangles else next(iter(self._triangles))
        for _ in range(len(self._triangles)):
            corners = self._triangles[t]
            if GHOST in corners:
                if self._in_cavity(corners, p):
                    return t
                break
            for a, b in self._edges(corners):
                if self._orient(a, b, p) < 0:
                    t = self._across[(b, a)]
                    break
            else:
                return t
        for idx, corners in self._triangles.items():
            if self._in_cavity(corners, p):
                return idx
        raise ValueError(f'point {self.points[p]} lies in no triangle')

    def _cavity(self, p: int, start: int) -> set[int]:
        '''The triangles `p` invalidates, grown outward from `start` across shared edges.

        A neighbour joins when its circumcircle contains `p`, and also when the rim
        edge it would leave behind faces away from `p`: the fan from `p` must be
        counter-clockwise, and round-off in the circle test can otherwise leave a
        rim edge `p` cannot see.
        '''
        cavity = {start}
        frontier = [start]
        while frontier:
            t = frontier.pop()
            for a, b in self._edges(self._triangles[t]):
                n = self._across[(b, a)]
                if n in cavity:
                    continue
                corners = self._triangles[n]
                faces_away = GHOST not in (a, b) and self._orient(a, b, p) <= 0
                if faces_away or self._in_cavity(corners, p):
                    cavity.add(n)
                    frontier.append(n)
        return cavity

    def _in_cavity(self, corners: Triangle, p: int) -> bool:
        '''Whether `p` invalidates the triangle: inside a real one's circumcircle, or
        beyond a ghost's hull edge, or on that edge strictly between its ends.'''
        a, b, c = corners
        if c == GHOST:
            side = self._orient(a, b, p)
            if side != 0:
                return side > 0
            (ax, ay), (bx, by), (px, py) = self._xy[a], self._xy[b], self._xy[p]
            return ((px - ax) * (bx - ax) + (py - ay) * (by - ay) > 0
                    and (px - bx) * (ax - bx) + (py - by) * (ay - by) > 0)
        return self._in_circle(a, b, c, p)

    def _orient(self, a: int, b: int, c: int) -> int:
        '''The turn a -> b -> c makes: 1 counter-clockwise, -1 clockwise, 0 collinear
        to within the determinant's round-off.'''
        (ax, ay), (bx, by), (cx, cy) = self._xy[a], self._xy[b], self._xy[c]
        left = (bx - ax) * (cy - ay)
        right = (by - ay) * (cx - ax)
        det = left - right
        if abs(det) <= _ORIENT_ERROR_BOUND * (abs(left) + abs(right)):
            return 0
        return 1 if det > 0 else -1

    def _in_circle(self, a: int, b: int, c: int, d: int) -> bool:
        '''Whether `d` lies strictly inside the circumcircle of counter-clockwise (a, b, c).'''
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = self._xy[a], self._xy[b], self._xy[c], self._xy[d]
        adx, ady = ax - dx, ay - dy
        bdx, bdy = bx - dx, by - dy
        cdx, cdy = cx - dx, cy - dy
        det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
               + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
               + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
        return det > 0

    # -- the dicts -----------------------------------------------------------------

    @staticmethod
    def _edges(corners: Triangle) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        a, b, c = corners
        return (a, b), (b, c), (c, a)

    def _add(self, corners: Triangle) -> int:
        idx = self._next_id
        self._next_id += 1
        self._triangles[idx] = corners
        for edge in self._edges(corners):
            self._across[edge] = idx
        return idx

    def _remove(self, idx: int) -> None:
        for edge in self._edges(self._triangles.pop(idx)):
            del self._across[edge]
