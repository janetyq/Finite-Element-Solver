"""The pieces an `Outline` is drawn from, and the projection onto them.

A `Mesh` is straight-line: its boundary is a chain of chords. A `Curve` records the
true curve a chain of boundary facets was sampled from, so a curved (isoparametric)
element can place its boundary nodes on the curve instead of the chord, and refinement
can drop a new boundary vertex onto it rather than at the chord midpoint. The one
operation that needs is `project`: the nearest point on the curve to a given point.

A `Piece` is a `Curve` with two ends and a sampler: one stretch of an outline, straight
(`Line`) or curved (`Arc`, `CubicBezier`), or a whole closed loop (`Circle`). An
`Outline` joins pieces end to end and samples them into chords only when it is meshed.
"""
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.polynomial import polynomial as P

from fem.typing import FloatArray


@runtime_checkable
class Curve(Protocol):
    '''An analytic boundary curve, keyed by the one thing meshing needs: projection.'''

    def project(self, points: FloatArray) -> FloatArray:
        '''Nearest point on the curve to each of `points`: `(..., 2) -> (..., 2)`.'''
        ...

    def to_dict(self) -> dict:
        '''A JSON-ready description, inverted by `curve_from_dict`, so a boundary curve
        survives `Mesh.save` / `Mesh.load`.'''
        ...


@runtime_checkable
class Piece(Curve, Protocol):
    '''One stretch of an outline: a curve with a start, an end, and a sampler.

    `sample(n)` is `n + 1` points from `start` to `end` inclusive, the chords an
    `Outline` meshes it by; `length` sizes `n`. A `Circle` is the one closed piece: its
    start and end coincide and `sample(n)` returns `n` points without the repeat.
    '''
    closed: bool

    @property
    def start(self) -> FloatArray: ...

    @property
    def end(self) -> FloatArray: ...

    def length(self) -> float: ...

    def sample(self, n: int) -> FloatArray: ...


def _coincide(a: FloatArray, b: FloatArray) -> bool:
    '''Whether two points are the same to floating-point precision, relative to their
    magnitude: a traced outline can hold edges a million times shorter than its extent,
    which are still edges.'''
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return bool(np.linalg.norm(a - b) <= 1e-12 * scale)


class Line:
    '''The straight piece from `start` to `end`.

    An `Outline` samples a line as its two ends and gives the chord no curve: a straight
    boundary facet carries `None`, so nothing downstream distinguishes a drawn line from
    a chord of a polygon.
    '''
    closed = False

    def __init__(self, start: FloatArray | list[float], end: FloatArray | list[float]) -> None:
        self._start = np.asarray(start, dtype=float)
        self._end = np.asarray(end, dtype=float)
        if self._start.shape != (2,) or self._end.shape != (2,):
            raise ValueError('a line needs two 2D endpoints')
        if _coincide(self._start, self._end):
            raise ValueError(f'a line needs distinct endpoints, got {self._start.tolist()} twice')

    @property
    def start(self) -> FloatArray:
        return self._start

    @property
    def end(self) -> FloatArray:
        return self._end

    def length(self) -> float:
        return float(np.linalg.norm(self._end - self._start))

    def sample(self, n: int) -> FloatArray:
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        return self._start + t * (self._end - self._start)

    def project(self, points: FloatArray) -> FloatArray:
        p = np.asarray(points, dtype=float)
        d = self._end - self._start
        t = ((p - self._start) @ d) / float(d @ d)
        t = np.clip(t, 0.0, 1.0)[..., None]
        return self._start + t * d

    def to_dict(self) -> dict:
        return {'type': 'Line', 'start': self._start.tolist(), 'end': self._end.tolist()}

    @classmethod
    def _from_dict(cls, data: dict) -> 'Line':
        return cls(data['start'], data['end'])

    def __repr__(self) -> str:
        return f'Line({self._start.tolist()}, {self._end.tolist()})'


class Circle:
    '''A full circle of radius `radius` about `center`: the one closed piece, a loop
    of its own in an `Outline`.'''
    closed = True

    def __init__(self, center: FloatArray | list[float], radius: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError(f'circle radius must be positive, got {self.radius}')

    @property
    def start(self) -> FloatArray:
        return self.center + [self.radius, 0.0]

    end = start

    def length(self) -> float:
        return 2.0 * np.pi * self.radius

    def project(self, points: FloatArray) -> FloatArray:
        p = np.asarray(points, dtype=float)
        offset = p - self.center
        distance = np.linalg.norm(offset, axis=-1, keepdims=True)
        # A point exactly at the center has no nearest point; leave the radius to pick a
        # direction rather than dividing by zero. Callers project midpoints of boundary
        # chords, which never sit at the center.
        safe = np.where(distance == 0, 1.0, distance)
        return self.center + self.radius * offset / safe

    def sample(self, n: int) -> FloatArray:
        '''`n` points around the circle, `(n, 2)`, starting at angle 0, without the
        closing repeat: the chords an `Outline` meshes the circle by.'''
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

    def to_dict(self) -> dict:
        return {'type': 'Circle', 'center': self.center.tolist(), 'radius': self.radius}

    @classmethod
    def _from_dict(cls, data: dict) -> 'Circle':
        return cls(data['center'], data['radius'])

    def __repr__(self) -> str:
        return f'Circle(center={self.center.tolist()}, radius={self.radius})'


class Arc:
    '''An arc of a circle, spanning `[start_angle, end_angle]` in radians.

    A point is projected onto the circle, then clamped to the arc's angular span: a
    point past an endpoint snaps to the nearer endpoint rather than to the far side of
    the circle. As a piece it runs counter-clockwise from `start_angle`; `reversed()`
    is the same arc traversed the other way, for an outline drawn clockwise through it.
    '''
    closed = False

    def __init__(
        self, center: FloatArray | list[float], radius: float,
        start_angle: float, end_angle: float,
    ) -> None:
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError(f'arc radius must be positive, got {self.radius}')
        if not end_angle > start_angle:
            raise ValueError(
                f'arc needs end_angle > start_angle, got {start_angle} to {end_angle}'
            )
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self._reversed = False

    def reversed(self) -> 'Arc':
        '''This arc traversed from `end_angle` back to `start_angle`.'''
        arc = Arc(self.center, self.radius, self.start_angle, self.end_angle)
        arc._reversed = not self._reversed
        return arc

    def _point(self, angle: float) -> FloatArray:
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])

    @property
    def start(self) -> FloatArray:
        return self._point(self.end_angle if self._reversed else self.start_angle)

    @property
    def end(self) -> FloatArray:
        return self._point(self.start_angle if self._reversed else self.end_angle)

    def length(self) -> float:
        return self.radius * (self.end_angle - self.start_angle)

    def project(self, points: FloatArray) -> FloatArray:
        p = np.asarray(points, dtype=float)
        offset = p - self.center
        angle = np.arctan2(offset[..., 1], offset[..., 0])
        lo, hi = self.start_angle, self.end_angle
        # Wrap each angle into [lo, lo + 2pi) so the comparison against the span is
        # single-valued. Inside the span the angle is kept; outside, the point falls in
        # the gap (hi, lo + 2pi), and its nearer endpoint is hi below the gap's midpoint
        # and lo (i.e. lo + 2pi) above it.
        wrapped = lo + np.mod(angle - lo, 2 * np.pi)
        gap_mid = 0.5 * (hi + lo + 2 * np.pi)
        clamped = np.where(wrapped <= hi, wrapped, np.where(wrapped < gap_mid, hi, lo))
        return self.center + self.radius * np.stack(
            [np.cos(clamped), np.sin(clamped)], axis=-1)

    def sample(self, n: int) -> FloatArray:
        '''`n + 1` points from `start` to `end`, `(n + 1, 2)`, both endpoints included.'''
        angles = np.linspace(self.start_angle, self.end_angle, n + 1)
        if self._reversed:
            angles = angles[::-1]
        return self.center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

    def to_dict(self) -> dict:
        return {'type': 'Arc', 'center': self.center.tolist(), 'radius': self.radius,
                'start_angle': self.start_angle, 'end_angle': self.end_angle,
                'reversed': self._reversed}

    @classmethod
    def _from_dict(cls, data: dict) -> 'Arc':
        arc = cls(data['center'], data['radius'], data['start_angle'], data['end_angle'])
        return arc.reversed() if data['reversed'] else arc

    def __repr__(self) -> str:
        return (f'Arc(center={self.center.tolist()}, radius={self.radius}, '
                f'start_angle={self.start_angle}, end_angle={self.end_angle})'
                + ('.reversed()' if self._reversed else ''))


class CubicBezier:
    '''A cubic Bezier curve through control points P0..P3, `B(t)` for `t` in [0, 1].

    The piece a traced SVG path's `C` command becomes (`Outline.from_svg`): the outline
    keeps the control points, and meshing rounds the chords it samples to the true curve.
    '''
    closed = False

    def __init__(
        self, p0: FloatArray | list[float], p1: FloatArray | list[float],
        p2: FloatArray | list[float], p3: FloatArray | list[float],
    ) -> None:
        self.controls = np.asarray([p0, p1, p2, p3], dtype=float)   # (4, 2)
        if self.controls.shape != (4, 2):
            raise ValueError(f'a cubic Bezier needs four 2D control points, got {self.controls.shape}')
        c0, c1, c2, c3 = self.controls
        # Power-basis coefficients: B(t) = a[0] + a[1] t + a[2] t^2 + a[3] t^3.
        self._a = np.array([
            c0,
            3.0 * (c1 - c0),
            3.0 * (c0 - 2.0 * c1 + c2),
            c3 - 3.0 * c2 + 3.0 * c1 - c0,
        ])   # (4, 2), rows a0..a3

    def _eval(self, t: FloatArray) -> FloatArray:
        '''B(t) at parameters `t` (any shape) -> `(..., 2)`.'''
        t = np.asarray(t, dtype=float)
        powers = np.stack([np.ones_like(t), t, t**2, t**3], axis=-1)   # (..., 4)
        return powers @ self._a

    @property
    def start(self) -> FloatArray:
        return self.controls[0]

    @property
    def end(self) -> FloatArray:
        return self.controls[3]

    def length(self) -> float:
        '''The arc length, to the accuracy of a 64-chord sample: enough to size the
        chords an outline meshes the curve by.'''
        pts = self.sample(64)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def sample(self, n: int) -> FloatArray:
        '''`n + 1` points along the curve at evenly spaced parameters, ends included.'''
        return self._eval(np.linspace(0.0, 1.0, n + 1))

    def project(self, points: FloatArray) -> FloatArray:
        '''Nearest point on the curve to each of `points`: `(..., 2) -> (..., 2)`.

        The nearest parameter solves `(B(t) - q).B'(t) = 0`, a quintic in `t`; its real
        roots in [0, 1], together with the endpoints, are the candidates, and the closest
        is returned. Clamping to [0, 1] via the endpoints mirrors `Arc`: a point past an
        end snaps to the nearer end rather than to an extrapolation of the curve.
        '''
        p = np.asarray(points, dtype=float)
        flat = p.reshape(-1, 2)
        ax, ay = self._a[:, 0], self._a[:, 1]              # power coeffs, low -> high
        dax = np.array([ax[1], 2 * ax[2], 3 * ax[3]])      # B_x'(t) coeffs
        day = np.array([ay[1], 2 * ay[2], 3 * ay[3]])
        out = np.empty_like(flat)
        for i, q in enumerate(flat):
            cx, cy = ax.copy(), ay.copy()
            cx[0] -= q[0]
            cy[0] -= q[1]
            # d/dt |B - q|^2 / 2 = (B_x - q_x) B_x' + (B_y - q_y) B_y', degree 5.
            f = P.polytrim(P.polyadd(P.polymul(cx, dax), P.polymul(cy, day)), tol=1e-12)
            candidates = [0.0, 1.0]
            if len(f) >= 2:
                roots = P.polyroots(f)
                real = roots[np.abs(roots.imag) < 1e-9].real
                candidates.extend(real[(real >= 0.0) & (real <= 1.0)].tolist())
            pts = self._eval(np.array(candidates))          # (k, 2)
            out[i] = pts[np.argmin(np.sum((pts - q) ** 2, axis=1))]
        return out.reshape(p.shape)

    def to_dict(self) -> dict:
        return {'type': 'CubicBezier', 'controls': self.controls.tolist()}

    @classmethod
    def _from_dict(cls, data: dict) -> 'CubicBezier':
        p0, p1, p2, p3 = data['controls']
        return cls(p0, p1, p2, p3)

    def __repr__(self) -> str:
        return f'CubicBezier({self.controls.tolist()})'


# The curve types that persist through `fem.post.io`. A saved mesh stores each boundary
# curve as one of these tagged dicts; reconstruction dispatches on the name here rather
# than resolving an arbitrary attribute, so a hand-edited file can name only a known type.
_CURVE_BUILDERS: dict[str, Callable[[dict], Curve]] = {
    'Line': Line._from_dict,
    'Circle': Circle._from_dict,
    'Arc': Arc._from_dict,
    'CubicBezier': CubicBezier._from_dict,
}


def curve_to_dict(curve: Curve) -> dict:
    '''A JSON-ready description of `curve`, inverted by `curve_from_dict`.'''
    name = type(curve).__name__
    if name not in _CURVE_BUILDERS:
        raise ValueError(f'cannot serialize a boundary curve of type {name!r}')
    return curve.to_dict()


def curve_from_dict(data: dict) -> Curve:
    '''Rebuild a curve from `curve_to_dict`'s output, dispatching on its `type`.'''
    name = data.get('type')
    if name not in _CURVE_BUILDERS:
        raise ValueError(f'unknown boundary curve type {name!r}')
    return _CURVE_BUILDERS[name](data)
