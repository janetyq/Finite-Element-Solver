"""Analytic boundary curves a mesh follows, and the projection onto them.

A `Mesh` is straight-line: its boundary is a chain of chords. A `Curve` records the
true curve a chain of boundary facets was sampled from, so a curved (isoparametric)
element can place its boundary nodes on the curve instead of the chord, and refinement
can drop a new boundary vertex onto it rather than at the chord midpoint.

The one operation everything needs is `project`: the nearest point on the curve to a
given point. Placing a P2 edge-midpoint node and splitting a boundary segment during
refinement are both a projection of the straight midpoint onto the curve.
"""
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.polynomial import polynomial as P

from fem.typing import FloatArray


@runtime_checkable
class Curve(Protocol):
    '''An analytic boundary curve, keyed by the one thing meshing needs: projection.'''

    def project(self, points: FloatArray) -> FloatArray:
        '''Nearest point on the curve to each of `points`: `(..., 2) -> (..., 2)`.'''
        ...


class Circle:
    '''A full circle of radius `radius` about `center`.'''

    def __init__(self, center: FloatArray | list[float], radius: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError(f'circle radius must be positive, got {self.radius}')

    def project(self, points: FloatArray) -> FloatArray:
        p = np.asarray(points, dtype=float)
        offset = p - self.center
        distance = np.linalg.norm(offset, axis=-1, keepdims=True)
        # A point exactly at the center has no nearest point; leave the radius to pick a
        # direction rather than dividing by zero. Callers project midpoints of boundary
        # chords, which never sit at the center.
        safe = np.where(distance == 0, 1.0, distance)
        return self.center + self.radius * offset / safe

    def polygon(self, n: int) -> FloatArray:
        '''`n` points around the circle, `(n, 2)`, starting at angle 0, without the
        closing repeat: the loop a `PSLG` samples the circle by.'''
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return self.center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

    def __repr__(self) -> str:
        return f'Circle(center={self.center.tolist()}, radius={self.radius})'


class Arc:
    '''An arc of a circle, spanning `[start_angle, end_angle]` in radians.

    A point is projected onto the circle, then clamped to the arc's angular span: a
    point past an endpoint snaps to the nearer endpoint rather than to the far side of
    the circle.
    '''

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

    def polygon(self, n: int) -> FloatArray:
        '''`n` points along the arc, `(n, 2)`, both endpoints included.'''
        angles = np.linspace(self.start_angle, self.end_angle, n)
        return self.center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

    def __repr__(self) -> str:
        return (f'Arc(center={self.center.tolist()}, radius={self.radius}, '
                f'start_angle={self.start_angle}, end_angle={self.end_angle})')


class CubicBezier:
    '''A cubic Bezier curve through control points P0..P3, `B(t)` for `t` in [0, 1].

    The boundary curve a traced SVG path carries: `read_svg_to_pslg` keeps a path's
    cubic control points and tags the segments sampled from it with one of these, so
    meshing rounds the outline to the true curve instead of the sampled chords.
    '''

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

    def sample(self, n: int) -> FloatArray:
        '''`n` points along the curve at evenly spaced parameters, endpoints included.'''
        return self._eval(np.linspace(0.0, 1.0, n))

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

    def __repr__(self) -> str:
        return f'CubicBezier({self.controls.tolist()})'
