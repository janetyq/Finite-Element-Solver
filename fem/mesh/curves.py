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

    def __repr__(self) -> str:
        return (f'Arc(center={self.center.tolist()}, radius={self.radius}, '
                f'start_angle={self.start_angle}, end_angle={self.end_angle})')
