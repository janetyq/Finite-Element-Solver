"""Position-based regions and fields: specifications written against coordinates
rather than vertex indices.

A region is any callable mapping an (N, spatial_dim) array of point coordinates to an
(N,) boolean mask, so a bare lambda qualifies. The helpers below name the common
cases and own the coordinate tolerance.

A field is a constant or a callable of position; `evaluate_field` normalizes both
into a (N, n_components) array. A `TimeDependent` field is a callable of position and
time; `at(t)` turns it into a field of position, which is what every consumer takes.

Both are mesh-independent, so a boundary condition described this way can be
resolved again against whatever mesh is current and survives refinement.
`at_indices` is the escape hatch for node-specific work; it marks itself mesh-bound
so remeshers can refuse it.
"""
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from fem.typing import BoolArray, FieldValue, FloatArray, IntArray, Point, Region, Vertices

# Coordinates come from linspace/midpoint arithmetic, so exact boundary values
# are representable; this only absorbs round-off, not genuine mesh spacing.
DEFAULT_ATOL: float = 1e-9


def everywhere() -> Region:
    '''Every point. Combined with the boundary-only resolution, this means
    "the entire boundary", the most common Dirichlet region.'''
    return lambda points: np.ones(len(points), dtype=bool)


def on_plane(axis: int, value: float, atol: float = DEFAULT_ATOL) -> Region:
    '''Points whose `axis` coordinate equals `value` (e.g. the left edge is
    on_plane(0, 0.0)).'''
    return lambda points: np.abs(points[:, axis] - value) <= atol


def in_box(
    lower: Sequence[float | None],
    upper: Sequence[float | None],
    atol: float = DEFAULT_ATOL,
) -> Region:
    '''Points inside an axis-aligned box, inclusive. A `None` bound on either
    side leaves that direction unbounded, so a band in y is
    in_box([None, 0.2], [None, 0.8]).'''
    def region(points: Vertices) -> BoolArray:
        mask = np.ones(len(points), dtype=bool)
        for axis, bound in enumerate(lower):
            if bound is not None:
                mask &= points[:, axis] >= bound - atol
        for axis, bound in enumerate(upper):
            if bound is not None:
                mask &= points[:, axis] <= bound + atol
        return mask
    return region


def intersect(*regions: Region) -> Region:
    '''Points in every one of `regions`.'''
    def region(points: Vertices) -> BoolArray:
        mask = np.ones(len(points), dtype=bool)
        for r in regions:
            mask &= r(points)
        return mask
    return _propagate_mesh_bound(region, regions)


def union(*regions: Region) -> Region:
    '''Points in any of `regions`.'''
    def region(points: Vertices) -> BoolArray:
        mask = np.zeros(len(points), dtype=bool)
        for r in regions:
            mask |= r(points)
        return mask
    return _propagate_mesh_bound(region, regions)


class at_indices:  # noqa: N801 - lowercase to read like the function helpers above
    '''Named vertex indices. The escape hatch for work that is about specific nodes
    rather than a place in the domain.

    Mesh-bound by construction: indices mean nothing once a remesher renumbers
    vertices, so `is_mesh_bound` reports True and callers that remesh refuse it.
    '''
    mesh_bound = True

    def __init__(self, indices: Sequence[int] | IntArray) -> None:
        self.indices = np.asarray(indices, dtype=int)

    def __call__(self, points: Vertices) -> BoolArray:
        mask = np.zeros(len(points), dtype=bool)
        mask[self.indices] = True
        return mask


class on_tag:  # noqa: N801 - lowercase to read like the function helpers above
    '''The boundary facets tagged `tag`: an outline of the `PSLG` the mesh was built
    from (the hole in a plate is `on_tag(1)`), or a physical group of an imported mesh.

    Unlike a geometric region it is resolved from the facets, not the coordinates, so
    it needs a node geometry carrying `boundary_tags`; `select_nodes` is what
    `Condition.select` calls. It is not mesh-bound: refinement carries tags onto the
    split facets, so the same condition resolves on every mesh of the hierarchy.
    '''

    def __init__(self, tag: int) -> None:
        self.tag = int(tag)

    def select_nodes(self, boundary: IntArray, boundary_tags: IntArray | None) -> IntArray:
        '''The nodes of every facet tagged `tag`, ascending.'''
        if boundary_tags is None:
            raise ValueError(
                f'on_tag({self.tag}) needs a mesh with boundary_tags; a PSLG mesh has '
                'them, a structured mesh does not (name its faces by coordinates)')
        return np.unique(np.asarray(boundary)[np.asarray(boundary_tags) == self.tag])

    def __call__(self, points: Vertices) -> BoolArray:
        raise TypeError(
            f'on_tag({self.tag}) selects boundary facets, not points, and cannot be '
            'combined with a geometric region through intersect or union')


class TimeDependent:
    '''A field that varies in time: `fn(p, t)` is the value at point `p` and time `t`.

    A source, a traction, a Robin `g`, or a Dirichlet value may be one; the
    integrators evaluate it at each step through `Problem.load_at` and
    `Problem.constraints_at`. `at(t)` fixes the time and returns the plain field of
    position every other consumer takes.
    '''

    def __init__(self, fn: Callable[[Point, float], Any]) -> None:
        self.fn = fn

    def at(self, t: float) -> Callable[[Point], Any]:
        fn = self.fn
        return lambda p: fn(p, t)


def field_at(value: FieldValue, t: float) -> FieldValue:
    '''`value` at time `t`: a `TimeDependent` field fixed at `t`, anything else as is.'''
    return value.at(t) if isinstance(value, TimeDependent) else value


def is_mesh_bound(region: Region) -> bool:
    '''Whether `region` is tied to one specific mesh's vertex numbering.'''
    return bool(getattr(region, 'mesh_bound', False))


def _propagate_mesh_bound(combined: Region, regions: tuple[Region, ...]) -> Region:
    if any(is_mesh_bound(r) for r in regions):
        # Mirrors the getattr in is_mesh_bound: the flag rides on the callable
        # itself, so a bare lambda can carry it without a wrapper type.
        setattr(combined, 'mesh_bound', True)
    return combined


def _coerce_components(value: FieldValue, points: Vertices, n_components: int) -> FloatArray:
    '''Normalize a constant or a callable-of-position into an (N, n_components) array.

    Mechanical only: a `None` found among a value's components becomes `np.nan`,
    with no judgment about whether that is meaningful. `evaluate_field` and
    `Dirichlet`'s resolver both build on this and differ only in
    what a `NaN` component means to each of them.
    '''
    if value is None:
        return np.zeros((len(points), n_components))
    if isinstance(value, TimeDependent):
        raise TypeError('a TimeDependent field has no value without a time; use field_at(value, t)')

    def coerce(raw: float | Sequence[float | None] | FloatArray) -> FloatArray:
        # object dtype defers numeric coercion to the comprehension below, so a
        # scalar and a sequence, with or without a None in it, all flatten
        # to something iterable the same way.
        components = np.atleast_1d(np.asarray(raw, dtype=object))
        return np.array([np.nan if c is None else float(c) for c in components])

    if callable(value):
        values = np.array([coerce(value(p)) for p in points])
    else:
        values = np.tile(coerce(value), (len(points), 1))
    return values


def evaluate_field(value: FieldValue, points: Vertices, n_components: int, *,
                   free_as_zero: bool = False) -> FloatArray:
    '''Normalize a constant or a callable-of-position into an (N, n_components) array.

    A single rule, "the value at a point", for both forms; a value's width is
    checked against `n_components`, never inferred from the point count.

    Every component must be a real number: `None` has no meaning for a source or a
    Robin `g`. A `Neumann` value is the exception, where `None` names a component
    the traction does not drive; `free_as_zero` reads it as zero, its value in the
    integral. `Dirichlet` has its own resolver, where `None` leaves a component
    unconstrained.
    '''
    values = _coerce_components(value, points, n_components)
    if free_as_zero:
        values = np.nan_to_num(values, nan=0.0)

    if values.shape != (len(points), n_components):
        raise ValueError(
            f'field must give {n_components} component(s) per point, got shape {values.shape} '
            f'for {len(points)} point(s)'
        )
    if np.any(np.isnan(values)):
        raise ValueError(
            'field component is None (or NaN); every component must be a real number here'
        )
    return values
