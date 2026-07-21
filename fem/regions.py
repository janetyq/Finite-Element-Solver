"""Position-based regions and fields: specifications written against coordinates
rather than vertex indices.

A *region* is any callable mapping an (N, spatial_dim) array of point coordinates to an
(N,) boolean mask, so a bare lambda qualifies. The helpers below just name the
cases that kept recurring and own the coordinate tolerance, which was previously
an ad-hoc `< 1e-6` re-derived at every call site.

A *field* is a value that may be either a constant or a callable of position;
`evaluate_field` normalizes both into a (N, n_components) array.

Both are deliberately mesh-independent, and that is the point: a boundary
condition described this way can be resolved afresh against whatever mesh is
current, which is what lets it survive refinement. `at_indices` is the escape
hatch for genuinely node-specific work; it marks itself mesh-bound so remeshers
can refuse it instead of silently relocating it.
"""
from collections.abc import Sequence

import numpy as np

from fem.typing import BoolArray, FieldValue, FloatArray, IntArray, Region, Vertices

# Coordinates come from linspace/midpoint arithmetic, so exact boundary values
# are representable; this only absorbs round-off, not genuine mesh spacing.
DEFAULT_ATOL: float = 1e-9


def everywhere() -> Region:
    '''Every point. Combined with the boundary-only resolution, this means
    "the entire boundary" -- the most common Dirichlet region.'''
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
    '''Named vertex indices. The escape hatch for work that is genuinely about
    specific nodes rather than a place in the domain.

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


def is_mesh_bound(region: Region) -> bool:
    '''Whether `region` is tied to one specific mesh's vertex numbering.'''
    return bool(getattr(region, 'mesh_bound', False))


def _propagate_mesh_bound(combined: Region, regions: tuple[Region, ...]) -> Region:
    if any(is_mesh_bound(r) for r in regions):
        # Mirrors the getattr in is_mesh_bound: the flag rides on the callable
        # itself, so a bare lambda can carry it without a wrapper type.
        setattr(combined, 'mesh_bound', True)
    return combined


def evaluate_field(value: FieldValue, points: Vertices, n_components: int) -> FloatArray:
    '''Normalize a constant or a callable-of-position into an (N, n_components) array.

    A single rule -- "the value at a point" -- for both forms. The previous API
    chose between "one value per index" and "one value shared by all indices" by
    comparing `len(indices) == len(values)`, so `add('dirichlet', left, [0, 0])`
    on 2D elasticity silently changed meaning when the edge happened to hold
    exactly two nodes.
    '''
    if value is None:
        return np.zeros((len(points), n_components))

    if callable(value):
        values = np.array([np.atleast_1d(np.asarray(value(p), dtype=float)) for p in points])
    else:
        single = np.atleast_1d(np.asarray(value, dtype=float))
        values = np.tile(single, (len(points), 1))

    if values.shape != (len(points), n_components):
        raise ValueError(
            f'field must give {n_components} component(s) per point, got shape {values.shape} '
            f'for {len(points)} point(s)'
        )
    return values
