"""Position-based regions and fields: specifications written against coordinates
rather than vertex indices.

A `Region` maps an (N, spatial_dim) array of point coordinates to an (N,) boolean
mask. `everywhere`, `on_plane`, `in_box`, and `at_indices` name the common cases;
a bare callable of the same shape is wrapped by `as_region`. Regions compose with
`&`, `|`, and `~` (`left & bottom`, `~left`), and own the coordinate tolerance. The
concrete region types are private: the helpers and the operators are the API.

A `Field` is a prescribed value over the domain: a source, a coefficient, a traction,
a boundary value. `as_field` normalizes a constant, a per-component constant, or a
callable of position into one; `sample` evaluates it to an (N, n_components) array. A
plain callable is read point by point; wrap it in `Vectorized` to promise it takes
the whole (N, d) array at once, the fast path for an assembly-hot source or
coefficient. A `TimeDependent` field is a callable of position and time; `at(t)`
fixes the time and returns the plain field of position every consumer takes.

Both regions and fields are mesh-independent, so a boundary condition described this
way can be resolved again against whatever mesh is current and survives refinement.
`at_indices` is the escape hatch for node-specific work; it reports `mesh_bound` so
remeshers can refuse it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from fem.typing import BoolArray, FieldValue, FloatArray, IntArray, Point, Vertices

# Coordinates come from linspace/midpoint arithmetic, so exact boundary values
# are representable; this only absorbs round-off, not genuine mesh spacing.
DEFAULT_ATOL: float = 1e-9


# -- regions -------------------------------------------------------------------------


class Region(ABC):
    '''A geometric region: (N, spatial_dim) coordinates -> (N,) boolean membership
    mask. Build one with a helper (`everywhere`, `on_plane`, `in_box`, `at_indices`)
    or `as_region` around a bare callable, and combine with `&`, `|`, and `~`.'''

    @abstractmethod
    def __call__(self, points: Vertices) -> BoolArray: ...

    @property
    def mesh_bound(self) -> bool:
        '''Whether the region names vertices of one specific mesh, so a remesh that
        renumbers vertices invalidates it. False for a geometric region.'''
        return False

    def __and__(self, other: RegionLike) -> 'Region':
        return _Intersect((self, as_region(other)))

    def __rand__(self, other: RegionLike) -> 'Region':
        return _Intersect((as_region(other), self))

    def __or__(self, other: RegionLike) -> 'Region':
        return _Union((self, as_region(other)))

    def __ror__(self, other: RegionLike) -> 'Region':
        return _Union((as_region(other), self))

    def __invert__(self) -> 'Region':
        return _Complement(self)


# A region argument: either a `Region` object or a bare callable `as_region` wraps.
RegionLike = Region | Callable[[Vertices], BoolArray]


@dataclass(frozen=True)
class _Everywhere(Region):
    '''Every point (`everywhere`). With the boundary-only resolution this means "the
    entire boundary", the most common Dirichlet region.'''

    def __call__(self, points: Vertices) -> BoolArray:
        return np.ones(len(points), dtype=bool)


@dataclass(frozen=True)
class _OnPlane(Region):
    '''Points whose `axis` coordinate equals `value` (`on_plane`).'''
    axis: int
    value: float
    atol: float = DEFAULT_ATOL

    def __call__(self, points: Vertices) -> BoolArray:
        return np.abs(points[:, self.axis] - self.value) <= self.atol


@dataclass(frozen=True)
class _InBox(Region):
    '''Points inside an axis-aligned box, inclusive (`in_box`). A `None` bound leaves
    that direction unbounded.'''
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]
    atol: float = DEFAULT_ATOL

    def __call__(self, points: Vertices) -> BoolArray:
        mask = np.ones(len(points), dtype=bool)
        for axis, bound in enumerate(self.lower):
            if bound is not None:
                mask &= points[:, axis] >= bound - self.atol
        for axis, bound in enumerate(self.upper):
            if bound is not None:
                mask &= points[:, axis] <= bound + self.atol
        return mask


@dataclass(frozen=True)
class _AtIndices(Region):
    '''Named vertex indices (`at_indices`): the escape hatch for work about specific
    nodes rather than a place in the domain.

    Mesh-bound by construction: indices mean nothing once a remesher renumbers
    vertices, so `mesh_bound` is True and callers that remesh refuse it.
    '''
    indices: tuple[int, ...]

    @property
    def mesh_bound(self) -> bool:
        return True

    def __call__(self, points: Vertices) -> BoolArray:
        mask = np.zeros(len(points), dtype=bool)
        mask[list(self.indices)] = True
        return mask


@dataclass(frozen=True)
class _Intersect(Region):
    '''Points in every one of `regions` (`&`, `intersect`).'''
    regions: tuple[Region, ...]

    @property
    def mesh_bound(self) -> bool:
        return any(r.mesh_bound for r in self.regions)

    def __call__(self, points: Vertices) -> BoolArray:
        mask = np.ones(len(points), dtype=bool)
        for r in self.regions:
            mask &= r(points)
        return mask


@dataclass(frozen=True)
class _Union(Region):
    '''Points in any of `regions` (`|`, `union`).'''
    regions: tuple[Region, ...]

    @property
    def mesh_bound(self) -> bool:
        return any(r.mesh_bound for r in self.regions)

    def __call__(self, points: Vertices) -> BoolArray:
        mask = np.zeros(len(points), dtype=bool)
        for r in self.regions:
            mask |= r(points)
        return mask


@dataclass(frozen=True)
class _Complement(Region):
    '''Points outside `region` (`~`). Unbounded over the whole domain, so it is
    meaningful in a boundary condition, where it intersects the boundary.'''
    region: Region

    @property
    def mesh_bound(self) -> bool:
        return self.region.mesh_bound

    def __call__(self, points: Vertices) -> BoolArray:
        return ~self.region(points)


@dataclass(frozen=True)
class _CallableRegion(Region):
    '''A bare callable wrapped as a `Region`, so a user-written lambda composes with
    the operators and answers `mesh_bound` (False) like any geometric region.'''
    fn: Callable[[Vertices], BoolArray]

    def __call__(self, points: Vertices) -> BoolArray:
        return np.asarray(self.fn(points), dtype=bool)


def as_region(region: RegionLike) -> Region:
    '''`region` as a `Region`: itself if it already is one, else a bare callable
    wrapped so it composes and reports `mesh_bound`.'''
    if isinstance(region, Region):
        return region
    if callable(region):
        return _CallableRegion(region)
    raise TypeError(
        'a region must be a Region or a callable over point coordinates; pass a '
        'helper from fem.regions (e.g. on_plane(0, 0.0)), or at_indices([...]) for '
        f'specific nodes. Got {type(region).__name__}.'
    )


def everywhere() -> Region:
    '''Every point.'''
    return _Everywhere()


def on_plane(axis: int, value: float, atol: float = DEFAULT_ATOL) -> Region:
    '''Points on the plane `axis = value` (the left edge is `on_plane(0, 0.0)`).'''
    return _OnPlane(axis, value, atol)


def in_box(
    lower: Sequence[float | None],
    upper: Sequence[float | None],
    atol: float = DEFAULT_ATOL,
) -> Region:
    '''Points inside an axis-aligned box, inclusive. A `None` bound leaves that
    direction unbounded, so a band in y is `in_box([None, 0.2], [None, 0.8])`.'''
    return _InBox(tuple(lower), tuple(upper), atol)


def intersect(*regions: RegionLike) -> Region:
    '''Points in every one of `regions`. `left & bottom` is the binary form.'''
    return _Intersect(tuple(as_region(r) for r in regions))


def union(*regions: RegionLike) -> Region:
    '''Points in any of `regions`. `left | bottom` is the binary form.'''
    return _Union(tuple(as_region(r) for r in regions))


def at_indices(indices: Sequence[int] | IntArray) -> Region:
    '''The named vertices `indices`. Mesh-bound, so a remesher refuses it.'''
    return _AtIndices(tuple(int(i) for i in np.asarray(indices, dtype=int).ravel()))


def is_mesh_bound(region: RegionLike) -> bool:
    '''Whether `region` is tied to one specific mesh's vertex numbering.'''
    return as_region(region).mesh_bound


# -- fields --------------------------------------------------------------------------


class Field(ABC):
    '''A prescribed value over the domain: a source, a coefficient, a traction, a
    boundary value. `as_field` builds one from a raw `FieldValue`; every consumer
    samples it the same way, so the polymorphic union is decoded once at the boundary
    rather than at each call site. `Vectorized` and `TimeDependent` are the two a user
    constructs by hand; a constant or a plain callable normalizes on its own.'''

    @abstractmethod
    def sample(self, points: Vertices) -> FloatArray:
        '''(N, spatial_dim) coordinates -> (N, n_components) values.'''

    @property
    def is_time_dependent(self) -> bool:
        return False

    @property
    def is_pointwise(self) -> bool:
        '''Whether the value varies within an element, so it must be sampled at the
        quadrature points rather than integrated exactly as a constant. True for every
        field but a constant.'''
        return True


@dataclass(frozen=True)
class _Constant(Field):
    '''A value the same at every point: a scalar, a per-component vector, or a `None`
    left as `NaN` for a free Dirichlet component. `values` is the (n_components,)
    per-point value, already validated by `as_field`.'''
    values: FloatArray

    def sample(self, points: Vertices) -> FloatArray:
        return np.tile(self.values, (len(points), 1))

    @property
    def is_pointwise(self) -> bool:
        return False


@dataclass(frozen=True)
class _Pointwise(Field):
    '''A callable of position read one point at a time: `fn` receives a single `(d,)`
    coordinate and returns that point's value. The safe default for a user-written
    lambda, since `p[0]` means the same thing on one point as the author intends;
    `Vectorized` is the fast path for a callable that takes the whole array. Built by
    `as_field` from a bare callable, so `n_components` comes from the consumer.'''
    fn: Callable[..., Any]
    n_components: int
    allow_free: bool = False

    def sample(self, points: Vertices) -> FloatArray:
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n == 0:
            return np.zeros((0, self.n_components))
        values = np.array([_coerce_value(self.fn(p)) for p in pts])
        return _validate(values, n, self.n_components, self.allow_free)


@dataclass(frozen=True)
class Vectorized(Field):
    '''A callable that takes the whole `(N, d)` array of coordinates at once and
    returns `(N, k)` (or `(N,)` for a scalar field): `fn` is array-aware, so one call
    replaces the per-point loop, the fast path for an assembly-hot source or
    coefficient. The user declares this contract by wrapping the callable; nothing
    tries to guess it, because a point-by-point lambda handed the batched array can
    return a plausibly-shaped but wrong result.

    `n_components` is inferred from the first sample's width unless given; pass it to
    pin the width and catch a wrong-shaped result early. Pass a `Vectorized` where a
    `FieldValue` is taken (a `Source`, a `Dirichlet` value); it is a `Field`, so
    `as_field` returns it unchanged. `allow_free` permits a `NaN` component,
    meaningful only for a Dirichlet value.
    '''
    fn: Callable[[Vertices], FloatArray]
    n_components: int | None = None
    allow_free: bool = False

    def sample(self, points: Vertices) -> FloatArray:
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n == 0:
            return np.zeros((0, self.n_components or 1))
        out = np.asarray(self.fn(pts), dtype=float)
        if out.ndim == 1:                       # a scalar field returning (N,)
            out = out.reshape(n, 1)
        k = self.n_components if self.n_components is not None else out.shape[-1]
        return _validate(out, n, k, self.allow_free)


@dataclass(frozen=True)
class TimeDependent(Field):
    '''A field that varies in time: `fn(p, t)` is the value at point `p` and time `t`.

    A source, a traction, a Robin `g`, or a Dirichlet value may be one; the
    integrators evaluate it at each step through `Problem.load_at` and
    `Problem.constraints_at`. `at(t)` fixes the time and returns the plain field of
    position (a callable) every other consumer takes.
    '''
    fn: Callable[[Point, float], Any]

    @property
    def is_time_dependent(self) -> bool:
        return True

    def sample(self, points: Vertices) -> FloatArray:
        raise TypeError('a TimeDependent field has no value without a time; use field_at(value, t)')

    def at(self, t: float) -> Callable[[Point], Any]:
        fn = self.fn
        return lambda p: fn(p, t)


def as_field(value: FieldValue, n_components: int, *, allow_free: bool = False) -> Field:
    '''Normalize a raw `FieldValue` into a `Field`.

    A `Field` (including a `Vectorized` or `TimeDependent`) is returned as is; a
    callable becomes a per-point field; anything else is a constant, its width checked
    against `n_components` now so a wrong-width value fails at the boundary.
    `allow_free` permits a `None` component (left as `NaN`), meaningful only for a
    Dirichlet value.
    '''
    if isinstance(value, Field):
        return value
    if callable(value):
        return _Pointwise(value, n_components, allow_free)
    return _Constant(_coerce_constant(value, n_components, allow_free))


def field_at(value: FieldValue, t: float) -> FieldValue:
    '''`value` at time `t`: a `TimeDependent` field fixed at `t`, anything else as is.'''
    return value.at(t) if isinstance(value, TimeDependent) else value


def is_pointwise(value: FieldValue) -> bool:
    '''Whether `value` varies within an element and must be sampled at the quadrature
    points, rather than integrated exactly as a constant: a callable, a `Vectorized`,
    or a `TimeDependent`, but not a constant or a nodal array. The predicate the
    assembling forms and loads branch on.'''
    if isinstance(value, Field):
        return value.is_pointwise
    return callable(value)


def evaluate_field(value: FieldValue, points: Vertices, n_components: int) -> FloatArray:
    '''Normalize `value` and sample it at `points`: an (N, n_components) array.

    A single rule, "the value at a point", for a constant and a callable alike; a
    value's width is checked against `n_components`, never inferred from the point
    count. Every component must be a real number: `None` has no meaning for a source,
    a traction, or a Robin `g`. Use a Dirichlet value for a component left free.
    '''
    return as_field(value, n_components).sample(points)


def sample_natural_width(value: FieldValue, points: Vertices) -> FloatArray:
    '''Sample `value` at the width it declares, a `None` component kept as `NaN`.

    The inspection path: where the DOF count is not known (`BoundaryConditions.entries`,
    which feeds the plot) the value's own width is the answer, so it is not checked
    against a fixed `n_components` the way `evaluate_field` checks it. A `Field` samples
    itself (a `TimeDependent` raises, as it has no value without a time); a constant or
    a callable takes the width it gives. Pass `field_at(value, t)` for a time value.'''
    if isinstance(value, Field):
        return value.sample(points)
    if value is None:
        return np.zeros((len(points), 1))
    if callable(value):
        if len(points) == 0:
            return np.zeros((0, 1))
        fn = cast(Callable[[Point], Any], value)
        return np.array([_coerce_value(fn(p)) for p in points])
    return np.tile(_coerce_value(value), (len(points), 1))


def _coerce_value(raw: float | Sequence[float | None] | FloatArray) -> FloatArray:
    '''A single point's raw value as a (k,) float array, `None` -> `NaN`. Object
    dtype defers numeric coercion so a scalar and a sequence flatten the same way;
    the width is checked by `_validate` against the whole array.'''
    components = np.atleast_1d(np.asarray(raw, dtype=object))
    return np.array([np.nan if c is None else float(c) for c in components])


def _coerce_constant(
    value: float | Sequence[float | None] | FloatArray | None, n_components: int, allow_free: bool,
) -> FloatArray:
    '''A constant value as a validated (n_components,) array.'''
    if value is None:
        return np.zeros(n_components)
    arr = _coerce_value(value)
    if arr.shape != (n_components,):
        raise ValueError(
            f'field must give {n_components} component(s) per point, got a constant '
            f'of width {arr.shape[0]}'
        )
    if not allow_free and np.any(np.isnan(arr)):
        raise ValueError('field component is None; every component must be a real number here')
    return arr


def _validate(values: FloatArray, n: int, n_components: int, allow_free: bool) -> FloatArray:
    '''Check a sampled array is (n, n_components) and, unless `allow_free`, real.'''
    values = np.asarray(values, dtype=float)
    if values.shape != (n, n_components):
        raise ValueError(
            f'field must give {n_components} component(s) per point, got shape '
            f'{values.shape} for {n} point(s)'
        )
    if not allow_free and np.any(np.isnan(values)):
        raise ValueError(
            'field component is None (or NaN); every component must be a real number here'
        )
    return values
