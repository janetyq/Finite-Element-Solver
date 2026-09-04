"""Builders the test files share: the boundary conditions, sources, meshes, and the
solve-then-return shapes most of them would otherwise retype.

Plain functions rather than fixtures, so a test names what it uses and a reader can
follow it without pytest's indirection. Imported as `from helpers import ...`, the way
the MMS machinery is imported from `mms`.
"""
from collections.abc import Callable, Sequence

import numpy as np

from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.loads import Source
from fem.mesh.mesh import Mesh
from fem.regions import everywhere, on_plane


def pinned(n_components: int = 1) -> Conditions:
    """Homogeneous Dirichlet on the whole boundary: the scalar zero, or the zero vector."""
    value = 0.0 if n_components == 1 else [0.0] * n_components
    return Conditions(Dirichlet(everywhere(), value))


def rollers(dim: int) -> Conditions:
    """Each coordinate plane through the origin a roller: rigid modes removed, every
    face free to move away from it."""
    return Conditions(*[
        Dirichlet(on_plane(axis, 0.0), [0.0 if c == axis else None for c in range(dim)])
        for axis in range(dim)
    ])


def close(actual, expected, **tolerances) -> None:
    """`assert_allclose` with `expected` broadcast to `actual`'s shape: a closed form is
    one tensor, the solve reports one per element."""
    actual, expected = np.asarray(actual), np.asarray(expected, dtype=float)
    # Round-off against a closed form: an entry that should be zero comes out at
    # machine precision times the largest entry, which a relative tolerance rejects.
    tolerances.setdefault('atol', 1e-12 * max(1.0, float(np.abs(expected).max())))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), **tolerances)


def cantilever_bc(traction: Sequence[float] = (0.0, -1.0), length: float = 1.0) -> Conditions:
    """A 2D cantilever: the edge x = 0 clamped, the edge x = `length` under `traction`."""
    return Conditions(
        Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
        Neumann(on_plane(0, length), list(traction)),
    )


def solved(equation, mesh: Mesh, bc: Conditions, **kwargs):
    """The equation's problem on `mesh` and its solution; `kwargs` reach `problem()`."""
    problem = equation.problem(mesh, bc, **kwargs)
    return problem, problem.solve()


def problem_for(equation, bc: Conditions, **kwargs) -> Callable[[Mesh], object]:
    """The mesh -> problem closure `AdaptiveRefinement` re-solves through."""
    return lambda mesh: equation.problem(mesh, bc, **kwargs)


def localised_source(center: float = 0.5, radius: float = 0.1, strength: float = 10.0) -> Source:
    """`strength` inside a disc about `center`, zero outside: something to refine toward."""
    return Source(lambda p: np.where(np.linalg.norm(p - center, axis=1) < radius, strength, 0.0))


def near_far_counts(mesh: Mesh, center=0.5, near: float = 0.2, far: float = 0.35) -> tuple[int, int]:
    """How many elements have their centroid within `near` of `center`, and beyond `far`."""
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    distance = np.linalg.norm(centroids - np.asarray(center), axis=1)
    return int((distance < near).sum()), int((distance > far).sum())


def global_estimate(eta) -> float:
    """The global estimate sqrt(sum eta_K^2) from the per-element indicators."""
    return float(np.sqrt((np.asarray(eta) ** 2).sum()))


def two_triangle_square() -> Mesh:
    """The unit square as two equal triangles sharing the diagonal 0-2: small enough to
    work every edge's normal by hand. Vertices 0 and 2 belong to both elements, 1 and 3
    to one each."""
    return Mesh(
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )


class CountingBackend:
    """A `Backend` that counts how often it factors and how often each factorization
    solves, delegating the algebra to `inner` (direct by default).

    For the performance contracts (`tests/test_perf_contracts.py`): a solve path's cost
    is fixed by how many factorizations and back-substitutions it performs, which are
    exact counts a test can assert where wall-clock time is not.
    """

    def __init__(self, inner=None) -> None:
        from fem.algebra.backends import DirectBackend
        self.inner = inner if inner is not None else DirectBackend()
        self.factorizations = 0
        self.solves = 0

    def prepare(self, A):
        self.factorizations += 1
        return _CountingFactorization(self.inner.prepare(A), self)


class _CountingFactorization:
    def __init__(self, inner, backend: CountingBackend) -> None:
        self._inner = inner
        self._backend = backend

    def solve(self, b):
        self._backend.solves += 1
        return self._inner.solve(b)
