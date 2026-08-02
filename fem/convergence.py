"""The Method of Manufactured Solutions: check the discretization against an answer
that is known exactly.

Nothing here validates a *model*. The manufactured solution is picked for convenience
rather than for physics -- what it establishes is that the assembly, the boundary
handling and the solve together reproduce a known field, and that the gap between them
closes at the rate the theory predicts. For P1 elements that rate is O(h^2) in L2, and
an implementation with a subtly wrong element matrix typically still converges, just at
order 1. The rate is the sharper claim, which is why it is what the study reports.

The manufactured problem is Poisson's on the unit square:

    u(x, y) = sin(pi x) sin(pi y)                 zero on the boundary, so the
    f(x, y) = -laplacian(u) = 2 pi^2 u(x, y)      Dirichlet data is homogeneous

Used by `tests/test_convergence.py`, which asserts the rate on every commit, and by
the `convergence` demo, which draws it.
"""
from dataclasses import dataclass

import numpy as np

from fem.boundary import BCType, BoundaryConditions
from fem.equations import Poisson
from fem.mesh.mesh import Mesh
from fem.mesh.ruppert import create_rect_mesh
from fem.regions import everywhere
from fem.solution import FieldSolution
from fem.solver import Solver
from fem.space import FunctionSpace
from fem.typing import FloatArray, Vertices, VertexField


def exact_solution(vertices: Vertices) -> VertexField:
    """The manufactured `u`, sampled at `vertices`."""
    x, y = vertices[:, 0], vertices[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_term(point: FloatArray) -> list[float]:
    """`f = -laplacian(u)`, the forcing that makes `exact_solution` the answer."""
    return [2 * np.pi**2 * np.sin(np.pi * point[0]) * np.sin(np.pi * point[1])]


def l2_norm(space: FunctionSpace, values: VertexField) -> float:
    """The discrete L2 norm of a nodal field: `sqrt(v^T M v)` with `M` the mass matrix.

    Not the Euclidean norm of the same numbers -- that has no mesh in it, so it drifts
    with resolution and cannot be compared across a refinement sequence.
    """
    return float(np.sqrt(values @ space.mass_matrix @ values))


@dataclass
class MMSSolve:
    """One solve of the manufactured problem, and how far off it came out."""
    h: float                   # grid spacing
    mesh: Mesh
    u: VertexField             # what the solver computed
    exact: VertexField         # the manufactured solution at the same nodes
    l2_error: float            # ||u - exact||_L2 -- the number a study plots

    @property
    def pointwise_error(self) -> VertexField:
        """`u - exact` node by node. `l2_error` is the norm of this field, not of
        these numbers; see `l2_norm`."""
        return self.u - self.exact


@dataclass
class ConvergenceStudy:
    """A refinement sequence: mesh sizes and the error at each, finest last."""
    h: FloatArray
    error: FloatArray

    @classmethod
    def from_solves(cls, solves: list[MMSSolve]) -> 'ConvergenceStudy':
        return cls(np.array([s.h for s in solves]), np.array([s.l2_error for s in solves]))

    @property
    def orders(self) -> FloatArray:
        """Order observed between each successive pair, so `len(h) - 1` of them.

        From `error ~ C h^p`: `p = log(e1/e2) / log(h1/h2)`.
        """
        return (np.log(self.error[:-1] / self.error[1:])
                / np.log(self.h[:-1] / self.h[1:]))

    @property
    def fitted_order(self) -> float:
        """One order for the whole sequence: the slope of log(error) against log(h).

        Steadier than any single pair, which is what makes it the figure's headline
        number, but it averages away a rate that degrades under refinement -- the
        per-pair `orders` are what would show that.
        """
        return float(np.polyfit(np.log(self.h), np.log(self.error), 1)[0])


def solve_poisson_mms(n: int) -> MMSSolve:
    """Solve the manufactured problem on an `n` x `n` unit-square grid."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solver = Solver(mesh, Poisson(source=source_term), bc)
    # Poisson is a scalar field equation, so this is a FieldSolution; `solve` declares
    # the base Solution, which carries no `u`.
    solution = solver.solve()
    assert isinstance(solution, FieldSolution)
    u = solution.u

    exact = exact_solution(mesh.vertices)
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=u,
        exact=exact,
        l2_error=l2_norm(solver.space, u - exact),
    )


def poisson_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured problem once per resolution, coarsest first."""
    return [solve_poisson_mms(n) for n in sorted(resolutions)]
