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
from fem.equations import LinearElastic, Poisson
from fem.forms import DiffusionForm, LinearForm
from fem.integrators import ThetaMethod
from fem.materials import Enu_to_Lame
from fem.mesh.mesh import Mesh
from fem.mesh.ruppert import create_rect_mesh
from fem.problem import LinearProblem, heat
from fem.regions import everywhere
from fem.solution import FieldSolution, TransientSolution
from fem.solve import LinearSolve
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
    """A refinement sequence: the parameter refined, and the error at each value.

    `step` is whichever discretization parameter is being taken to zero -- the mesh
    size `h` for a spatial study, the time step `dt` for a temporal one. The
    arithmetic is the same either way, which is why one type serves both; what
    differs is only what the axis is called.

    Ordered coarsest first, so the last entry is the most refined.
    """
    step: FloatArray
    error: FloatArray

    @classmethod
    def from_solves(cls, solves: list['MMSSolve']) -> 'ConvergenceStudy':
        return cls(np.array([s.h for s in solves]), np.array([s.l2_error for s in solves]))

    @property
    def orders(self) -> FloatArray:
        """Order observed between each successive pair, so `len(step) - 1` of them.

        From `error ~ C step^p`: `p = log(e1/e2) / log(s1/s2)`.
        """
        return (np.log(self.error[:-1] / self.error[1:])
                / np.log(self.step[:-1] / self.step[1:]))

    @property
    def fitted_order(self) -> float:
        """One order for the whole sequence: the slope of log(error) against log(step).

        Steadier than any single pair, which is what makes it the figure's headline
        number, but it averages away a rate that degrades under refinement -- the
        per-pair `orders` are what would show that.
        """
        return float(np.polyfit(np.log(self.step), np.log(self.error), 1)[0])


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


# --- elasticity: the same idea for a vector unknown -----------------------------

# The manufactured displacement moves in x only:
#
#     u = (sin(pi x) sin(pi y), 0)
#
# but the off-diagonal shear terms of sigma make *both* components of the forcing
# non-zero, so this exercises the coupled vector path rather than a scalar solve
# wearing two components. Asserted in tests/test_convergence_elasticity.py, which
# also covers the 3D case.
ELASTIC_E, ELASTIC_NU = 200.0, 0.3


def elastic_source(point: FloatArray) -> list[float]:
    """The body force that makes `elastic_exact` the answer, for a plane-strain solid."""
    mu, lamb = Enu_to_Lame(ELASTIC_E, ELASTIC_NU)
    x, y = point
    return [
        np.pi**2 * (3*mu + lamb) * np.sin(np.pi * x) * np.sin(np.pi * y),
        -(mu + lamb) * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y),
    ]


def elastic_exact(vertices: Vertices) -> FloatArray:
    """The manufactured displacement, `(n_vertices, 2)`."""
    exact = np.zeros((len(vertices), 2))
    exact[:, 0] = np.sin(np.pi * vertices[:, 0]) * np.sin(np.pi * vertices[:, 1])
    return exact


def solve_elastic_mms(n: int) -> MMSSolve:
    """Solve the manufactured elasticity problem on an `n` x `n` unit-square grid."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0])
    equation = LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU, source=elastic_source)
    solver = Solver(mesh, equation, bc)
    solution = solver.solve()
    assert isinstance(solution, FieldSolution)

    exact = elastic_exact(mesh.vertices)
    # The space's mass matrix is the scalar one repeated per component, so this is
    # the true vector L2 norm rather than the norm of component 0.
    error = solution.u.reshape(exact.shape) - exact
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=solution.u,
        exact=exact.flatten(),
        l2_error=l2_norm(solver.space, error.flatten()),
    )


def elastic_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured elasticity problem once per resolution, coarsest first."""
    return [solve_elastic_mms(n) for n in sorted(resolutions)]


# --- variable coefficient: -div(kappa(x) grad u) = f ----------------------------
#
# The same manufactured u = sin(pi x) sin(pi y) (zero on the boundary), but the
# operator now carries a position-dependent conductivity kappa = 1 + x + y. The
# forcing gains the grad(kappa).grad(u) term a constant coefficient does not have:
#
#     f = -div(kappa grad u) = -(grad kappa . grad u) - kappa laplacian(u)
#
# kappa varies within every element, so a constant-coefficient assembly cannot
# represent this at all. It exercises the quadrature layer on both sides -- the
# DiffusionForm operator and a LinearForm load, each sampling its field at the
# quadrature points. Asserted in tests/test_convergence_variable_coefficient.py.


def variable_coefficient(point: FloatArray) -> float:
    """kappa(x, y) = 1 + x + y -- smooth and positive on the unit square."""
    return 1.0 + point[0] + point[1]


def variable_source(point: FloatArray) -> list[float]:
    """f = -div(kappa grad u) for the kappa above and u = sin(pi x) sin(pi y)."""
    x, y = point[0], point[1]
    sx, sy = np.sin(np.pi * x), np.sin(np.pi * y)
    cx, cy = np.cos(np.pi * x), np.cos(np.pi * y)
    grad_kappa_dot_grad_u = np.pi * cx * sy + np.pi * sx * cy
    kappa_times_laplacian = (1.0 + x + y) * (-2 * np.pi**2 * sx * sy)
    return [-(grad_kappa_dot_grad_u + kappa_times_laplacian)]


def solve_variable_coefficient_mms(n: int) -> MMSSolve:
    """Solve the manufactured variable-coefficient problem on an `n` x `n` grid."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, n_components=1)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    # Both sides sampled at the quadrature points: the operator's kappa through the
    # DiffusionForm, the load's f through a LinearForm. A plain-field source would
    # integrate f's nodal interpolant instead, and also converge; the LinearForm is
    # the load half of what the quadrature layer added.
    problem = LinearProblem(
        space,
        DiffusionForm(variable_coefficient),
        LinearForm(variable_source, n_components=1),
        bc,
    )
    u = LinearSolve().solve(problem)

    exact = exact_solution(mesh.vertices)
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=u,
        exact=exact,
        l2_error=l2_norm(space, u - exact),
    )


def variable_coefficient_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured variable-coefficient problem per resolution, coarsest first."""
    return [solve_variable_coefficient_mms(n) for n in sorted(resolutions)]


# --- the heat integrators: convergence in dt rather than in h -------------------

def theta_convergence(theta: float, step_counts: tuple[int, ...], T: float = 0.02,
                      n: int = 11) -> ConvergenceStudy:
    """Temporal convergence of `ThetaMethod` at `theta`, over `T` on an `n` x `n` grid.

    Measured against the exact solution of the *semi-discrete* system `M u' = -K u`,
    which is `expm(-t M^-1 K) u0` -- not against the continuous PDE. That is what
    isolates the integrator: no spatial discretization error enters, so the observed
    order is purely temporal and the mesh can stay coarse.

    theta = 1 is backward Euler and first order; theta = 1/2 is Crank-Nicolson,
    the default, and second.
    """
    from scipy.linalg import expm

    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    u0 = exact_solution(mesh.vertices)   # an eigenmode, and zero on the boundary

    problem = heat(mesh, bc=bc)
    free = problem.constraints[0]
    M = problem.space.mass_matrix.toarray()
    K = problem.tangent(None).toarray()
    propagator = expm(-T * np.linalg.solve(M[np.ix_(free, free)], K[np.ix_(free, free)]))
    reference = np.zeros_like(u0)
    reference[free] = propagator @ u0[free]

    errors = []
    for steps in sorted(step_counts):
        # A time-stepped solve returns a TransientSolution; `run` declares the base
        # Solution, which carries no series.
        run = ThetaMethod(dt=T / steps, steps=steps, theta=theta).run(problem, u0.copy())
        assert isinstance(run, TransientSolution)
        u_h = run.u[-1]
        errors.append(l2_norm(problem.space, u_h - reference))

    return ConvergenceStudy(np.array([T / k for k in sorted(step_counts)]), np.array(errors))
