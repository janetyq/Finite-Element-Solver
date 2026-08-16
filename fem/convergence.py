"""The Method of Manufactured Solutions: check the discretization against an answer
that is known exactly. It runs backwards from an ordinary solve -- the exact solution
u is *chosen* first and the forcing f (and boundary data) derived from it, so the
answer is known before the solver runs, and the solver never sees it.

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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fem.boundary import BCType, BoundaryConditions
from fem.elements import ElementGeometry, QuadraticTriangleElement
from fem.equations import LinearElastic, Poisson
from fem.forms import DiffusionForm, LaplacianForm, LinearElasticForm, LinearForm
from fem.integrators import ThetaMethod
from fem.materials import Enu_to_Lame, LinearElasticMaterial
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


def exact_gradient(points: FloatArray) -> FloatArray:
    """`grad u` of the manufactured solution, sampled at `points`.

    Broadcasts over any leading axes: `points` shaped `(..., 2)` gives `(..., 2)`,
    so it takes either the `(n_vertices, 2)` nodes or the `(n_elements, n_qp, 2)`
    quadrature points the H1 error integrates over. The closed-form gradient is what
    makes the H1 error independent of the assembled stiffness (see `h1_seminorm_error`).
    """
    x, y = points[..., 0], points[..., 1]
    return np.stack(
        [np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
         np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)],
        axis=-1,
    )


def source_term(point: FloatArray) -> list[float]:
    """`f = -laplacian(u)`, the forcing that makes `exact_solution` the answer."""
    return [2 * np.pi**2 * np.sin(np.pi * point[0]) * np.sin(np.pi * point[1])]


def l2_norm(space: FunctionSpace, values: VertexField) -> float:
    """The discrete L2 norm of a nodal field: `sqrt(v^T M v)` with `M` the mass matrix.

    Not the Euclidean norm of the same numbers -- that has no mesh in it, so it drifts
    with resolution and cannot be compared across a refinement sequence.

    This measures the distance to the *interpolant* of a reference field (it reads
    the reference only at the nodes). For the honest continuous error against a
    closed-form field, integrated at the quadrature points, see `quadrature_l2`.
    """
    return float(np.sqrt(values @ space.mass_matrix @ values))


def quadrature_l2(geometry: ElementGeometry, diff: FloatArray) -> float:
    """The L2 norm of a per-quadrature-point field: `sqrt(int |diff|^2 dx)`.

    `diff` carries a leading `(n_elements, n_qp)` pair and any number of trailing
    component axes -- a scalar `(n_el, n_qp)`, a gradient `(n_el, n_qp, d)`, or a
    stress tensor `(n_el, n_qp, d, d)`. Every trailing axis is summed (the Frobenius
    norm for a tensor), then integrated against the geometry's `weight_detJ`.

    Unlike `l2_norm` this samples the field at the interior quadrature points rather
    than reading a nodal interpolant, so with an analytic reference it is the true
    continuous error -- the shared kernel of the H1 seminorm and stress errors.
    """
    per_point = np.sum((diff * diff).reshape(diff.shape[0], diff.shape[1], -1), axis=2)
    return float(np.sqrt(np.sum(per_point * geometry.weight_detJ)))


def h1_seminorm_error(
    space: FunctionSpace, u: VertexField, exact_gradient: Callable[[FloatArray], FloatArray],
    degree: int = 2,
) -> float:
    """The H1 seminorm error `||grad(u_h) - grad(u_exact)||_L2`.

    The gradient error, and for P1 the O(h) quantity -- one order below the O(h^2)
    L2 error -- that is the sharper probe of the stiffness matrix. It is computed by
    quadrature against the *analytic* `exact_gradient`, so unlike `sqrt(e^T K e)` it
    never reuses the assembled `K` it is meant to test: a wrong `grad_phi` shows up
    here directly rather than being measured in its own distorted norm.

    Uses `geometry.gradients` at every quadrature point rather than the P1 shortcut
    `space.gradient`, so it is correct for the quadratic space too, where the
    gradient varies within an element.
    """
    geometry = space.geometry_at(degree)
    grad_h = geometry.gradients(u[space.element_nodes])   # (n_el, n_qp, spatial_dim)
    diff = exact_gradient(geometry.points) - grad_h
    return quadrature_l2(geometry, diff)


@dataclass
class MMSSolve:
    """One solve of the manufactured problem, and how far off it came out."""
    h: float                   # grid spacing
    mesh: Mesh
    u: VertexField             # what the solver computed
    exact: VertexField         # the manufactured solution at the same nodes
    l2_error: float            # ||u - exact||_L2 -- the number a study plots
    h1_error: float | None = None  # ||grad(u - exact)||_L2, where a closed-form gradient exists

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
        h1_error=h1_seminorm_error(solver.space, u, exact_gradient),
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
# operator now carries a position-dependent conductivity kappa = 1 + x + y.
# Differentiating that known u through the operator gives a forcing with an extra
# grad(kappa).grad(u) term a constant coefficient does not have:
#
#     f = -div(kappa grad u) = -(grad kappa . grad u) - kappa laplacian(u)
#
# The two varying fields feed opposite sides of the solve: kappa the operator
# (DiffusionForm -> stiffness matrix), the whole of f the load (LinearForm -> load
# vector). Neither is constant within an element, so both sides exercise the
# quadrature layer that a constant-coefficient assembly lacks. Asserted in
# tests/test_convergence_variable_coefficient.py.


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
    # f is fed as a LinearForm so it too is sampled at the quadrature points. A plain
    # field source would instead integrate f's nodal interpolant -- also convergent, but
    # LinearForm is the load half of what the quadrature layer added.
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


# --- P2 elements: the same Poisson problem, one polynomial degree higher ---------
#
# The same manufactured u = sin(pi x) sin(pi y), solved on the quadratic space. The
# only differences from solve_poisson_mms are the element type and that the exact
# solution and the error norm are sampled at *all* the P2 nodes -- corners and edge
# midpoints -- since the extra DOFs live there. The payoff is the rate: P2 is O(h^3)
# in L2 where P1 is O(h^2), which is what test_convergence_p2.py asserts and what
# Fact B (the edge-node DOFs) was built to deliver.


def solve_poisson_mms_p2(n: int) -> MMSSolve:
    """Solve the manufactured Poisson problem on a P2 space over an `n` x `n` grid."""
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    # The P2 stiffness integrand is degree 2, integrated exactly by the space's
    # default rule; the load int f phi is degree-4-ish, so the LinearForm samples f
    # at a degree-4 rule to keep quadrature error below the O(h^3) discretization error.
    problem = LinearProblem(
        space, LaplacianForm(), LinearForm(source_term, quadrature_degree=4), bc,
    )
    u = LinearSolve().solve(problem)

    # Sampled at the P2 nodes, not just the vertices: the L2 norm needs the edge-node
    # values the mass matrix is sized to.
    exact = exact_solution(space.node_coords)
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=u,
        exact=exact,
        l2_error=l2_norm(space, u - exact),
    )


def poisson_p2_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured Poisson problem on P2 per resolution, coarsest first."""
    return [solve_poisson_mms_p2(n) for n in sorted(resolutions)]


def solve_elastic_mms_p2(n: int) -> MMSSolve:
    """Solve the manufactured elasticity problem on a P2 space over an `n` x `n` grid.

    The vector P2 path: two DOFs on every node, corners and edge midpoints alike. It
    exercises the node numbering under `n_components = 2` and the coupled elastic
    operator, and it must converge at O(h^3) like the scalar P2 Poisson does.
    """
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)

    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), [0.0, 0.0])
    operator = LinearElasticForm(LinearElasticMaterial(ELASTIC_E, ELASTIC_NU))
    load = LinearForm(elastic_source, n_components=2, quadrature_degree=4)
    problem = LinearProblem(space, operator, load, bc)
    u = LinearSolve().solve(problem)

    exact = elastic_exact(space.node_coords)   # (n_nodes, 2)
    error = u.reshape(exact.shape) - exact
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=u,
        exact=exact.flatten(),
        l2_error=l2_norm(space, error.flatten()),
    )


def elastic_p2_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured elasticity problem on P2 per resolution, coarsest first."""
    return [solve_elastic_mms_p2(n) for n in sorted(resolutions)]


# --- quadrature-sampled load vs the nodal shortcut ------------------------------
#
# The MMS idea aimed at the load rather than the operator. The manufactured
# u = sin(k pi x) sin(k pi y) (still zero on the boundary) has a source
# f = 2 (k pi)^2 u that oscillates on a length scale 1/k. On a mesh that only just
# resolves it the two ways of building the load part company: a plain field source is
# integrated as its P1 interpolant (mass_matrix @ f(nodes)), which reads f only at the
# vertices and misses its swing between them, while a LinearForm samples f at the
# interior quadrature points. Both loads give O(h^2); the sampled one has the smaller
# constant. Drawn by the `quadrature_load` demo.

LOAD_MMS_FREQUENCY = 3   # source wavelengths across the unit square


def oscillatory_exact(vertices: Vertices) -> VertexField:
    """u = sin(k pi x) sin(k pi y): the manufactured solution, zero on the boundary."""
    k = LOAD_MMS_FREQUENCY
    x, y = vertices[:, 0], vertices[:, 1]
    return np.sin(k * np.pi * x) * np.sin(k * np.pi * y)


def oscillatory_source(point: FloatArray) -> list[float]:
    """f = -laplacian(u) = 2 (k pi)^2 sin(k pi x) sin(k pi y)."""
    k = LOAD_MMS_FREQUENCY
    return [2 * (k * np.pi) ** 2 * np.sin(k * np.pi * point[0]) * np.sin(k * np.pi * point[1])]


@dataclass
class LoadComparison:
    """One solve of the oscillatory problem with each kind of load, and how far off."""
    n: int                    # grid resolution (n x n nodes), for slicing a row of it
    h: float
    mesh: Mesh
    exact: VertexField
    nodal: VertexField        # source integrated as its P1 interpolant
    sampled: VertexField      # source sampled at the quadrature points (LinearForm)
    nodal_error: float
    sampled_error: float


def solve_load_comparison(n: int, quadrature_degree: int = 4) -> LoadComparison:
    """Solve -laplacian(u) = f on an `n` x `n` grid two ways.

    The same P1 space and operator both times; only the load differs -- a plain field
    source (integrated as its nodal interpolant) against a LinearForm that samples the
    source at the quadrature points.
    """
    mesh = create_rect_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, n_components=1)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    nodal = LinearSolve().solve(
        LinearProblem(space, LaplacianForm(), oscillatory_source, bc))
    sampled = LinearSolve().solve(
        LinearProblem(space, LaplacianForm(),
                      LinearForm(oscillatory_source, quadrature_degree=quadrature_degree), bc))

    exact = oscillatory_exact(mesh.vertices)
    return LoadComparison(
        n=n, h=1.0 / (n - 1), mesh=mesh, exact=exact, nodal=nodal, sampled=sampled,
        nodal_error=l2_norm(space, nodal - exact),
        sampled_error=l2_norm(space, sampled - exact),
    )


def load_comparison_convergence(resolutions: tuple[int, ...]) -> list[LoadComparison]:
    """Solve the oscillatory-load comparison per resolution, coarsest first."""
    return [solve_load_comparison(n) for n in sorted(resolutions)]


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
