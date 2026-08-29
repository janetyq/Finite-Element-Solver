"""The Method of Manufactured Solutions: check the discretization against an answer
that is known exactly. It runs backwards from an ordinary solve: the exact solution
u is chosen first and the forcing f (and boundary data) derived from it, so the answer
is known before the problem runs, and the problem never sees it.

The manufactured solution is picked for convenience, not physics. What it establishes
is that assembly, boundary handling, and the solve together reproduce a known field,
and that the error closes at the rate the theory predicts: O(h^2) in L2 for P1. An
implementation with a subtly wrong element matrix typically still converges, just at
order 1, so the rate is the sharper claim and is what the study reports.

The manufactured problem is Poisson's on the unit square:

    u(x, y) = sin(pi x) sin(pi y)                 zero on the boundary, so the
    f(x, y) = -laplacian(u) = 2 pi^2 u(x, y)      Dirichlet data is homogeneous

Used by `tests/test_convergence.py`, which asserts the rate on every commit, and by
the `convergence` demo, which draws it.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fem.boundary import Dirichlet
from fem.conditions import Conditions
from fem.elements import (
    Element,
    ElementGeometry,
    IsoparametricTriangleElement,
    QuadraticTriangleElement,
)
from fem.physics.equations import Heat, LinearElastic, Poisson
from fem.physics.forms import DiffusionForm, LinearElasticForm
from fem.algebra.integrators import ThetaMethod
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.mesh.mesh import Mesh
from fem.mesh.structured import annulus_mesh, box_mesh
from fem.loads import Source
from fem.problem import LinearProblem
from fem.regions import everywhere
from fem.algebra.solve import LinearSolve
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
    quadrature points the H1 error integrates over. The closed-form gradient makes the
    H1 error independent of the assembled stiffness (see `h1_seminorm_error`).
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

    Not the Euclidean norm of the same numbers, which has no mesh in it, so it drifts
    with resolution and cannot be compared across a refinement sequence.

    This measures the distance to the interpolant of a reference field (it reads the
    reference only at the nodes). For the continuous error against a closed-form
    field, integrated at the quadrature points, see `quadrature_l2`.
    """
    return float(np.sqrt(values @ space.mass_matrix @ values))


def quadrature_l2(geometry: ElementGeometry, diff: FloatArray) -> float:
    """The L2 norm of a per-quadrature-point field: `sqrt(int |diff|^2 dx)`.

    `diff` carries a leading `(n_elements, n_qp)` pair and any number of trailing
    component axes: a scalar `(n_el, n_qp)`, a gradient `(n_el, n_qp, d)`, or a stress
    tensor `(n_el, n_qp, d, d)`. Every trailing axis is summed (the Frobenius norm for
    a tensor), then integrated against the geometry's `weight_detJ`.

    Unlike `l2_norm` this samples the field at the interior quadrature points rather
    than reading a nodal interpolant, so with an analytic reference it is the true
    continuous error, the shared kernel of the H1 seminorm and stress errors.
    """
    per_point = np.sum((diff * diff).reshape(diff.shape[0], diff.shape[1], -1), axis=2)
    return float(np.sqrt(np.sum(per_point * geometry.weight_detJ)))


def h1_seminorm_error(
    space: FunctionSpace, u: VertexField, exact_gradient: Callable[[FloatArray], FloatArray],
    degree: int = 2,
) -> float:
    """The H1 seminorm error `||grad(u_h) - grad(u_exact)||_L2`.

    The gradient error. For P1 it is O(h), one order below the O(h^2) L2 error, and
    the sharper probe of the stiffness matrix. It is computed by quadrature against
    the analytic `exact_gradient`, so unlike `sqrt(e^T K e)` it never reuses the
    assembled `K` it is meant to test: a wrong `grad_phi` shows up here directly
    rather than being measured in its own distorted norm.

    Uses `geometry.gradients` at every quadrature point rather than the P1 shortcut
    `space.gradient`, so it is correct for the quadratic space too, where the gradient
    varies within an element.
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
    u: VertexField             # what the problem computed
    exact: VertexField         # the manufactured solution at the same nodes
    l2_error: float            # ||u - exact||_L2, the number a study plots
    h1_error: float | None = None  # ||grad(u - exact)||_L2, where a closed-form gradient exists

    @property
    def pointwise_error(self) -> VertexField:
        """`u - exact` node by node. `l2_error` is the norm of this field, not of
        these numbers; see `l2_norm`."""
        return self.u - self.exact


@dataclass
class ConvergenceStudy:
    """A refinement sequence: the parameter refined, and the error at each value.

    `step` is whichever discretization parameter is being taken to zero: the mesh
    size `h` for a spatial study, the time step `dt` for a temporal one. The arithmetic
    is the same either way, so one type serves both; only the axis name differs.

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

        Steadier than any single pair, so it is the figure's headline number, but it
        averages away a rate that degrades under refinement; the per-pair `orders`
        would show that.
        """
        return float(np.polyfit(np.log(self.step), np.log(self.error), 1)[0])


def solve_poisson_mms(n: int) -> MMSSolve:
    """Solve the manufactured problem on an `n` x `n` unit-square grid."""
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    bc = Conditions(Dirichlet(everywhere(), 0.0))
    problem = Poisson().problem(mesh, bc + Source(source_term))
    solution = problem.solve()
    u = solution.u

    exact = exact_solution(mesh.vertices)
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=u,
        exact=exact,
        l2_error=l2_norm(problem.space, u - exact),
        h1_error=h1_seminorm_error(problem.space, u, exact_gradient),
    )


def poisson_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured problem once per resolution, coarsest first."""
    return [solve_poisson_mms(n) for n in sorted(resolutions)]


# --- elasticity: the same idea for a vector unknown -----------------------------

# The manufactured displacement moves in x only:
#
#     u = (sin(pi x) sin(pi y), 0)
#
# but the off-diagonal shear terms of sigma make both components of the forcing
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
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))

    bc = Conditions(Dirichlet(everywhere(), [0.0, 0.0]))
    equation = LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU)
    problem = equation.problem(mesh, bc + Source(elastic_source))
    solution = problem.solve()

    exact = elastic_exact(mesh.vertices)
    # The space's mass matrix is the scalar one repeated per component, so this is
    # the true vector L2 norm rather than the norm of component 0.
    error = solution.u.reshape(exact.shape) - exact
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        u=solution.u,
        exact=exact.flatten(),
        l2_error=l2_norm(problem.space, error.flatten()),
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
# (DiffusionForm -> stiffness matrix), the whole of f the load (Source -> load
# vector). Neither is constant within an element, so both sides exercise the
# quadrature layer that a constant-coefficient assembly lacks. Asserted in
# tests/test_convergence_variable_coefficient.py.


def variable_coefficient(point: FloatArray) -> float:
    """kappa(x, y) = 1 + x + y: smooth and positive on the unit square."""
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
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, n_components=1)

    bc = Conditions(Dirichlet(everywhere(), 0.0))
    # The source is a callable, so it is sampled at the quadrature points like the
    # coefficient; its nodal interpolant would also converge, at a larger constant.
    problem = Poisson(coefficient=variable_coefficient).problem(space, bc + Source(variable_source))
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
# solution and the error norm are sampled at all the P2 nodes (corners and edge
# midpoints) since the extra DOFs live there. P2 is O(h^3) in L2 where P1 is O(h^2),
# which test_convergence_p2.py asserts.


def solve_poisson_mms_p2(n: int) -> MMSSolve:
    """Solve the manufactured Poisson problem on a P2 space over an `n` x `n` grid."""
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)

    bc = Conditions(Dirichlet(everywhere(), 0.0))
    # The P2 stiffness integrand is degree 2, integrated exactly by the space's
    # default rule; the load int f phi is degree-4-ish, so the Source samples f
    # at a degree-4 rule to keep quadrature error below the O(h^3) discretization error.
    problem = LinearProblem(space, DiffusionForm(), bc + Source(source_term, quadrature_degree=4))
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
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)

    bc = Conditions(Dirichlet(everywhere(), [0.0, 0.0]))
    operator = LinearElasticForm(LinearElasticMaterial(ELASTIC_E, ELASTIC_NU))
    load = Source(elastic_source, n_components=2, quadrature_degree=4)
    problem = LinearProblem(space, operator, bc + load)
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


# --- curved (isoparametric) elements on an annulus ------------------------------
#
# The manufactured u = sin(x) sin(y), the same smooth field as the Poisson study but
# with inhomogeneous Dirichlet data, solved on an annulus whose two rims are true
# circles. Straight P2 approximates each rim by a chain of chords, so a geometry error
# of order h^2 caps its accuracy however high the element order. Isoparametric P2 puts
# its boundary edge nodes on the true circle and recovers the O(h^3) rate. That gap,
# the geometry floor against the recovered rate, is what test_convergence_curved.py
# asserts.

ANNULUS_INNER, ANNULUS_OUTER = 1.0, 2.0


def annulus_exact(points: FloatArray) -> FloatArray:
    """The manufactured u = sin(x) sin(y), sampled at `points` (any leading axes)."""
    return np.sin(points[..., 0]) * np.sin(points[..., 1])


def annulus_gradient(points: FloatArray) -> FloatArray:
    """grad u of the manufactured solution, broadcasting over any leading axes."""
    x, y = points[..., 0], points[..., 1]
    return np.stack([np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)], axis=-1)


def annulus_source(point: FloatArray) -> list[float]:
    """f = -laplacian(u) = 2 sin(x) sin(y)."""
    return [2.0 * np.sin(point[0]) * np.sin(point[1])]


def solve_annulus_mms(
    n: int, element_type: type[Element] = IsoparametricTriangleElement,
) -> MMSSolve:
    """Solve the manufactured annulus problem on a P2 space of `element_type`.

    `element_type` is `IsoparametricTriangleElement` (curved boundary, the accurate
    solve) or `QuadraticTriangleElement` (straight facets, the geometry floor), so one
    function measures both. `n` sets the radial resolution; the angular count scales
    with it to keep triangle aspect ratios bounded.
    """
    mesh = annulus_mesh(ANNULUS_INNER, ANNULUS_OUTER, n, 4 * n)

    bc = Conditions(Dirichlet(everywhere(), lambda p: [float(annulus_exact(np.asarray(p)))]))
    problem = Poisson().problem(mesh, bc + Source(annulus_source), element_type=element_type)
    solution = problem.solve()
    space = problem.space

    exact = annulus_exact(space.node_coords)
    return MMSSolve(
        h=(ANNULUS_OUTER - ANNULUS_INNER) / (n - 1),
        mesh=mesh,
        u=solution.u,
        exact=exact,
        l2_error=l2_norm(space, solution.u - exact),
        h1_error=h1_seminorm_error(space, solution.u, annulus_gradient, degree=4),
    )


def annulus_convergence(
    resolutions: tuple[int, ...],
    element_type: type[Element] = IsoparametricTriangleElement,
) -> list[MMSSolve]:
    """Solve the manufactured annulus problem per resolution, coarsest first."""
    return [solve_annulus_mms(n, element_type) for n in sorted(resolutions)]


# --- quadrature-sampled load vs the nodal shortcut ------------------------------
#
# The MMS idea aimed at the load rather than the operator. The manufactured
# u = sin(k pi x) sin(k pi y) (still zero on the boundary) has a source
# f = 2 (k pi)^2 u that oscillates on a length scale 1/k. On a mesh that only just
# resolves it the two ways of building the load part company: a plain field source is
# integrated as its P1 interpolant (mass_matrix @ f(nodes)), which reads f only at the
# vertices and misses its swing between them, while a Source samples f at the
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
    sampled: VertexField      # source sampled at the quadrature points (Source)
    nodal_error: float
    sampled_error: float


def solve_load_comparison(n: int, quadrature_degree: int = 4) -> LoadComparison:
    """Solve -laplacian(u) = f on an `n` x `n` grid two ways.

    The same P1 space and operator both times; only the load differs: a `Source`
    (the source integrated as its P1 interpolant) against a Source that samples
    the source at the quadrature points.
    """
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    space = FunctionSpace(mesh, n_components=1)
    bc = Conditions(Dirichlet(everywhere(), 0.0))

    nodal = LinearSolve().solve(
        LinearProblem(space, DiffusionForm(), bc + Source(oscillatory_source, nodal=True)))
    sampled = LinearSolve().solve(
        LinearProblem(space, DiffusionForm(), bc + Source(oscillatory_source, quadrature_degree=quadrature_degree)))

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

    Measured against the exact solution of the semi-discrete system `M u' = -K u`,
    which is `expm(-t M^-1 K) u0`, not against the continuous PDE. That isolates the
    integrator: no spatial discretization error enters, so the observed order is
    purely temporal and the mesh can stay coarse.

    theta = 1 is backward Euler and first order; theta = 1/2 is Crank-Nicolson,
    the default, and second.
    """
    from scipy.linalg import expm

    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = Conditions(Dirichlet(everywhere(), 0.0))
    u0 = exact_solution(mesh.vertices)   # an eigenmode, and zero on the boundary

    problem = Heat().problem(mesh, bc)
    free = problem.constraints[0]
    M = problem.space.mass_matrix.toarray()
    K = problem.tangent(None).toarray()
    propagator = expm(-T * np.linalg.solve(M[np.ix_(free, free)], K[np.ix_(free, free)]))
    reference = np.zeros_like(u0)
    reference[free] = propagator @ u0[free]

    errors = []
    for steps in sorted(step_counts):
        run = ThetaMethod(dt=T / steps, steps=steps, theta=theta).solve(problem, u0.copy())
        u_h = run.u[-1]
        errors.append(l2_norm(problem.space, u_h - reference))

    return ConvergenceStudy(np.array([T / k for k in sorted(step_counts)]), np.array(errors))
