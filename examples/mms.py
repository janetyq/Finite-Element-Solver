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

from fem.algebra.backends import Backend
from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial
from fem.field import NodalField
from fem.elements import (
    Element,
    ElementGeometry,
    IsoparametricTriangleElement,
    QuadraticTriangleElement,
)
from fem.physics.equations import Heat, LinearElastic, Poisson
from fem.physics.forms import DiffusionForm, LinearElasticForm, ThermalStrain
from fem.algebra.integrators import ThetaMethod
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial, Reduction
from fem.mesh.curves import Circle
from fem.mesh.mesh import Mesh, boundary_facets
from fem.mesh.structured import box_mesh
from fem.loads import Source
from fem.problem import LinearProblem
from fem.regions import everywhere, on_plane
from fem.algebra.solve import LinearSolve
from fem.space import FunctionSpace
from fem.typing import FloatArray, Vertices, NodalValues


def exact_solution(vertices: Vertices) -> NodalValues:
    """The manufactured `u = prod_i sin(pi x_i)`, sampled at `vertices`.

    Zero on the boundary of the unit box in any dimension, so the Dirichlet data is
    homogeneous whether the box is 2D or 3D. In 2D it is `sin(pi x) sin(pi y)`.
    """
    return np.prod(np.sin(np.pi * vertices), axis=-1)


def exact_gradient(points: FloatArray) -> FloatArray:
    """`grad u` of the manufactured solution, sampled at `points`.

    Component `i` is `pi cos(pi x_i) prod_{j != i} sin(pi x_j)`. Broadcasts over any
    leading axes: `points` shaped `(..., d)` gives `(..., d)`, so it takes either the
    `(n_vertices, d)` nodes or the `(n_elements, n_qp, d)` quadrature points the H1
    error integrates over. The closed-form gradient makes the H1 error independent of
    the assembled stiffness (see `h1_seminorm_error`).
    """
    sines = np.sin(np.pi * points)
    cosines = np.cos(np.pi * points)
    dim = points.shape[-1]
    components = [
        np.pi * cosines[..., i] * np.prod(np.delete(sines, i, axis=-1), axis=-1)
        for i in range(dim)
    ]
    return np.stack(components, axis=-1)


def source_term(point: Vertices) -> list[FloatArray]:
    """`f = -laplacian(u) = dim * pi^2 * prod_i sin(pi x_i)`, the forcing that makes
    `exact_solution` the answer (`2 pi^2 u` in 2D, `3 pi^2 u` in 3D)."""
    dim = point.shape[1]
    return [dim * np.pi**2 * np.prod(np.sin(np.pi * point), axis=1)]


def l2_norm(space: FunctionSpace, values: NodalValues) -> float:
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
    space: FunctionSpace, u: NodalValues, exact_gradient: Callable[[FloatArray], FloatArray],
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
    varies within an element. A vector field passes `u` as `(n_nodes, n_components)` and
    `exact_gradient` returns the deformation gradient `F[c, i] = du_c/dx_i`; `quadrature_l2`
    takes the Frobenius norm over the component axes, so the same kernel serves both.
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
    dofs: NodalValues             # what the problem computed
    exact: NodalValues         # the manufactured solution at the same nodes
    l2_error: float            # ||u - exact||_L2, the number a study plots
    h1_error: float | None = None  # ||grad(u - exact)||_L2, where a closed-form gradient exists

    @property
    def pointwise_error(self) -> NodalValues:
        """`u - exact` node by node. `l2_error` is the norm of this field, not of
        these numbers; see `l2_norm`."""
        return self.dofs - self.exact


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


def solve_poisson_mms(n: int, dim: int = 2, backend: Backend | None = None) -> MMSSolve:
    """Solve the manufactured problem on an `n`-per-side unit box in `dim` dimensions.

    `backend` picks the linear solver (the default is the direct one), so the same
    study can measure an iterative backend's accuracy. `dim=3` runs the same scalar
    Poisson study on a tetrahedral box, the 3D analogue of the 2D rate.
    """
    mesh = box_mesh(corners=[[0.0] * dim, [1.0] * dim], resolution=(n,) * dim)

    bc = Conditions(Dirichlet(everywhere(), 0.0))
    problem = Poisson().problem(mesh, bc + Source(source_term))
    solution = problem.solve(backend=backend)
    u = solution.dofs

    exact = exact_solution(mesh.vertices)
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        dofs=u,
        exact=exact,
        l2_error=l2_norm(problem.space, u - exact),
        h1_error=h1_seminorm_error(problem.space, u, exact_gradient),
    )


def poisson_convergence(resolutions: tuple[int, ...], dim: int = 2) -> list[MMSSolve]:
    """Solve the manufactured problem once per resolution, coarsest first."""
    return [solve_poisson_mms(n, dim) for n in sorted(resolutions)]


# --- elasticity: the same idea for a vector unknown -----------------------------

# The manufactured displacement moves in x only:
#
#     2D:  u = (sin(pi x) sin(pi y), 0)
#     3D:  u = (sin(pi x) sin(pi y) sin(pi z), 0, 0)
#
# but the off-diagonal shear terms of sigma make every component of the forcing
# non-zero, so this exercises the coupled vector path rather than a scalar solve
# wearing two components. Asserted in tests/test_convergence.py.
ELASTIC_E, ELASTIC_NU = 200.0, 0.3


def elastic_source(point: Vertices, reduction: Reduction = 'plane_strain') -> list[FloatArray]:
    """The body force that makes `elastic_exact` the answer, for a 2D solid under
    `reduction` or a 3D one by the points' dimension.

    Plane stress is the same Navier operator with the plane-stress lambda, so one
    formula serves both reductions.
    """
    mu, lamb = LinearElasticMaterial(ELASTIC_E, ELASTIC_NU, reduction).in_plane_lame(point.shape[1])
    if point.shape[1] == 2:
        x, y = point.T
        return [
            np.pi**2 * (3*mu + lamb) * np.sin(np.pi * x) * np.sin(np.pi * y),
            -(mu + lamb) * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y),
        ]
    x, y, z = point.T
    return [
        np.pi**2 * (4*mu + lamb) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z),
        -(mu + lamb) * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z),
        -(mu + lamb) * np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z),
    ]


def elastic_exact(vertices: Vertices) -> FloatArray:
    """The manufactured displacement, `(n_vertices, dim)`."""
    exact = np.zeros_like(vertices, dtype=float)
    exact[:, 0] = np.prod(np.sin(np.pi * vertices), axis=1)
    return exact


def elastic_exact_gradient(points: FloatArray) -> FloatArray:
    """grad of the manufactured displacement: `F[c, i] = d u_c / d x_i`.

    Only component 0 moves (`u_0 = prod_j sin(pi x_j)`), so its row is the scalar
    `exact_gradient` and the rest are zero. Broadcasts over any leading axes:
    `(..., d)` points give `(..., d, d)`, so it takes the `(n_elements, n_qp, d)`
    quadrature points the H1 error integrates over. Closed form, so the seminorm error
    it feeds is independent of the assembled elastic stiffness (see `h1_seminorm_error`).
    """
    dim = points.shape[-1]
    grad = np.zeros(points.shape + (dim,))
    grad[..., 0, :] = exact_gradient(points)
    return grad


def solve_elastic_mms(n: int, dim: int = 2, backend: Backend | None = None,
                      reduction: Reduction = 'plane_strain') -> MMSSolve:
    """Solve the manufactured elasticity problem on an `n`-per-side unit box in `dim`
    dimensions, with `backend` picking the linear solver (the default is direct) and
    `reduction` the 2D model (plane strain or plane stress)."""
    mesh = box_mesh(corners=[[0.0] * dim, [1.0] * dim], resolution=(n,) * dim)

    bc = Conditions(Dirichlet(everywhere(), [0.0] * dim))
    equation = LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU, reduction=reduction)
    problem = equation.problem(mesh, bc + Source(lambda p: elastic_source(p, reduction)))
    return _elastic_mms_solve(n, problem, backend)


def _elastic_mms_solve(n: int, problem: LinearProblem, backend: Backend | None = None) -> MMSSolve:
    """Solve a P1 elastic `problem` whose answer is `elastic_exact` and measure its errors."""
    mesh = problem.space.mesh
    solution = problem.solve(backend=backend)

    exact = elastic_exact(mesh.vertices)
    # The space's mass matrix is the scalar one repeated per component, so this is
    # the true vector L2 norm rather than the norm of component 0.
    error = solution.dofs.reshape(exact.shape) - exact
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        dofs=solution.dofs,
        exact=exact.flatten(),
        l2_error=l2_norm(problem.space, error.flatten()),
        h1_error=h1_seminorm_error(problem.space, solution.dofs.reshape(exact.shape),
                                   elastic_exact_gradient),
    )


def elastic_convergence(resolutions: tuple[int, ...], dim: int = 2,
                        backend: Backend | None = None,
                        reduction: Reduction = 'plane_strain') -> list[MMSSolve]:
    """Solve the manufactured elasticity problem once per resolution, coarsest first."""
    return [solve_elastic_mms(n, dim, backend, reduction) for n in sorted(resolutions)]


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
# tests/test_convergence.py.


def variable_coefficient(point: Vertices) -> FloatArray:
    """kappa(x, y) = 1 + x + y: smooth and positive on the unit square."""
    return 1.0 + point[:, 0] + point[:, 1]


def variable_source(point: Vertices) -> list[FloatArray]:
    """f = -div(kappa grad u) for the kappa above and u = sin(pi x) sin(pi y)."""
    x, y = point[:, 0], point[:, 1]
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
        dofs=u,
        exact=exact,
        l2_error=l2_norm(space, u - exact),
    )


def variable_coefficient_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured variable-coefficient problem per resolution, coarsest first."""
    return [solve_variable_coefficient_mms(n) for n in sorted(resolutions)]


# --- inhomogeneous Neumann and Robin: the boundary-load quadrature in a rate ----
#
# Every study above is Dirichlet on the whole boundary, so a boundary-load quadrature
# wrong by a factor never shows up in a rate. Here the manufactured
#
#     u(x, y) = (1 + x) sin(pi y)      f = -laplacian(u) = pi^2 (1 + x) sin(pi y)
#
# is nonzero on the right edge and has a nonzero normal derivative on all four, and the
# boundary is all natural: Neumann (the flux du/dn) on the left, bottom, and top, and
# Robin (du/dn + kappa u = g) on the right, where u itself is nonzero so the Robin
# boundary-mass term carries real data too. No edge is Dirichlet; the kappa > 0 Robin
# term alone makes the form coercive, so the problem is well posed without one, and no
# node is both pinned and loaded. A boundary quadrature wrong by a constant factor
# breaks the O(h^2) rate, where the constant-solution and large-kappa checks in
# test_robin.py would not notice. Asserted in tests/test_convergence.py.

ROBIN_KAPPA = 3.0


def mixed_bc_exact(points: FloatArray) -> FloatArray:
    """u = (1 + x) sin(pi y), sampled at `points` (any leading axes)."""
    return (1.0 + points[..., 0]) * np.sin(np.pi * points[..., 1])


def mixed_bc_gradient(points: FloatArray) -> FloatArray:
    """grad u = (sin(pi y), pi (1 + x) cos(pi y)), broadcasting over any leading axes."""
    x, y = points[..., 0], points[..., 1]
    return np.stack([np.sin(np.pi * y), np.pi * (1.0 + x) * np.cos(np.pi * y)], axis=-1)


def mixed_bc_source(point: Vertices) -> list[FloatArray]:
    """f = -laplacian(u) = pi^2 (1 + x) sin(pi y): x is linear, so only the y curvature
    survives."""
    return [np.pi**2 * (1.0 + point[:, 0]) * np.sin(np.pi * point[:, 1])]


def solve_mixed_bc_mms(n: int) -> MMSSolve:
    """Solve the natural-boundary manufactured Poisson problem on an `n` x `n` grid.

    Neumann du/dn on the left, bottom, and top edges and Robin du/dn + kappa u = g on
    the right. Every edge carries nonzero data, so the boundary-load assembly and the
    Robin boundary-mass term both enter the error, and the Robin term makes the pure
    natural problem well posed.
    """
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    left, right = on_plane(0, 0.0), on_plane(0, 1.0)
    bottom, top = on_plane(1, 0.0), on_plane(1, 1.0)

    # The outward normal derivative on each natural edge. du/dx = sin(pi y);
    # du/dy = pi (1 + x) cos(pi y). Left (n = -x): -du/dx = -sin(pi y). Bottom (n = -y)
    # and top (n = +y): -du/dy and +du/dy both come to -pi (1 + x) (cos(pi) = -1 at the
    # top). The Robin g is du/dn + kappa u with du/dn = du/dx = sin(pi y) and
    # u = (1 + x) sin(pi y) on the right.
    def left_flux(p: Vertices) -> FloatArray:
        return -np.sin(np.pi * p[:, 1])

    def horizontal_flux(p: Vertices) -> FloatArray:
        return -np.pi * (1.0 + p[:, 0])

    def robin_g(p: Vertices) -> FloatArray:
        return (1.0 + ROBIN_KAPPA * (1.0 + p[:, 0])) * np.sin(np.pi * p[:, 1])

    bc = Conditions(
        Neumann(left, left_flux),
        Neumann(bottom, horizontal_flux),
        Neumann(top, horizontal_flux),
        Robin(right, kappa=ROBIN_KAPPA, g=robin_g),
    )
    problem = Poisson().problem(mesh, bc + Source(mixed_bc_source))
    solution = problem.solve()

    exact = mixed_bc_exact(mesh.vertices)
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        dofs=solution.dofs,
        exact=exact,
        l2_error=l2_norm(problem.space, solution.dofs - exact),
        h1_error=h1_seminorm_error(problem.space, solution.dofs, mixed_bc_gradient),
    )


def mixed_bc_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the mixed-boundary manufactured problem per resolution, coarsest first."""
    return [solve_mixed_bc_mms(n) for n in sorted(resolutions)]


# --- P2 elements: the same Poisson problem, one polynomial degree higher ---------
#
# The same manufactured u = sin(pi x) sin(pi y), solved on the quadratic space. The
# only differences from solve_poisson_mms are the element type and that the exact
# solution and the error norm are sampled at all the P2 nodes (corners and edge
# midpoints) since the extra DOFs live there. P2 is O(h^3) in L2 where P1 is O(h^2),
# which tests/test_convergence.py asserts.


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
        dofs=u,
        exact=exact,
        # The P2 gradient is linear within an element; sample the seminorm error at a
        # degree-4 rule so the norm's own quadrature error stays below the O(h^2) rate.
        l2_error=l2_norm(space, u - exact),
        h1_error=h1_seminorm_error(space, u, exact_gradient, degree=4),
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
    load = Source(elastic_source, quadrature_degree=4)
    problem = LinearProblem(space, operator, bc + load)
    u = LinearSolve().solve(problem)

    exact = elastic_exact(space.node_coords)   # (n_nodes, 2)
    error = u.reshape(exact.shape) - exact
    return MMSSolve(
        h=1.0 / (n - 1),
        mesh=mesh,
        dofs=u,
        exact=exact.flatten(),
        l2_error=l2_norm(space, error.flatten()),
        h1_error=h1_seminorm_error(space, u.reshape(exact.shape), elastic_exact_gradient, degree=4),
    )


def elastic_p2_convergence(resolutions: tuple[int, ...]) -> list[MMSSolve]:
    """Solve the manufactured elasticity problem on P2 per resolution, coarsest first."""
    return [solve_elastic_mms_p2(n) for n in sorted(resolutions)]


# --- thermoelasticity: the elastic study under a manufactured temperature -------
#
# The same manufactured displacement, now with a temperature rise
#
#     dT(x, y) = sin(pi x) sin(pi y)
#
# entering through the thermal strain alpha dT I. The law is sigma = D eps - beta dT I
# with beta = (3 lambda + 2 mu) alpha, so the body force that keeps `elastic_exact` the
# answer gains the gradient of the thermal stress:
#
#     f = -div(sigma) = elastic_source + beta grad(dT)
#
# Both the thermal load and the corrected stress enter the rate. alpha is picked for
# convenience (the thermal stress of the same order as the mechanical one), not
# physics. Asserted in tests/test_convergence.py twice: with dT sampled at the
# quadrature points, and handed over as its nodal interpolant on the same mesh, the
# way a heat solve's `NodalField` is.

THERMAL_ALPHA = 0.5


def thermal_field(point: Vertices) -> FloatArray:
    """dT = sin(pi x) sin(pi y), the same shape as the Poisson solution, at `point`."""
    return exact_solution(point)


def thermal_stress_modulus() -> float:
    """beta = (3 lambda + 2 mu) alpha for the study's material."""
    mu, lamb = Enu_to_Lame(ELASTIC_E, ELASTIC_NU)
    return (3 * lamb + 2 * mu) * THERMAL_ALPHA


def thermoelastic_source(point: Vertices) -> list[FloatArray]:
    """The body force that makes `elastic_exact` the answer under the thermal strain:
    the elastic forcing plus `beta grad(dT)`."""
    mechanical = elastic_source(point)
    thermal = thermal_stress_modulus() * exact_gradient(point)   # (n, 2)
    return [mechanical[0] + thermal[:, 0], mechanical[1] + thermal[:, 1]]


def solve_thermoelastic_mms(n: int, nodal: bool = False) -> MMSSolve:
    """Solve the manufactured thermoelastic problem on an `n` x `n` grid.

    With `nodal`, the temperature is handed over as its P1 interpolant on the mesh (a
    `NodalField`, as a heat solution would be) rather than sampled at the quadrature
    points; the interpolation error is O(h^2), so the rate is the same.
    """
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    temperature = (FunctionSpace(mesh, n_components=1).interpolate(thermal_field)
                   if nodal else thermal_field)
    thermal = ThermalStrain(THERMAL_ALPHA, temperature)

    bc = Conditions(Dirichlet(everywhere(), [0.0, 0.0]))
    equation = LinearElastic(E=ELASTIC_E, nu=ELASTIC_NU, thermal=thermal)
    return _elastic_mms_solve(n, equation.problem(mesh, bc + Source(thermoelastic_source)))


def thermoelastic_convergence(resolutions: tuple[int, ...], nodal: bool = False) -> list[MMSSolve]:
    """Solve the manufactured thermoelastic problem per resolution, coarsest first."""
    return [solve_thermoelastic_mms(n, nodal) for n in sorted(resolutions)]


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


def annulus_mesh(
    inner_radius: float, outer_radius: float, n_radial: int, n_theta: int,
) -> Mesh:
    """Structured triangle mesh of the annulus about the origin, with its rims attached
    as `Circle`s.

    `n_radial` nodes across the radial direction and `n_theta` sectors around. The
    inner and outer boundary facets carry a `Circle`, so a curved space places their
    midside nodes on the true rim rather than at the chord midpoint. Structured so the
    curved convergence study refines uniformly; an `Outline` of two `Circle`s would
    mesh the same domain unstructured.
    """
    rings = np.arange(n_radial)
    radii = inner_radius + (outer_radius - inner_radius) * (rings / (n_radial - 1))
    thetas = 2 * np.pi * np.arange(n_theta) / n_theta
    r, t = np.meshgrid(radii, thetas, indexing="ij")
    vertices = np.column_stack([(r * np.cos(t)).ravel(), (r * np.sin(t)).ravel()])

    def node(ring: int, sector: int) -> int:
        return ring * n_theta + sector % n_theta

    elements = []
    for ring in range(n_radial - 1):
        for sector in range(n_theta):
            a, b = node(ring, sector), node(ring, sector + 1)
            c, d = node(ring + 1, sector + 1), node(ring + 1, sector)
            elements.extend([[a, b, c], [a, c, d]])
    elements = np.array(elements)

    boundary = boundary_facets(elements)
    inner_curve = Circle([0.0, 0.0], inner_radius)
    outer_curve = Circle([0.0, 0.0], outer_radius)
    midradius = 0.5 * (inner_radius + outer_radius)
    boundary_curves = [
        inner_curve if float(np.hypot(*vertices[facet[0]])) < midradius else outer_curve
        for facet in boundary
    ]
    return Mesh(vertices, elements, boundary, boundary_curves)


def annulus_exact(points: FloatArray) -> FloatArray:
    """The manufactured u = sin(x) sin(y), sampled at `points` (any leading axes)."""
    return np.sin(points[..., 0]) * np.sin(points[..., 1])


def annulus_gradient(points: FloatArray) -> FloatArray:
    """grad u of the manufactured solution, broadcasting over any leading axes."""
    x, y = points[..., 0], points[..., 1]
    return np.stack([np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)], axis=-1)


def annulus_source(point: Vertices) -> list[FloatArray]:
    """f = -laplacian(u) = 2 sin(x) sin(y)."""
    return [2.0 * np.sin(point[:, 0]) * np.sin(point[:, 1])]


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

    bc = Conditions(Dirichlet(everywhere(), annulus_exact))
    problem = Poisson().problem(mesh, bc + Source(annulus_source), element_type=element_type)
    solution = problem.solve()
    space = problem.space

    exact = annulus_exact(space.node_coords)
    return MMSSolve(
        h=(ANNULUS_OUTER - ANNULUS_INNER) / (n - 1),
        mesh=mesh,
        dofs=solution.dofs,
        exact=exact,
        l2_error=l2_norm(space, solution.dofs - exact),
        h1_error=h1_seminorm_error(space, solution.dofs, annulus_gradient, degree=4),
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


def oscillatory_exact(vertices: Vertices) -> NodalValues:
    """u = sin(k pi x) sin(k pi y): the manufactured solution, zero on the boundary."""
    k = LOAD_MMS_FREQUENCY
    x, y = vertices[:, 0], vertices[:, 1]
    return np.sin(k * np.pi * x) * np.sin(k * np.pi * y)


def oscillatory_source(point: Vertices) -> list[FloatArray]:
    """f = -laplacian(u) = 2 (k pi)^2 sin(k pi x) sin(k pi y)."""
    k = LOAD_MMS_FREQUENCY
    return [2 * (k * np.pi) ** 2 * np.sin(k * np.pi * point[:, 0]) * np.sin(k * np.pi * point[:, 1])]


@dataclass
class LoadComparison:
    """One solve of the oscillatory problem with each kind of load, and how far off."""
    n: int                    # grid resolution (n x n nodes), for slicing a row of it
    h: float
    mesh: Mesh
    exact: NodalValues
    nodal: NodalValues        # source integrated as its P1 interpolant
    sampled: NodalValues      # source sampled at the quadrature points (Source)
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
        run = ThetaMethod(dt=T / steps, steps=steps, theta=theta).solve(
            problem, initial=Initial(NodalField(problem.space, u0)))
        u_h = run.dofs[-1]
        errors.append(l2_norm(problem.space, u_h - reference))

    return ConvergenceStudy(np.array([T / k for k in sorted(step_counts)]), np.array(errors))
