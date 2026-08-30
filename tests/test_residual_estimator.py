"""The residual error estimator: what it measures, on P1 and P2.

The shared contract (shape, the patch test, the peak, the refinement loop) lives in
`test_estimator_contract.py`. Here: the flux it jumps is kappa grad u, not grad u; on P2
the interior term carries div(flux), read from the P2 Hessians; and the elastic boundary
term matches a hand derivation on a two-triangle square.
"""
import numpy as np
import pytest

from fem.analysis.adaptivity import AdaptiveRefinement
from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from mms import exact_gradient, h1_seminorm_error, source_term
from fem.elements import IsoparametricTriangleElement, QuadraticTriangleElement
from fem.physics.energies import StVenantKirchhoff
from fem.physics.equations import LinearElastic, Poisson
from fem.analysis.estimators import ResidualEstimator
from fem.physics.forms import DiffusionForm, EnergyForm, LinearElasticForm
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.mesh.structured import box_mesh
from fem.physics.derived import GradientFlux
from fem.problem import LinearProblem
from fem.regions import everywhere, on_plane
from fem.post.solution import DiffusionSolution, ElasticSolution
from fem.space import FunctionSpace
from fem.loads import Source
from helpers import cantilever_bc, global_estimate, pinned, problem_for, solved, two_triangle_square


# -- the diffusion coefficient in the flux ------------------------------------
#
# The flux the estimator jumps and differentiates is kappa grad u, not grad u. Each test
# below states a problem whose exact solution the space reproduces, so the estimate
# should vanish; reading grad u alone would leave a spurious residual behind.


def test_affine_coefficient_exact_p1_solution_is_quiet(make_unit_square):
    """kappa = 1 + x, u = x: the flux (1 + x) is continuous, and its divergence, 1,
    cancels the source f = -1. Reading grad u alone would report f + laplacian(u) = -1."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(everywhere(), lambda p: p[:, 0]), Source(-1.0))
    problem, solution = solved(Poisson(coefficient=lambda p: 1.0 + p[:, 0]), mesh, bc)
    np.testing.assert_allclose(solution.dofs, mesh.vertices[:, 0], atol=1e-12)

    eta = ResidualEstimator().estimate(problem, solution)
    assert np.all(eta < 1e-10)


def test_affine_coefficient_exact_p2_solution_is_quiet(make_unit_square):
    """kappa = 1 + x, u = x^2 on P2: div(kappa grad u) = 2 + 4x needs both the
    kappa laplacian(u) and the grad(kappa) . grad(u) terms; the source f = -(2 + 4x)
    cancels it exactly, and the flux 2x(1 + x) is continuous across every edge."""
    mesh = make_unit_square(4)
    space = FunctionSpace(mesh, QuadraticTriangleElement)
    # An affine kappa against P2 gradients, and an affine source against a P2 test
    # function, are cubic integrands: raise both rules so the discrete solution is the
    # exact u = x^2.
    bc = Conditions(
        Dirichlet(everywhere(), lambda p: p[:, 0]**2),
        Source(lambda p: -(2.0 + 4.0 * p[:, 0]), quadrature_degree=4),
    )
    problem = LinearProblem(space, DiffusionForm(lambda p: 1.0 + p[:, 0], rule_degree=4), bc)
    solution = problem.solve()
    np.testing.assert_allclose(solution.dofs, space.node_coords[:, 0]**2, atol=1e-10)

    eta = ResidualEstimator().estimate(problem, solution)
    assert np.all(eta < 1e-9)


def test_constant_coefficient_scales_the_estimate(make_unit_square):
    """Scaling kappa and the source together leaves u unchanged and scales every
    residual, so the estimate scales with them."""
    mesh = make_unit_square(8)

    def source(p):
        return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    unit, u_unit = solved(Poisson(), mesh, pinned() + Source(source))
    scaled, u_scaled = solved(Poisson(coefficient=3.0), mesh, pinned() + Source(lambda p: 3.0 * source(p)))
    np.testing.assert_allclose(u_scaled.dofs, u_unit.dofs, atol=1e-12)

    eta_unit = ResidualEstimator().estimate(unit, u_unit)
    eta_scaled = ResidualEstimator().estimate(scaled, u_scaled)
    np.testing.assert_allclose(eta_scaled, 3.0 * eta_unit, rtol=1e-10)


def test_scaled_form_scales_its_flux(make_unit_square):
    """`3 * DiffusionForm()` and `DiffusionForm(3.0)` are the same operator, so their
    estimates agree: the scaled form's flux carries the factor."""
    mesh = make_unit_square(6)
    bc = pinned() + Source(lambda p: p[:, 0] * p[:, 1])
    space = FunctionSpace(mesh)
    by_coefficient = LinearProblem(space, DiffusionForm(3.0), bc)
    by_factor = LinearProblem(space, 3.0 * DiffusionForm(), bc)
    solution = by_coefficient.solve()
    np.testing.assert_allclose(by_factor.solve().dofs, solution.dofs, atol=1e-12)

    eta_coefficient = ResidualEstimator().estimate(by_coefficient, solution)
    eta_factor = ResidualEstimator().estimate(by_factor, solution)
    np.testing.assert_allclose(eta_factor, eta_coefficient, rtol=1e-10)
    assert eta_coefficient.max() > 1e-3


def test_neumann_edge_registers_a_missed_flux(make_unit_square):
    """u = x with kappa = 2: the outward flux on x = 1 is 2. The matching Neumann value
    is quiet; a mismatched one shows up on the elements along that edge only."""
    mesh = make_unit_square(6)
    kappa = 2.0
    left = Conditions(Dirichlet(on_plane(0, 0.0), 0.0))

    matched, solution = solved(Poisson(coefficient=kappa), mesh, left + Neumann(on_plane(0, 1.0), kappa))
    np.testing.assert_allclose(solution.dofs, mesh.vertices[:, 0], atol=1e-12)
    eta = ResidualEstimator().estimate(matched, solution)
    # `g` is nodal, so at the corners (1, 0) and (1, 1) it also reaches the flux-free top
    # and bottom edges through the shared vertex (the elastic test below works the same
    # effect by hand); every other element is quiet.
    corner = np.any(np.isclose(mesh.vertices[mesh.elements][:, :, 0], 1.0)
                    & np.isin(mesh.vertices[mesh.elements][:, :, 1], [0.0, 1.0]), axis=1)
    assert np.all(eta[~corner] < 1e-10)

    mismatched = Poisson(coefficient=kappa).problem(mesh, left + Neumann(on_plane(0, 1.0), 0.5 * kappa))
    eta = ResidualEstimator().estimate(mismatched, mismatched.solution(mesh.vertices[:, 0]))
    on_right = np.any(np.isclose(mesh.vertices[mesh.elements][:, :, 0], 1.0), axis=1)
    # A boundary element touches x = 1 along an edge, not just at a corner, when two
    # of its vertices lie on it.
    right_edge = np.sum(np.isclose(mesh.vertices[mesh.elements][:, :, 0], 1.0), axis=1) == 2
    assert np.all(eta[right_edge] > 1e-3)
    assert np.all(eta[~on_right] < 1e-10)


# -- the P2 interior term: div(flux) from the P2 Hessians ---------------------


def _p2_solution(n, source=None):
    mesh = box_mesh(corners=[[0, 0], [1, 1]], resolution=(n, n))
    bc = pinned() if source is None else pinned() + source
    return solved(Poisson(), mesh, bc, element_type=QuadraticTriangleElement)


def test_p2_poisson_divergence_is_the_laplacian():
    """div(grad u) = laplacian(u). For u = x^2 + 2 y^2 that is a constant 6, recovered
    per element from the P2 field."""
    mesh = box_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 4))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=1)
    x, y = space.node_coords[:, 0], space.node_coords[:, 1]
    u = x**2 + 2 * y**2
    solution = DiffusionSolution(space, u, gradient=np.zeros((len(mesh.elements), 2)))

    div = DiffusionForm().flux().divergence(solution)
    assert np.allclose(div, 6.0)


def test_p2_elastic_divergence_is_the_navier_operator():
    """div(sigma) in Navier form (lambda + mu) grad(div u) + mu laplacian(u). For
    u = (x^2, 0) that is (2 lambda + 4 mu, 0), the exact strong-form residual the
    interior term needs."""
    E, nu = 200.0, 0.3
    mu, lamb = Enu_to_Lame(E, nu)
    mesh = box_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 4))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)
    u = np.zeros((space.n_nodes, 2))
    u[:, 0] = space.node_coords[:, 0]**2
    n_el = len(mesh.elements)
    solution = ElasticSolution(space, u.ravel(), strain=np.zeros((n_el, 3, 3)),
                               stress=np.zeros((n_el, 3, 3)), compliance=np.zeros(n_el))

    div = LinearElasticForm(LinearElasticMaterial(E, nu)).flux().divergence(solution)
    assert np.allclose(div, [2 * lamb + 4 * mu, 0.0])


def test_stress_divergence_refuses_a_finite_strain_form():
    """The divergence is the small-strain Navier operator, so a residual estimate of a
    finite-strain solve is refused rather than computed with the wrong operator."""
    E, nu = 200.0, 0.3
    mesh = box_mesh(corners=[[0, 0], [2, 1]], resolution=(5, 4))
    space = FunctionSpace(mesh, QuadraticTriangleElement, n_components=2)
    n_el = len(mesh.elements)
    solution = ElasticSolution(space, np.zeros(space.n_dofs), strain=np.zeros((n_el, 3, 3)),
                               stress=np.zeros((n_el, 3, 3)), compliance=np.zeros(n_el))

    field = EnergyForm(StVenantKirchhoff(E, nu)).flux()
    with pytest.raises(NotImplementedError, match='small-strain'):
        field.divergence(solution)


def test_p2_interior_term_is_active():
    """A guard that the interior div term is really carried: the P2 Laplacian of a
    non-harmonic solve is not identically zero, so dropping it would change the estimate."""
    _, solution = _p2_solution(8, Source(source_term))
    laplacian = GradientFlux().divergence(solution)
    assert np.abs(laplacian).max() > 1.0


def test_p2_reliability_is_bounded_and_stable():
    """The estimate stays a faithful upper bound on the true error across a P2 refinement
    sequence: the effectivity index is bounded and does not drift up. A residual estimator
    over-estimates (its constant is looser than a recovery estimator's), so this checks
    boundedness and stability, not that the index tends to 1."""
    indices = []
    for n in (6, 11, 21):
        problem, solution = _p2_solution(n, Source(source_term))
        eta = global_estimate(ResidualEstimator().estimate(problem, solution))
        true_error = h1_seminorm_error(problem.space, solution.dofs, exact_gradient)
        indices.append(eta / true_error)

    assert all(1.0 < i < 15.0 for i in indices)
    assert indices[-1] <= indices[0] + 1e-9        # does not drift upward under refinement


def test_refuses_curved_elements(make_unit_square):
    """The interior term's divergence assumes a constant Jacobian, so a curved element
    is refused."""
    problem, solution = solved(Poisson(), make_unit_square(4), pinned() + Source(1.0),
                               element_type=IsoparametricTriangleElement)
    with pytest.raises(NotImplementedError, match='RecoveryEstimator'):
        ResidualEstimator().estimate(problem, solution)


# -- the elastic boundary term ------------------------------------------------
#
# An error estimator can be tested two ways, and these tests use the second:
#
#   1. Solve a real problem and check refinement lands where the error should
#      be. This is what the contract's Poisson tests do -- a peaked source pulls
#      refinement toward the centre.
#   2. Feed in a stress by hand and check eta equals the number the formula
#      gives for it.
#
# Way 1 is unreliable for elasticity: the error has no single home. A clamped
# edge and the corners are real sources of error competing with the hole rim,
# and which one dominates depends on mesh resolution and refinement budget. An
# assertion about *where* refinement lands would test this problem's physics,
# not whether the formula is implemented correctly.
#
# Way 2 works because the estimator never solves: it reads the problem's space and
# resolved conditions and the solution's stress. So these tests state the problem on a
# hand-sized mesh, pair it with a synthetic solution carrying a known constant stress
# (`_constant_stress`), and check eta against a value worked out by hand.

def _constant_stress(problem, sigma_xx, sigma_yy, sigma_xy):
    """A synthetic ElasticSolution carrying the same constant in-plane stress on every
    element of `problem`'s space."""
    space = problem.space
    n = len(space.mesh.elements)
    stress = np.zeros((n, 3, 3))
    stress[:, 0, 0] = sigma_xx
    stress[:, 1, 1] = sigma_yy
    stress[:, 0, 1] = stress[:, 1, 0] = sigma_xy
    return ElasticSolution(
        space, dofs=np.zeros(space.n_dofs),
        strain=np.zeros((n, 3, 3)), stress=stress, compliance=np.zeros(n),
    )


def test_elastic_linear_solution_has_small_jumps(make_unit_square):
    """A globally linear displacement is constant-strain, an exact equilibrium solution with
    zero body force, so eta vanishes. Dirichlet everywhere, so only the interior and
    jump terms are exercised."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(everywhere(), lambda p: [0.01 * p[:, 0], -0.003 * p[:, 1]]))
    problem, solution = solved(LinearElastic(E=200, nu=0.3), mesh, bc)

    eta = ResidualEstimator().estimate(problem, solution)

    assert np.all(eta < 1e-8)


def test_elastic_boundary_term_matches_hand_derivation():
    """The Neumann/natural boundary residual against a hand derivation: the bottom edge is
    Dirichlet and skipped, the other three are natural with g=0, so each free edge's
    residual is ||sigma.n||^2 for the injected stress."""
    mesh = two_triangle_square()
    bc = Conditions(Dirichlet(on_plane(1, 0.0), [0, 0]))
    problem = LinearElastic(E=200, nu=0.3).problem(mesh, bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    eta = ResidualEstimator().estimate(problem, _constant_stress(problem, Sxx, Syy, Sxy))

    h_K = np.sqrt(2.0)
    right_residual = Sxx**2 + Sxy**2          # element 0: bottom (skip) + right
    left_residual = Sxx**2 + Sxy**2           # element 1: top + left
    top_residual = Sxy**2 + Syy**2
    expected_e0 = np.sqrt(h_K * right_residual)
    expected_e1 = np.sqrt(h_K * (top_residual + left_residual))

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_elastic_neumann_matching_traction_is_quiet():
    """A Neumann condition that exactly matches the injected stress's traction contributes
    nothing on its edge. `g` is nodal, so it also reaches the neighbouring free edges
    through the shared corner; that is worked through by hand."""
    mesh = two_triangle_square()
    bc = Conditions(
        Dirichlet(on_plane(1, 0.0), [0, 0]),
        Neumann(on_plane(1, 1.0), [0.5, 1.0]),
    )
    problem = LinearElastic(E=200, nu=0.3).problem(mesh, bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    eta = ResidualEstimator().estimate(problem, _constant_stress(problem, Sxx, Syy, Sxy))

    h_K = np.sqrt(2.0)
    # top's g=[0.5, 1.0] is shared by nodes 2 and 3; right (1,2) and left (0,3)
    # each average one matched endpoint (g=[0.5,1.0]) with one untouched
    # endpoint (g=[0,0]), giving g=[0.25, 0.5] on both. top itself matches
    # sigma.n exactly and drops out.
    g = np.array([0.25, 0.5])
    right_residual = np.sum((g - np.array([Sxx, Sxy]))**2)   # sigma.n on (1,0)
    left_residual = np.sum((g - np.array([-Sxx, -Sxy]))**2)  # sigma.n on (-1,0)
    expected_e0 = np.sqrt(h_K * right_residual)
    expected_e1 = np.sqrt(h_K * left_residual)

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_elastic_roller_edge_only_tests_its_free_component():
    """On a roller (bottom pinned in x, free in y) only the free component is tested; the
    pinned direction's reaction stress must not count as error."""
    mesh = two_triangle_square()
    bc = Conditions(Dirichlet(on_plane(1, 0.0), [0, None]))
    problem = LinearElastic(E=200, nu=0.3).problem(mesh, bc)

    Sxx, Syy, Sxy = 3.0, 1.0, 0.5
    eta = ResidualEstimator().estimate(problem, _constant_stress(problem, Sxx, Syy, Sxy))

    h_K = np.sqrt(2.0)
    # bottom's outward normal is (0, -1), so sigma.n = (-Sxy, -Syy); only the
    # free y-component (residual Syy^2, g=0) counts, not x (-Sxy is pinned
    # reaction stress, not error).
    bottom_residual = Syy**2
    right_residual = Sxx**2 + Sxy**2   # untouched: same free edge as before
    top_residual = Sxy**2 + Syy**2     # untouched: bottom's roller doesn't reach it
    left_residual = Sxx**2 + Sxy**2    # untouched
    expected_e0 = np.sqrt(h_K * (bottom_residual + right_residual))
    expected_e1 = np.sqrt(h_K * (top_residual + left_residual))

    assert np.allclose(eta, [expected_e0, expected_e1])


def test_elastic_refinement_keeps_the_clamp_exact(make_unit_square):
    """The full elastic loop: the mesh grows, the solution stays finite, and the geometric
    Dirichlet conditions still hold exactly on the refined mesh."""
    mesh = make_unit_square(6)
    bc = cantilever_bc(traction=(1.0, 0.0))

    n_before = len(mesh.elements)
    driver = AdaptiveRefinement(mesh, problem_for(LinearElastic(E=200, nu=0.3), bc), ResidualEstimator(),
                                max_triangles=n_before + 200, max_iters=5)
    solution = driver.run()

    final = driver.mesh
    assert len(final.elements) > n_before
    assert solution.mesh is final
    assert np.all(np.isfinite(solution.dofs))

    left = np.flatnonzero(np.abs(final.vertices[:, 0]) < 1e-12)
    assert np.allclose(solution.nodal_values[left], 0.0, atol=1e-12)
