"""How the two elasticity paths relate to each other.

Elasticity is solved along two independent axes:

    strain measure   Green-Lagrange  S = 1/2 (F^T F - I)       exact
                     small strain    eps = 1/2 (grad u + grad u^T)

    method           direct assembly      K u = b, one linear solve
                     energy minimization  Newton on grad(Pi) = 0

`LinearElastic` picks the strain measure; the solve strategy picks the method. A
small-strain `EnergyForm` fills the off-diagonal cell, so each axis can be varied alone.
"""
import numpy as np
import pytest

from fem.backends import MinresBackend
from fem.boundary import BoundaryConditions, Dirichlet
from fem.materials import LinearElasticMaterial
from fem.problem import Problem
from fem.regions import on_plane
from fem.equations import LinearElastic, FiniteStrainElastic
from fem.solver import Solver
from fem.solve import BacktrackingLineSearch, NewtonSolve, TangentRegularization
from fem.energies import SmallStrain, StVenantKirchhoff
from fem.forms import EnergyForm
from fem.numerics import central_difference_order


def test_hooke_matrix_is_the_second_derivative_of_the_small_strain_energy():
    """`Material`'s D and `SmallStrain`'s W are one material, D = d2W/de2: for any strain,
    1/2 eps_v^T D eps_v equals W(eps), shear terms included. Checked in 2D, the only
    dimension where both representations exist."""
    E, nu = 200.0, 0.3
    D = LinearElasticMaterial(E, nu).constitutive_matrices(reference_dim=2, n_elements=1)[0]
    density = SmallStrain(E, nu)

    rng = np.random.default_rng(0)
    for _ in range(8):
        strain = rng.normal(size=(2, 2))
        strain = 0.5 * (strain + strain.T)  # a strain tensor is symmetric
        strain_voigt = np.array([strain[0, 0], strain[1, 1], 2 * strain[0, 1]])

        energy_from_D = 0.5 * strain_voigt @ D @ strain_voigt
        energy_from_W = density.calculate_W_from_S(strain)
        assert energy_from_D == pytest.approx(energy_from_W)


def _stretched_square(make_unit_square, stretch=0.1, n=8):
    """Unit square, left edge pinned, right edge displaced by `stretch` in x. Displacement
    driven with no load, the only setup both solvers can be compared on."""
    mesh = make_unit_square(n)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [stretch, 0]),
    )
    return mesh, bc


def _energy_problem(mesh, bc, model):
    """The energy-minimisation statement of an elastic `model` (`LinearElastic` or
    `FiniteStrainElastic`): its density under an `EnergyForm`."""
    equation = model(E=200, nu=0.4)
    return Problem(equation.space(mesh), EnergyForm(equation.energy_density()), bc=bc)


def _minimise(problem):
    return NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)


def _one_newton_step(problem):
    """Displacement after a single Newton step from the zero initial guess."""
    return NewtonSolve(max_iters=1).solve(problem)


def test_line_search_converges_from_a_seed_a_full_step_diverges_from(make_unit_square):
    """Under strong compression (the right edge pushed 70% through the block) the St-VK
    tangent loses ellipticity and a full Newton step from the zero seed diverges;
    backtracking on the energy reaches equilibrium."""
    mesh = make_unit_square(8)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [-0.7, 0]),
    )
    equation = FiniteStrainElastic(E=200, nu=0.4)
    problem = equation.problem(mesh, bc)
    free = problem.constraints[0]

    def free_residual(line_search):
        u = NewtonSolve(max_iters=50, line_search=line_search).solve(problem)
        return float(np.linalg.norm(problem.residual(u)[free]))

    r_full = free_residual(None)
    r_searched = free_residual(BacktrackingLineSearch())

    # The point is that the line search reaches equilibrium at all where the full step
    # blows up (below versus O(10)). The residual bound is deliberately loose: the tangent
    # is near-singular here, so the residual at the minimum amplifies floating-point noise
    # in the displacement and is not a stable polish target.
    assert r_searched < 1e-3, f"line search should converge, got residual {r_searched:.2e}"
    assert not np.isfinite(r_full) or r_full > 1e-2, (
        f"the full step should fail to converge here, got residual {r_full:.2e}"
    )


def test_each_elastic_model_names_its_energy_density():
    """`LinearElastic` minimises the small-strain energy, `FiniteStrainElastic` the
    St-Venant-Kirchhoff one by default, and a `law` chooses another."""
    assert isinstance(LinearElastic(200, 0.4).energy_density(), SmallStrain)
    # SmallStrain subclasses StVenantKirchhoff, so pin the finite member by exact type.
    assert type(FiniteStrainElastic(200, 0.4).energy_density()) is StVenantKirchhoff
    assert type(FiniteStrainElastic(200, 0.4, law=SmallStrain).energy_density()) is SmallStrain


def test_finite_strain_solve_matches_recorded_solution(make_unit_square):
    """Regression pin on the St Venant-Kirchhoff answer: values recorded from the
    implementation, so this catches drift rather than proving correctness."""
    mesh, bc = _stretched_square(make_unit_square)
    equation = FiniteStrainElastic(E=200, nu=0.4)
    solver = Solver(mesh, equation, bc)
    u = solver.solve().u

    np.testing.assert_allclose(np.linalg.norm(u), 0.503442620332, rtol=1e-9)
    np.testing.assert_allclose(u.max(), 0.1, rtol=1e-12)
    np.testing.assert_allclose(u.min(), -0.037995668257, rtol=1e-9)
    np.testing.assert_allclose(solver.problem().energy(u.copy()), 1.590561321584, rtol=1e-9)


def test_residual_and_tangent_are_consistent_by_finite_difference(make_unit_square):
    """`residual` is the gradient of `energy`, and `tangent` the gradient of `residual`,
    each to O(eps^2) under central differences. This checks the assembly (scatter,
    transposes), which a density-level check cannot see. St-VK is used because its
    quartic energy gives a clean O(eps^2) slope."""
    mesh = make_unit_square(5)
    equation = FiniteStrainElastic(E=200, nu=0.4)
    # No BCs: energy, residual, and tangent are the raw, unconstrained quantities,
    # evaluated at an imposed state rather than a solve.
    problem = equation.problem(mesh)

    # A non-trivial state, so the nonlinearity is active: at u = 0 the tangent is the
    # small-strain one and the cubic term this check leans on would vanish.
    rng = np.random.default_rng(0)
    u = 0.1 * rng.standard_normal(problem.space.n_dofs)
    gradient = problem.residual(u)
    hessian = problem.tangent(u)

    grad_order = central_difference_order(problem.energy, lambda d: gradient @ d, u)
    hess_order = central_difference_order(problem.residual, lambda d: hessian @ d, u)

    assert 1.9 < grad_order < 2.1, f"residual disagrees with d(energy): order {grad_order:.3f}"
    assert 1.9 < hess_order < 2.1, f"tangent disagrees with d(residual): order {hess_order:.3f}"


def test_small_strain_energy_equals_direct_solve(make_unit_square):
    """Energy minimisation with small strain reproduces `Solver`'s direct assembly exactly,
    and in a single Newton step: a quadratic energy has an affine gradient."""
    mesh, bc = _stretched_square(make_unit_square)

    u_direct = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve().u.flatten()
    u_energy = _one_newton_step(_energy_problem(mesh, bc, LinearElastic))

    np.testing.assert_allclose(u_energy, u_direct, atol=1e-12)


def test_stvk_needs_more_than_one_newton_step(make_unit_square):
    """St-VK is nonlinear in u, so one Newton step leaves a residual."""
    mesh, bc = _stretched_square(make_unit_square)
    equation = FiniteStrainElastic(E=200, nu=0.4)

    u_one = _one_newton_step(equation.problem(mesh, bc))
    u_converged = Solver(mesh, equation, bc).solve().u

    rel = np.linalg.norm(u_one - u_converged) / np.linalg.norm(u_converged)
    assert rel > 0.1, f"one step should be far from converged, got rel={rel:.2e}"


def test_models_agree_to_second_order_in_strain(make_unit_square):
    """At small strain the two measures agree to O(||grad u||^2): halving the imposed
    stretch shrinks the displacement gap by ~4x."""
    gaps = []
    for stretch in (0.08, 0.04, 0.02, 0.01):
        mesh, bc = _stretched_square(make_unit_square, stretch=stretch)
        u_small = _minimise(_energy_problem(mesh, bc, LinearElastic))
        u_stvk = _minimise(_energy_problem(mesh, bc, FiniteStrainElastic))
        gaps.append(np.linalg.norm(u_small - u_stvk))

    ratios = [a / b for a, b in zip(gaps[:-1], gaps[1:])]
    # Quadratic gap -> 4x per halving. Loose bounds: the far field is not purely
    # asymptotic and the mesh is coarse, but the trend must be unambiguously ~4.
    for r in ratios:
        assert 3.5 < r < 4.5, f"gap ratio {r:.2f} is not the ~4x of a quadratic difference"


def test_green_lagrange_is_frame_indifferent(make_unit_square):
    """A rigid rotation is strain-free under Green-Lagrange (S = 1/2 (R^T R - I) = 0),
    while small strain reads a spurious compression eps = (cos theta - 1) I. Evaluated
    directly on the rotation field, with no solve."""
    mesh = make_unit_square(8)
    center = mesh.vertices.mean(axis=0)
    space = LinearElastic(E=200, nu=0.4).space(mesh)

    def total_energy(density, u_nodal):
        return space.total_energy(EnergyForm(density), u_nodal.flatten())

    def rotation_field(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (mesh.vertices - center) @ R.T + center - mesh.vertices

    small_energies = []
    for theta in (0.4, 0.2, 0.1):
        u = rotation_field(theta)
        stvk = total_energy(StVenantKirchhoff(200, 0.4), u)
        small = total_energy(SmallStrain(200, 0.4), u)
        assert stvk < 1e-18, f"Green-Lagrange stored {stvk:.2e} under a rigid rotation"
        assert small > 1e-6, f"small strain should read a spurious {theta} rotation as strain"
        small_energies.append(small)

    # Spurious strain ~ theta^2, energy quadratic in strain -> ~theta^4, i.e.
    # ~16x per halving of theta.
    ratios = [a / b for a, b in zip(small_energies[:-1], small_energies[1:])]
    for r in ratios:
        assert 13 < r < 19, f"spurious energy ratio {r:.1f} is not the ~16x of a theta^4 law"


def _stretched_stvk(make_unit_square, n=8, stretch=0.1):
    """A well-constrained St-Venant-Kirchhoff pull: left edge fixed, right edge stretched."""
    mesh = make_unit_square(n)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [stretch, 0]),
    )
    equation = FiniteStrainElastic(E=200, nu=0.4)
    return mesh, equation, bc


def test_finite_strain_solve_reaches_the_minres_backend(make_unit_square):
    """A nonlinear St-VK solve converges through MINRES to the same minimum as direct: the
    Hessian is indefinite at the zero seed, exercising MINRES and the regularization."""
    mesh, equation, bc = _stretched_stvk(make_unit_square)

    direct = Solver(mesh, equation, bc).solve().u
    iterative = Solver(mesh, equation, bc, backend=MinresBackend()).solve().u

    assert np.abs(direct).max() > 0, "trivial solution; test proves nothing"
    np.testing.assert_allclose(iterative, direct, atol=1e-7 * np.abs(direct).max())


def test_solver_uses_the_strategy_it_is_given(make_unit_square):
    """A caller's `NewtonSolve` replaces the default; the facade adds no policy of its own."""
    mesh, equation, bc = _stretched_stvk(make_unit_square)
    plain = NewtonSolve(line_search=None)
    solver = Solver(mesh, equation, bc, strategy=plain)

    assert solver.strategy is plain
    reference = Solver(mesh, equation, bc).solve().u
    np.testing.assert_allclose(solver.solve().u, reference, atol=1e-7 * np.abs(reference).max())


def test_regularization_leaves_an_spd_tangent_unshifted(make_unit_square):
    """On a LinearProblem (SPD tangent), regularization changes nothing: tau stays 0."""
    mesh = make_unit_square(8)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [0.05, 0]),
    )
    equation = LinearElastic(E=200, nu=0.3)
    problem = equation.problem(mesh, bc)

    plain = NewtonSolve().solve(problem)
    regularized = NewtonSolve(regularization=TangentRegularization()).solve(problem)
    np.testing.assert_allclose(regularized, plain, atol=1e-12)
