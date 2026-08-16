"""How the two elasticity paths relate to each other.

The package solves elasticity twice, along two independent axes:

    strain measure   Green-Lagrange  S = 1/2 (F^T F - I)       exact
                     small strain    eps = 1/2 (grad u + grad u^T)

    method           direct assembly      K u = b, one linear solve
                     energy minimization  Newton on grad(Pi) = 0

`Solver` occupies (small strain, direct); `EnergySolver` occupies
(Green-Lagrange, energy). Because they sit on the diagonal, switching solver
also switches physics, and neither difference can be observed alone.

`SmallStrain` (`fem/energies.py`) is the missing off-diagonal cell
-- Green-Lagrange's linearization dropped into the energy machinery. Holding the
method fixed while changing only the strain measure, and vice versa, makes each
axis observable on its own. That turns the two comments which used to carry this
knowledge -- "does not exactly match linear elastic solve for larger
deformations" and "not the linear approx ... so iterative solver does not
converge in 1 iteration" -- into executable claims.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.materials import LinearElasticMaterial
from fem.regions import on_plane
from fem.equations import LinearElastic, StrainMeasure
from fem.solver import Solver
from fem.energy_solver import EnergySolver
from fem.solve import BacktrackingLineSearch, NewtonSolve
from fem.energies import SmallStrain, StVenantKirchhoff
from fem.forms import EnergyForm
from fem.numerics import central_difference_order


def test_hooke_matrix_is_the_second_derivative_of_the_small_strain_energy():
    """`Material`'s D and `SmallStrain`'s W are one material, D = d2W/de2.

    Both `Solver` (via LinearElasticMaterial -> D) and the small-strain energy
    path describe the same linear material by different routes. This pins that
    they agree: for any strain, the quadratic energy the Voigt D implies,
    1/2 eps_v^T D eps_v, equals the energy density's W(eps) -- the shear terms
    included, where the engineering-shear factor of 2 in eps_v = [e_xx, e_yy,
    2 e_xy] cancels D's bare mu against W's 2 mu e_xy^2.

    Checked in 2D, the only dimension where both representations exist:
    `energies.py` is fixed-rank-2, so the 3D D is not cross-checked here -- it
    stands on its closed form and the 3D MMS convergence test. This is why the
    duplication is left in place rather than derived away: the closed-form D is
    correct and dimension-general, and deriving it from the 2D energy density
    would forfeit 3D.
    """
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
    """Unit square, left edge pinned, right edge displaced by `stretch` in x.

    Displacement-driven with no body force and no traction, which is the only
    setup the two solvers can be compared on: EnergySolver builds no load
    vector, so `Solver` must see b = 0 for the problems to coincide.
    """
    mesh = make_unit_square(n)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [stretch, 0])
    return mesh, bc


def _energy_solver(mesh, bc, kinematics):
    """An EnergySolver on a LinearElastic with the given strain measure.

    The kinematics is an equation-level choice, so it flows through construction
    (`LinearElastic.energy_density` maps it to the density) rather than being
    swapped in after.
    """
    equation = LinearElastic(E=200, nu=0.4, kinematics=kinematics)
    return EnergySolver(mesh, equation, bc)


def _one_newton_step(solver):
    """Displacement after a single Newton step from the zero initial guess."""
    problem = solver.problem()
    _, fixed, fixed_values = problem.constraints
    u = np.zeros(solver.space.n_dofs)
    u[fixed] = fixed_values
    return NewtonSolve(max_iters=1).solve(problem, u0=u)


def test_line_search_converges_from_a_seed_a_full_step_diverges_from(make_unit_square):
    """Under strong compression the St-Venant–Kirchhoff tangent loses ellipticity, so a
    full Newton step from the zero seed overshoots and the iteration diverges. Backtracking
    on the energy keeps every step a descent and reaches equilibrium.

    The right edge is pushed 70% of the way through the block, deep into the non-convex
    regime. Convergence is read from the free-DOF residual (the fixed DOFs carry reaction
    forces that are nonzero at equilibrium): the line-searched solve drives it to zero while
    the full-step solve blows up by orders of magnitude within the same iteration budget.
    """
    mesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [-0.7, 0])   # 70% compression
    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    solver = EnergySolver(mesh, equation, bc)

    problem = solver.problem()
    free, fixed, fixed_values = problem.constraints
    seed = np.zeros(solver.space.n_dofs)
    seed[fixed] = fixed_values

    def free_residual(line_search):
        u = NewtonSolve(max_iters=50, line_search=line_search).solve(problem, u0=seed.copy())
        return float(np.linalg.norm(problem.residual(u)[free]))

    r_full = free_residual(None)
    r_searched = free_residual(BacktrackingLineSearch())

    assert r_searched < 1e-5, f"line search should converge, got residual {r_searched:.2e}"
    assert not np.isfinite(r_full) or r_full > 1e-2, (
        f"the full step should fail to converge here, got residual {r_full:.2e}"
    )


def test_kinematics_is_an_equation_level_choice():
    """The strain measure selects the density on the equation itself, not in a
    solver: LinearElastic.energy_density owns the mapping, and defaults to small
    strain (matching the equation's name and the linear path)."""
    small = LinearElastic(200, 0.4, kinematics=StrainMeasure.SMALL)
    finite = LinearElastic(200, 0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    default = LinearElastic(200, 0.4)

    assert isinstance(small.energy_density(), SmallStrain)
    assert isinstance(default.energy_density(), SmallStrain)
    # SmallStrain subclasses StVenantKirchhoff, so pin the finite member by exact type.
    assert type(finite.energy_density()) is StVenantKirchhoff


def test_energy_solver_matches_recorded_solution(make_unit_square):
    """Regression pin on the St Venant-Kirchhoff answer.

    The relationship tests below fix how this model compares to small strain,
    but nothing else pins its absolute values, so a change in the energy
    density or its derivatives could shift both sides together and go unseen.
    Values recorded from the implementation, not derived independently -- this
    catches drift, it does not prove correctness.
    """
    mesh, bc = _stretched_square(make_unit_square)
    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    solver = EnergySolver(mesh, equation, bc)
    u = solver.solve().u

    np.testing.assert_allclose(np.linalg.norm(u), 0.503442620332, rtol=1e-9)
    np.testing.assert_allclose(u.max(), 0.1, rtol=1e-12)
    np.testing.assert_allclose(u.min(), -0.037995668257, rtol=1e-9)
    np.testing.assert_allclose(solver.energy(u.copy()), 1.590561321584, rtol=1e-9)


def test_energy_gradient_and_hessian_are_consistent_by_finite_difference(make_unit_square):
    """`energy_gradient` is the gradient of `energy`, and `energy_hessian` the gradient
    of `energy_gradient`, each to O(eps^2) under central differences.

    These are the assembled quantities Newton uses: the potential energy Pi(u) summed
    over the mesh, with its global gradient and Hessian. The density they are built
    from is checked separately, by StVenantKirchhoff.check_gradients. Between the two
    lies the element assembly, which evaluates the density per element and sums each
    contribution into the global arrays by DOF. A mistake there (a wrong scatter, or a
    transposed element matrix) leaves the density correct while making the assembled
    gradient and Hessian inconsistent with the assembled energy. The density check
    cannot see that; this test can.

    St-VK (not small strain) is needed for a meaningful slope: its energy is quartic in
    u, so the error is a clean O(eps^2), whereas a quadratic small-strain energy makes
    the central difference exact and the slope meaningless.
    """
    mesh = make_unit_square(5)
    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    # BCs are unused here: energy/gradient/hessian are the raw, unconstrained
    # quantities, evaluated at an imposed state rather than a solve.
    solver = EnergySolver(mesh, equation)

    # A non-trivial state, so the nonlinearity is active -- at u = 0 the tangent is the
    # small-strain one and the cubic term this check leans on would vanish.
    rng = np.random.default_rng(0)
    u = 0.1 * rng.standard_normal(solver.space.n_dofs)
    gradient = solver.energy_gradient(u)
    hessian = solver.energy_hessian(u)

    grad_order = central_difference_order(solver.energy, lambda d: gradient @ d, u)
    hess_order = central_difference_order(solver.energy_gradient, lambda d: hessian @ d, u)

    assert 1.9 < grad_order < 2.1, f"energy_gradient disagrees with d(energy): order {grad_order:.3f}"
    assert 1.9 < hess_order < 2.1, f"energy_hessian disagrees with d(energy_gradient): order {hess_order:.3f}"


def test_small_strain_energy_equals_direct_solve(make_unit_square):
    """Same method, both strain measures agree in the limit: hold the *method*
    fixed at energy minimization and switch to small strain, and it reproduces
    `Solver`'s direct assembly exactly.

    And in a single Newton step: a small-strain energy is quadratic, so its
    gradient is affine and one step from any start lands on the minimizer. This
    is the "converges in one iteration" the St-VK comment said it does *not* --
    the difference is the strain measure, not the solver.
    """
    mesh, bc = _stretched_square(make_unit_square)

    u_direct = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve().u.flatten()
    u_energy = _one_newton_step(_energy_solver(mesh, bc, StrainMeasure.SMALL))

    np.testing.assert_allclose(u_energy, u_direct, atol=1e-12)


def test_stvk_needs_more_than_one_newton_step(make_unit_square):
    """The complement: St-VK is nonlinear in u, so one Newton step is *not*
    enough -- it leaves a residual against the converged answer. This is what
    makes the small-strain one-step result above a real distinction rather than
    a property of the solver.
    """
    mesh, bc = _stretched_square(make_unit_square)
    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    solver = EnergySolver(mesh, equation, bc)

    u_one = _one_newton_step(solver)
    u_converged = solver.solve().u

    rel = np.linalg.norm(u_one - u_converged) / np.linalg.norm(u_converged)
    assert rel > 0.1, f"one step should be far from converged, got rel={rel:.2e}"


def test_models_agree_to_second_order_in_strain(make_unit_square):
    """Same strain regime, both models: hold the *strain* small and the two
    measures agree to O(||grad u||^2). Halving the imposed stretch shrinks the
    displacement gap by ~4x, the signature of a quadratic difference -- the
    precise content of "does not exactly match ... for larger deformations".
    """
    gaps = []
    for stretch in (0.08, 0.04, 0.02, 0.01):
        mesh, bc = _stretched_square(make_unit_square, stretch=stretch)
        u_small = _energy_solver(mesh, bc, StrainMeasure.SMALL).solve().u
        u_stvk = _energy_solver(mesh, bc, StrainMeasure.GREEN_LAGRANGE).solve().u
        gaps.append(np.linalg.norm(u_small - u_stvk))

    ratios = [a / b for a, b in zip(gaps[:-1], gaps[1:])]
    # Quadratic gap -> 4x per halving. Loose bounds: the far field is not purely
    # asymptotic and the mesh is coarse, but the trend must be unambiguously ~4.
    for r in ratios:
        assert 3.5 < r < 4.5, f"gap ratio {r:.2f} is not the ~4x of a quadratic difference"


def test_green_lagrange_is_frame_indifferent(make_unit_square):
    """A rigid rotation is strain-free -- the defining property of an exact
    strain measure, and the reason St-VK exists.

    Under a rigid rotation F = R exactly (the displacement is affine, which P1
    represents without error), so Green-Lagrange gives S = 1/2 (R^T R - I) = 0
    and zero energy. Small strain instead reads eps = (cos theta - 1) I, a
    spurious compression, and stores energy that grows like theta^4. Evaluated
    directly on the rotation field -- no solve -- so it isolates the strain
    measure from the solver.
    """
    mesh = make_unit_square(8)
    center = mesh.vertices.mean(axis=0)
    # No pinned DOFs would make an EnergySolver degenerate; this BC is unused,
    # since the energies are evaluated on an imposed field rather than solved.
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    solver = EnergySolver(mesh, LinearElastic(E=200, nu=0.4), bc)

    def total_energy(density, u_nodal):
        return solver.space.total_energy(EnergyForm(density), u_nodal.flatten())

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
