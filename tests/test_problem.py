"""The composition core: the Problem, its constant-tangent LinearProblem, and the solve
strategies. A LinearProblem has a constant tangent and an affine residual, so Newton
reaches the LinearSolve answer in one step from any seed.
"""
import numpy as np
import pytest

from fem.boundary import Dirichlet, Neumann, Robin
from fem.conditions import Conditions, Initial
from fem.field import NodalField
from fem.physics.energies import StVenantKirchhoff
from fem.physics.forms import EnergyForm, DiffusionForm, LinearElasticForm, ScaledForm
from fem.physics.materials import LinearElasticMaterial
from fem.numerics import central_difference_order
from fem.problem import LinearProblem, Problem
from fem.regions import everywhere, on_plane
from fem.algebra.solve import BacktrackingLineSearch, LinearSolve, NewtonSolve
from fem.physics.equations import Heat, LinearElastic, Poisson, Projection, FiniteStrainElastic
from fem.space import FunctionSpace
from fem.loads import Source
from helpers import pinned
from mms import exact_solution, source_term


def _problem(equation, mesh, bc=None):
    return equation.problem(mesh, bc)


def _poisson_problem(mesh):
    bc = pinned()
    return _problem(Poisson(), mesh, bc + Source(source_term))


def test_linear_solve_and_newton_agree_on_a_linear_problem(make_unit_square):
    problem = _poisson_problem(make_unit_square(15))

    u_linear = LinearSolve().solve(problem)
    u_newton = NewtonSolve().solve(problem)
    np.testing.assert_allclose(u_newton, u_linear, atol=1e-10)


def test_newton_on_a_linear_problem_is_seed_independent(make_unit_square):
    problem = _poisson_problem(make_unit_square(12))
    reference = LinearSolve().solve(problem)

    rng = np.random.default_rng(0)
    for _ in range(3):
        seed = rng.normal(size=problem.space.n_dofs)
        start = Initial(NodalField(problem.space, seed))
        np.testing.assert_allclose(NewtonSolve().solve(problem, initial=start), reference, atol=1e-10)


def test_line_search_is_a_noop_on_a_linear_problem(make_unit_square):
    """A LinearProblem's exact Newton step already lands on the solution, so backtracking
    accepts alpha = 1 on the first test and changes nothing. The merit is the quadratic
    energy 1/2 u.K.dofs - b.dofs, which the full step minimises."""
    problem = _poisson_problem(make_unit_square(15))
    reference = LinearSolve().solve(problem)

    searched = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    np.testing.assert_allclose(searched, reference, atol=1e-10)
    # And identical to the plain full-step path, not merely close.
    np.testing.assert_allclose(searched, NewtonSolve().solve(problem), atol=1e-12)


def test_composed_poisson_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(15)
    bc = pinned()
    equation = Poisson()

    u_composed = LinearSolve().solve(_problem(equation, mesh, bc + Source(source_term)))
    u_solver = equation.problem(mesh, bc + Source(source_term)).solve().dofs
    np.testing.assert_allclose(u_composed, u_solver, atol=1e-12)


def test_p2_is_reachable_through_the_solver_facade(make_unit_square):
    """P2 is a first-class option on the documented entry point, not only by hand-building
    a LinearProblem: `element_type` flows through `Equation.problem` to the space, and the
    solve is the accurate quadratic one."""
    from fem.elements import QuadraticTriangleElement

    mesh = make_unit_square(9)
    problem = Poisson().problem(mesh, pinned() + Source(source_term), element_type=QuadraticTriangleElement)
    solution = problem.solve()

    # P2 on this mesh is already far past the P1 error floor at the same spacing.
    assert len(solution.dofs) == problem.space.n_dofs
    assert np.abs(solution.dofs - exact_solution(problem.space.node_coords)).max() < 5e-3


def test_composed_linear_elastic_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(12)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )
    equation = LinearElastic(E=200, nu=0.4)

    u_composed = LinearSolve().solve(_problem(equation, mesh, bc))
    u_solver = equation.problem(mesh, bc).solve().dofs
    np.testing.assert_allclose(u_composed, u_solver, atol=1e-12)


def test_problem_packages_its_solution_by_physics(make_unit_square):
    """`Problem.solve` packages by physics: stress for an elastic operator, flux for
    a diffusion one, a bare field for a projection. The facade returns the same typed
    result."""
    from fem.post.solution import ElasticSolution, FieldSolution, DiffusionSolution

    mesh = make_unit_square(8)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )
    elastic = _problem(LinearElastic(E=200, nu=0.4), mesh, bc)
    solution = elastic.solve()
    assert isinstance(solution, ElasticSolution)
    facade = LinearElastic(E=200, nu=0.4).problem(mesh, bc).solve()
    assert isinstance(facade, ElasticSolution)
    np.testing.assert_allclose(solution.stress, facade.stress, atol=1e-12)

    scalar = _poisson_problem(mesh)
    assert isinstance(scalar.solve(), DiffusionSolution)

    projected = _problem(Projection(), mesh, Conditions(Source(2.0)))
    assert type(projected.solve()) is FieldSolution
    assert projected.near_null_space() is None
    assert elastic.near_null_space().shape == (elastic.space.n_dofs, 3)


def test_finite_strain_problem_packages_an_elastic_solution(make_unit_square):
    from fem.post.solution import ElasticSolution

    space = FunctionSpace(make_unit_square(5), n_components=2)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [0.05, 0]),
    )
    problem = Problem(space, EnergyForm(StVenantKirchhoff(200, 0.4)), bc)
    u = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    solution = problem.solution(u)
    assert isinstance(solution, ElasticSolution)
    assert solution.stress.shape == (len(space.mesh.elements), 3, 3)


def test_with_operator_matches_a_problem_built_from_scratch(make_unit_square):
    """Deriving a problem under a new operator is indistinguishable from stating it directly."""
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=2)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [50, 0]),
    )
    stiff = LinearElasticForm(LinearElasticMaterial(200.0, 0.4))
    soft = LinearElasticForm(LinearElasticMaterial(20.0, 0.4))

    derived = LinearProblem(space, stiff, bc + Source([0, -1])).with_operator(soft)
    direct = LinearProblem(space, soft, bc + Source([0, -1]))

    np.testing.assert_allclose(derived.load, direct.load, atol=1e-12)
    np.testing.assert_allclose(derived.tangent().toarray(), direct.tangent().toarray(), atol=1e-9)
    np.testing.assert_allclose(LinearSolve().solve(derived), LinearSolve().solve(direct), atol=1e-10)


def test_with_operator_reapplies_the_robin_boundary_term(make_unit_square):
    """A Robin condition sits on the operator as well as the load, so a derived
    problem has to carry it onto the new operator rather than lose it with the old."""
    mesh = make_unit_square(8)
    space = FunctionSpace(mesh, n_components=1)
    bc = Conditions(Robin(everywhere(), kappa=3.0, g=1.0))

    laplacian = DiffusionForm()
    doubled = ScaledForm(2.0, laplacian)
    derived = LinearProblem(space, laplacian, bc + Source(1.0)).with_operator(doubled)
    direct = LinearProblem(space, doubled, bc + Source(1.0))

    np.testing.assert_allclose(derived.tangent().toarray(), direct.tangent().toarray(), atol=1e-12)
    np.testing.assert_allclose(derived.load, direct.load, atol=1e-12)
    # And the Robin term is present: without it the operator would be 2K.
    bare = 2.0 * space.assemble(laplacian).toarray()
    assert np.abs(derived.tangent().toarray() - bare).max() > 1e-6


def test_traction_load_has_the_exact_resultant(make_unit_square):
    """A uniform edge traction assembles to a load totalling traction x loaded-length.
    Masking each Neumann region to its own facets is what makes this exact."""
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=1)
    bc = Conditions(Neumann(on_plane(0, 1.0), 2.0))
    load = LinearProblem(space, DiffusionForm(), bc).load
    np.testing.assert_allclose(load.sum(), 2.0, atol=1e-12)


def test_traction_stays_on_its_own_edge(make_unit_square):
    """The masked traction integrates over its region's facets only, so no load lands on a
    node off the loaded edge."""
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=1)
    bc = Conditions(Neumann(on_plane(0, 1.0), 2.0))
    load = LinearProblem(space, DiffusionForm(), bc).load
    off_edge = mesh.vertices[:, 0] < 1.0 - 1e-9
    np.testing.assert_allclose(load[off_edge], 0.0, atol=1e-12)


def test_derived_problem_does_not_answer_with_the_parents_operator(make_unit_square):
    """A derived problem must not keep the parent's already-assembled tangent."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    bc = pinned()
    parent = LinearProblem(space, DiffusionForm(), bc + Source(1.0))
    parent.tangent()   # populate the parent's cache *before* deriving

    derived = parent.with_operator(ScaledForm(3.0, DiffusionForm()))

    np.testing.assert_allclose(
        derived.tangent().toarray(), 3.0 * parent.tangent().toarray(), atol=1e-12,
    )


def test_tangent_is_assembled_once_and_held(make_unit_square):
    """Deferring the assembly must not turn into repeating it: the operator is
    constant, so every later call answers from the first assembly."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    problem = LinearProblem(space, DiffusionForm(), Conditions(Source(1.0)))

    assert problem.tangent() is problem.tangent()


def test_stating_a_problem_does_not_assemble_it(make_unit_square, monkeypatch):
    """A problem that is never solved costs nothing to state."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    assembled = []
    assemble = FunctionSpace.assemble

    def recording(self, form, boundary=False):
        assembled.append(form)
        return assemble(self, form, boundary)

    monkeypatch.setattr(FunctionSpace, 'assemble', recording)

    # The load still assembles a mass matrix, so it is the operator specifically
    # that must not have been touched yet.
    problem = LinearProblem(space, DiffusionForm(), Conditions(Source(1.0)))
    assert not any(isinstance(form, DiffusionForm) for form in assembled)

    problem.tangent()
    assert any(isinstance(form, DiffusionForm) for form in assembled)


def test_with_operator_leaves_the_original_alone(make_unit_square):
    """The derived problem is a new one; the operator it was derived from still
    answers with its own tangent."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    bc = pinned()
    original = LinearProblem(space, DiffusionForm(), bc + Source(1.0))
    before = original.tangent().toarray()

    original.with_operator(ScaledForm(5.0, DiffusionForm()))

    np.testing.assert_array_equal(original.tangent().toarray(), before)


def test_callable_source_is_sampled_at_the_quadrature_points(make_unit_square):
    """A callable source is integrated at the quadrature points, which differs from
    the mass matrix times its nodal values; `nodal=True` selects the latter."""
    from fem.loads import Source

    mesh = make_unit_square(6)
    space = FunctionSpace(mesh)

    def peaked(point):
        return np.exp(-40 * np.sum((point - 0.5) ** 2, axis=1))

    sampled = LinearProblem(space, DiffusionForm(), Conditions(Source(peaked))).load
    interpolated = LinearProblem(space, DiffusionForm(), Conditions(Source(peaked, nodal=True))).load
    nodal = space.mass_matrix @ peaked(space.node_coords)
    # nodal=True reads the callable at the nodes and integrates its interpolant, so it
    # equals mass @ nodal values; the default samples at the quadrature points, so it does not.
    assert np.allclose(interpolated, nodal)
    assert not np.allclose(sampled, nodal)


def _loaded_bc(scale=1.0):
    """Supports, a traction, and a Robin spring: every term the composition has."""
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 1.0), [0, -2.0 * scale]),
        Robin(on_plane(1, 0.0), kappa=15.0, g=[0.0, 0.5 * scale]),
    )
    return bc


@pytest.mark.parametrize('model', [LinearElastic, FiniteStrainElastic])
def test_composed_energy_residual_and_tangent_are_consistent(make_unit_square, model):
    """With a body force, a traction, and a Robin term all present, the problem's
    residual is the gradient of its energy and its tangent the gradient of its residual,
    to O(eps^2) under central differences. Holds for the constant and the
    state-dependent tangent alike, so a line search on the energy and Newton on the
    residual agree on which way is downhill."""
    equation = model(E=200, nu=0.4)
    problem = equation.problem(make_unit_square(5), _loaded_bc() + Source([1.0, -3.0]))

    rng = np.random.default_rng(1)
    u = 0.05 * rng.standard_normal(problem.space.n_dofs)
    residual = problem.residual(u)
    tangent = problem.tangent(u)

    if model is LinearElastic:
        # A quadratic energy's central difference is exact at any step, so there is no
        # order to measure; check the values directly.
        d = rng.standard_normal(problem.space.n_dofs)
        eps = 1e-4
        fd = (problem.energy(u + eps * d) - problem.energy(u - eps * d)) / (2 * eps)
        assert fd == pytest.approx(residual @ d, rel=1e-8)
        fd_r = (problem.residual(u + eps * d) - problem.residual(u - eps * d)) / (2 * eps)
        np.testing.assert_allclose(fd_r, tangent @ d, rtol=1e-8, atol=1e-10)
    else:
        grad_order = central_difference_order(problem.energy, lambda d: residual @ d, u)
        hess_order = central_difference_order(problem.residual, lambda d: tangent @ d, u)
        assert 1.9 < grad_order < 2.1, f"residual disagrees with d(energy): order {grad_order:.3f}"
        assert 1.9 < hess_order < 2.1, f"tangent disagrees with d(residual): order {hess_order:.3f}"


def test_forced_finite_strain_problem_balances_its_load(make_unit_square):
    """A Green-Lagrange problem takes a body force, a traction, and a Robin support: Newton
    drives the free residual to zero, the internal force balances the load there, and
    at small load the answer agrees with the small-strain solve to second order."""
    mesh = make_unit_square(6)
    scale = 1e-3
    bc = _loaded_bc(scale)
    finite = FiniteStrainElastic(E=200, nu=0.4)
    problem = finite.problem(mesh, bc + Source([scale, -3 * scale]))
    free = problem.constraints[0]
    load_scale = float(np.abs(problem.load).max())
    assert load_scale > 0

    u = NewtonSolve(line_search=BacktrackingLineSearch(), tol=1e-10).solve(problem)
    np.testing.assert_allclose(problem.residual(u)[free], 0.0, atol=1e-8 * load_scale)
    np.testing.assert_allclose(problem.internal_residual(u)[free], problem.load[free],
                               atol=1e-8 * load_scale)

    linear = LinearElastic(E=200, nu=0.4)
    u_linear = LinearSolve().solve(linear.problem(mesh, bc + Source([scale, -3 * scale])))
    assert np.abs(u_linear).max() > 0
    rel = np.linalg.norm(u - u_linear) / np.linalg.norm(u_linear)
    assert rel < 1e-2, f"finite and small strain should agree at small load, got {rel:.2e}"


def test_linear_problem_refuses_a_state_dependent_operator(make_unit_square):
    """`LinearProblem` is the type consumers needing one fixed operator ask for, so a
    state-dependent form is refused at construction and by `with_operator`; and a
    `LinearSolve` refuses the `Problem` such a form makes."""
    space = FunctionSpace(make_unit_square(4), n_components=2)
    stvk = EnergyForm(StVenantKirchhoff(200, 0.4))
    with pytest.raises(TypeError, match='state-dependent'):
        LinearProblem(space, stvk)
    with pytest.raises(TypeError, match='state-dependent'):
        LinearProblem(space, LinearElasticForm(LinearElasticMaterial(200, 0.4))).with_operator(stvk)

    problem = Problem(space, stvk)
    with pytest.raises(TypeError, match='constant tangent'):
        LinearSolve().solve(problem)
    with pytest.raises(ValueError, match='state-dependent'):
        problem.tangent()


def test_problem_solve_is_the_strategy_solve_packaged(make_unit_square):
    """`Problem.solve()` returns the same typed solution as solving with the default
    strategy by hand; a strategy and the problem's backend are independent choices, so
    both may be given."""
    from fem.algebra.backends import DirectBackend
    problem = _poisson_problem(make_unit_square(8))
    by_hand = problem.solution(LinearSolve().solve(problem))
    solution = problem.solve()
    assert type(solution) is type(by_hand)
    np.testing.assert_array_equal(solution.dofs, by_hand.dofs)
    both = problem.with_backend(DirectBackend()).solve(strategy=LinearSolve())
    np.testing.assert_array_equal(both.dofs, by_hand.dofs)


def test_problem_solve_picks_newton_for_a_state_dependent_operator(make_unit_square):
    """A Green-Lagrange problem solved through `Problem.solve()` matches a hand-run
    line-searched Newton solve."""
    mesh = make_unit_square(4)
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
        Dirichlet(on_plane(0, 1.0), [0.05, 0.0]),
    )
    finite = FiniteStrainElastic(E=200, nu=0.3)
    problem = finite.problem(mesh, bc)
    assert type(problem) is Problem
    u_newton = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    np.testing.assert_allclose(problem.solve().dofs, u_newton, atol=1e-12)


def test_equation_problem_takes_a_mesh_or_a_space(make_unit_square):
    """`Equation.problem` on a mesh builds the space the equation implies, honouring
    `element_type`; on a space it uses that space and refuses an `element_type`."""
    from fem.elements import QuadraticTriangleElement
    mesh = make_unit_square(4)
    equation = LinearElastic(E=1.0, nu=0.3)
    from_mesh = equation.problem(mesh, element_type=QuadraticTriangleElement)
    assert from_mesh.space.element_type is QuadraticTriangleElement
    assert from_mesh.space.n_components == 2

    space = equation.space(mesh)
    from_space = equation.problem(space)
    assert from_space.space is space
    with pytest.raises(ValueError, match='element_type'):
        equation.problem(space, element_type=QuadraticTriangleElement)


def test_mass_is_the_density_scaled_mass_matrix_held_across_operators(make_unit_square):
    """`Problem.mass` is density times the space's mass matrix, assembled once, and a
    problem derived with `with_operator` keeps it."""
    from fem.physics.forms import MassForm
    mesh = make_unit_square(5)
    problem = Heat(capacity=3.0).problem(mesh)
    assert problem.density == 3.0
    np.testing.assert_allclose(problem.mass.toarray(), 3.0 * problem.space.mass_matrix.toarray())
    assert problem.mass is problem.mass
    assert problem.with_operator(MassForm()).mass is problem.mass
    with pytest.raises(ValueError, match='density'):
        Problem(problem.space, DiffusionForm(), density=0.0)


# -- Robin flux ------------------------------------------------------------------


def test_robin_flux_is_the_heat_leaving_through_the_film(make_unit_square):
    """u = 1 at x = 0 and du/dn + u = 0 at x = 1 has the linear solution u = 1 - x/2,
    so the flux through the film is κu = 1/2 along a unit edge. Exact on P1."""
    mesh = make_unit_square(6)
    bc = Conditions(Dirichlet(on_plane(0, 0.0), 1.0), Robin(on_plane(0, 1.0), kappa=1.0, g=0.0))
    problem = Poisson().problem(mesh, bc)
    u = problem.solve()
    assert problem.robin_flux(u) == pytest.approx(0.5)
    assert problem.robin_flux(u.dofs) == pytest.approx(0.5)


def test_robin_flux_subtracts_the_condition_value(make_unit_square):
    """With g = κ u_ambient the flux is κ ∫ (u - u_ambient): zero for a field at ambient."""
    mesh = make_unit_square(4)
    bc = Conditions(Robin(everywhere(), kappa=2.0, g=2.0 * 300.0))
    problem = Poisson().problem(mesh, bc)
    ambient = problem.space.interpolate(300.0)
    assert problem.robin_flux(ambient) == pytest.approx(0.0, abs=1e-9)
    assert problem.robin_flux(problem.space.interpolate(301.0)) == pytest.approx(2.0 * 4.0)


def test_robin_flux_takes_the_condition_when_there_are_several(make_unit_square):
    mesh = make_unit_square(4)
    left = Robin(on_plane(0, 0.0), kappa=1.0, g=0.0)
    right = Robin(on_plane(0, 1.0), kappa=3.0, g=0.0)
    problem = Poisson().problem(mesh, Conditions(left, right))
    ones = problem.space.interpolate(1.0)
    assert problem.robin_flux(ones, left) == pytest.approx(1.0)
    assert problem.robin_flux(ones, right) == pytest.approx(3.0)
    with pytest.raises(ValueError, match='2 Robin conditions'):
        problem.robin_flux(ones)
    with pytest.raises(ValueError, match='not a Robin condition of this problem'):
        problem.robin_flux(ones, Robin(on_plane(1, 0.0), kappa=1.0, g=0.0))
