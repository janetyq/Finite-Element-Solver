"""The composition core: the Problem, its constant-tangent LinearProblem, and the solve
strategies. A LinearProblem has a constant tangent and an affine residual, so Newton
reaches the LinearSolve answer in one step from any seed.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.energies import StVenantKirchhoff
from fem.forms import EnergyForm, LaplacianForm, LinearElasticForm, ScaledForm
from fem.materials import LinearElasticMaterial
from fem.numerics import central_difference_order
from fem.problem import LinearProblem, Problem
from fem.regions import everywhere, on_plane
from fem.solve import BacktrackingLineSearch, LinearSolve, NewtonSolve
from fem.equations import LinearElastic, Poisson, Projection, StrainMeasure
from fem.solver import Solver
from fem.space import FunctionSpace


def _mms_source(p):
    return [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])]


def _problem(equation, mesh, bc=None):
    return equation.problem(mesh, bc)


def _poisson_problem(mesh):
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    return _problem(Poisson(source=_mms_source), mesh, bc)


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
        np.testing.assert_allclose(NewtonSolve().solve(problem, u0=seed), reference, atol=1e-10)


def test_line_search_is_a_noop_on_a_linear_problem(make_unit_square):
    """A LinearProblem's exact Newton step already lands on the solution, so backtracking
    accepts alpha = 1 on the first test and changes nothing. The merit is the quadratic
    energy 1/2 u.K.u - b.u, which the full step minimises."""
    problem = _poisson_problem(make_unit_square(15))
    reference = LinearSolve().solve(problem)

    searched = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    np.testing.assert_allclose(searched, reference, atol=1e-10)
    # And identical to the plain full-step path, not merely close.
    np.testing.assert_allclose(searched, NewtonSolve().solve(problem), atol=1e-12)


def test_composed_poisson_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(15)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    equation = Poisson(source=_mms_source)

    u_composed = LinearSolve().solve(_problem(equation, mesh, bc))
    u_solver = Solver(mesh, equation, bc).solve().u
    np.testing.assert_allclose(u_composed, u_solver, atol=1e-12)


def test_composed_linear_elastic_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(12)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])
    equation = LinearElastic(E=200, nu=0.4)

    u_composed = LinearSolve().solve(_problem(equation, mesh, bc))
    u_solver = Solver(mesh, equation, bc).solve().u
    np.testing.assert_allclose(u_composed, u_solver, atol=1e-12)


def test_problem_packages_its_solution_by_physics(make_unit_square):
    """`Problem.solve` packages by physics: stress for an elastic operator, flux for
    a diffusion one, a bare field for a projection. The facade returns the same typed
    result."""
    from fem.solution import ElasticSolution, FieldSolution, ScalarFieldSolution

    mesh = make_unit_square(8)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])
    elastic = _problem(LinearElastic(E=200, nu=0.4), mesh, bc)
    solution = elastic.solve()
    assert isinstance(solution, ElasticSolution)
    facade = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve()
    assert isinstance(facade, ElasticSolution)
    np.testing.assert_allclose(solution.stress, facade.stress, atol=1e-12)

    scalar = _poisson_problem(mesh)
    assert isinstance(scalar.solve(), ScalarFieldSolution)

    projected = _problem(Projection(source=2.0), mesh)
    assert type(projected.solve()) is FieldSolution
    assert projected.near_null_space() is None
    assert elastic.near_null_space().shape == (elastic.space.n_dofs, 3)


def test_finite_strain_problem_packages_an_elastic_solution(make_unit_square):
    from fem.solution import ElasticSolution

    space = FunctionSpace(make_unit_square(5), n_components=2)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [0.05, 0])
    problem = Problem(space, EnergyForm(StVenantKirchhoff(200, 0.4)), bc=bc)
    u = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    solution = problem.solution(u)
    assert isinstance(solution, ElasticSolution)
    assert solution.stress.shape == (len(space.mesh.elements), 3, 3)


def test_with_operator_matches_a_problem_built_from_scratch(make_unit_square):
    """Deriving a problem under a new operator is indistinguishable from stating it directly."""
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=2)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])
    stiff = LinearElasticForm(LinearElasticMaterial(200.0, 0.4))
    soft = LinearElasticForm(LinearElasticMaterial(20.0, 0.4))

    derived = LinearProblem(space, stiff, [0, -1], bc).with_operator(soft)
    direct = LinearProblem(space, soft, [0, -1], bc)

    np.testing.assert_allclose(derived.load, direct.load, atol=1e-12)
    np.testing.assert_allclose(derived.tangent().toarray(), direct.tangent().toarray(), atol=1e-9)
    np.testing.assert_allclose(LinearSolve().solve(derived), LinearSolve().solve(direct), atol=1e-10)


def test_with_operator_reapplies_the_robin_boundary_term(make_unit_square):
    """A Robin condition sits on the operator as well as the load, so a derived
    problem has to carry it onto the new operator rather than lose it with the old."""
    mesh = make_unit_square(8)
    space = FunctionSpace(mesh, n_components=1)
    bc = BoundaryConditions()
    bc.add_robin(everywhere(), kappa=3.0, g=1.0)

    laplacian = LaplacianForm()
    doubled = ScaledForm(2.0, laplacian)
    derived = LinearProblem(space, laplacian, 1.0, bc).with_operator(doubled)
    direct = LinearProblem(space, doubled, 1.0, bc)

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
    bc = BoundaryConditions()
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), 2.0)   # flux 2 on the right edge, length 1
    load = LinearProblem(space, LaplacianForm(), None, bc).load
    np.testing.assert_allclose(load.sum(), 2.0, atol=1e-12)


def test_traction_stays_on_its_own_edge(make_unit_square):
    """The masked traction integrates over its region's facets only, so no load lands on a
    node off the loaded edge."""
    mesh = make_unit_square(10)
    space = FunctionSpace(mesh, n_components=1)
    bc = BoundaryConditions()
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), 2.0)
    load = LinearProblem(space, LaplacianForm(), None, bc).load
    off_edge = mesh.vertices[:, 0] < 1.0 - 1e-9
    np.testing.assert_allclose(load[off_edge], 0.0, atol=1e-12)


def test_derived_problem_does_not_answer_with_the_parents_operator(make_unit_square):
    """A derived problem must not keep the parent's already-assembled tangent."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    parent = LinearProblem(space, LaplacianForm(), 1.0, bc)
    parent.tangent()   # populate the parent's cache *before* deriving

    derived = parent.with_operator(ScaledForm(3.0, LaplacianForm()))

    np.testing.assert_allclose(
        derived.tangent().toarray(), 3.0 * parent.tangent().toarray(), atol=1e-12,
    )


def test_tangent_is_assembled_once_and_held(make_unit_square):
    """Deferring the assembly must not turn into repeating it: the operator is
    constant, so every later call answers from the first assembly."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    problem = LinearProblem(space, LaplacianForm(), 1.0)

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
    problem = LinearProblem(space, LaplacianForm(), 1.0)
    assert not any(isinstance(form, LaplacianForm) for form in assembled)

    problem.tangent()
    assert any(isinstance(form, LaplacianForm) for form in assembled)


def test_with_operator_leaves_the_original_alone(make_unit_square):
    """The derived problem is a new one; the operator it was derived from still
    answers with its own tangent."""
    space = FunctionSpace(make_unit_square(6), n_components=1)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    original = LinearProblem(space, LaplacianForm(), 1.0, bc)
    before = original.tangent().toarray()

    original.with_operator(ScaledForm(5.0, LaplacianForm()))

    np.testing.assert_array_equal(original.tangent().toarray(), before)


def test_callable_source_is_sampled_at_the_quadrature_points(make_unit_square):
    """A callable source builds the same load as the LinearForm it is wrapped in, not
    the mass matrix times its nodal values."""
    from fem.forms import LinearForm

    mesh = make_unit_square(6)
    space = FunctionSpace(mesh)

    def peaked(point):
        return float(np.exp(-40 * np.sum((point - 0.5) ** 2)))

    sampled = LinearProblem(space, LaplacianForm(), peaked).load
    explicit = LinearProblem(space, LaplacianForm(), LinearForm(peaked)).load
    nodal = space.mass_matrix @ np.array([peaked(p) for p in space.node_coords])
    assert np.allclose(sampled, explicit)
    assert not np.allclose(sampled, nodal)


def _loaded_bc(scale=1.0):
    """Supports, a traction, and a Robin spring: every term the composition has."""
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0, -2.0 * scale])
    bc.add_robin(on_plane(1, 0.0), kappa=15.0, g=[0.0, 0.5 * scale])
    return bc


@pytest.mark.parametrize('kinematics', [StrainMeasure.SMALL, StrainMeasure.GREEN_LAGRANGE])
def test_composed_energy_residual_and_tangent_are_consistent(make_unit_square, kinematics):
    """With a body force, a traction, and a Robin term all present, the problem's
    residual is the gradient of its energy and its tangent the gradient of its residual,
    to O(eps^2) under central differences. Holds for the constant and the
    state-dependent tangent alike, so a line search on the energy and Newton on the
    residual agree on which way is downhill."""
    equation = LinearElastic(E=200, nu=0.4, source=[1.0, -3.0], kinematics=kinematics)
    problem = equation.problem(make_unit_square(5), _loaded_bc())

    rng = np.random.default_rng(1)
    u = 0.05 * rng.standard_normal(problem.space.n_dofs)
    residual = problem.residual(u)
    tangent = problem.tangent(u)

    if kinematics is StrainMeasure.SMALL:
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
    finite = LinearElastic(E=200, nu=0.4, source=[scale, -3 * scale],
                           kinematics=StrainMeasure.GREEN_LAGRANGE)
    problem = finite.problem(mesh, bc)
    free = problem.constraints[0]
    load_scale = float(np.abs(problem.load).max())
    assert load_scale > 0

    u = NewtonSolve(line_search=BacktrackingLineSearch(), tol=1e-10).solve(problem)
    np.testing.assert_allclose(problem.residual(u)[free], 0.0, atol=1e-8 * load_scale)
    np.testing.assert_allclose(problem.internal_residual(u)[free], problem.load[free],
                               atol=1e-8 * load_scale)

    linear = LinearElastic(E=200, nu=0.4, source=[scale, -3 * scale])
    u_linear = LinearSolve().solve(linear.problem(mesh, bc))
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
    strategy by hand; a strategy and a backend together are refused."""
    from fem.backends import DirectBackend
    problem = _poisson_problem(make_unit_square(8))
    by_hand = problem.solution(LinearSolve().solve(problem))
    solution = problem.solve()
    assert type(solution) is type(by_hand)
    np.testing.assert_array_equal(solution.u, by_hand.u)
    with pytest.raises(ValueError, match='one or the other'):
        problem.solve(strategy=LinearSolve(), backend=DirectBackend())


def test_problem_solve_picks_newton_for_a_state_dependent_operator(make_unit_square):
    """A Green-Lagrange problem solved through `Problem.solve()` matches a hand-run
    line-searched Newton solve."""
    mesh = make_unit_square(4)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.DIRICHLET, on_plane(0, 1.0), [0.05, 0.0])
    finite = LinearElastic(E=200, nu=0.3, kinematics=StrainMeasure.GREEN_LAGRANGE)
    problem = finite.problem(mesh, bc)
    assert type(problem) is Problem
    u_newton = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    np.testing.assert_allclose(problem.solve().u, u_newton, atol=1e-12)


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
