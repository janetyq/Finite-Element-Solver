"""The composition core: LinearProblem / EnergyProblem and the solve strategies. A
LinearProblem has a constant tangent and an affine residual, so Newton reaches the
LinearSolve answer in one step from any seed.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.energies import StVenantKirchhoff
from fem.forms import EnergyForm, LaplacianForm, LinearElasticForm, ScaledForm
from fem.materials import LinearElasticMaterial
from fem.problem import EnergyProblem, LinearProblem, linear_elastic, poisson
from fem.regions import everywhere, on_plane
from fem.solve import BacktrackingLineSearch, LinearSolve, NewtonSolve
from fem.equations import LinearElastic, Poisson
from fem.solver import Solver
from fem.space import FunctionSpace


def _mms_source(p):
    return [2 * np.pi**2 * np.sin(np.pi * p[0]) * np.sin(np.pi * p[1])]


def _poisson_problem(mesh):
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    return poisson(mesh, _mms_source, bc)


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
    accepts alpha = 1 on the first test and changes nothing. This also exercises the
    residual-norm merit fallback: a LinearProblem is not `SupportsEnergy`, so the search
    scores states by 1/2||r||^2, which is zero at the solution the full step reaches."""
    problem = _poisson_problem(make_unit_square(15))
    reference = LinearSolve().solve(problem)

    searched = NewtonSolve(line_search=BacktrackingLineSearch()).solve(problem)
    np.testing.assert_allclose(searched, reference, atol=1e-10)
    # And identical to the plain full-step path, not merely close.
    np.testing.assert_allclose(searched, NewtonSolve().solve(problem), atol=1e-12)


def test_poisson_factory_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(15)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    u_factory = LinearSolve().solve(poisson(mesh, _mms_source, bc))
    u_solver = Solver(mesh, Poisson(source=_mms_source), bc).solve().u
    np.testing.assert_allclose(u_factory, u_solver, atol=1e-12)


def test_linear_elastic_factory_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(12)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])

    u_factory = LinearSolve().solve(linear_elastic(mesh, LinearElasticMaterial(200, 0.4), bc))
    u_solver = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve().u
    np.testing.assert_allclose(u_factory, u_solver, atol=1e-12)


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
    # And the Robin term is genuinely there: without it the operator would be 2K.
    bare = 2.0 * space.assemble(laplacian).toarray()
    assert np.abs(derived.tangent().toarray() - bare).max() > 1e-6


def test_traction_load_has_the_exact_resultant(make_unit_square):
    """A uniform edge traction assembles to a load totalling traction x loaded-length.
    Masking each Neumann region to its own facets is what makes this exact; the unmasked
    boundary mass it replaces leaked onto the neighbours and inflated the resultant."""
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


def test_energy_problem_rejects_a_source(make_unit_square):
    space = FunctionSpace(make_unit_square(6), n_components=2)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    with pytest.raises(NotImplementedError):
        EnergyProblem(space, EnergyForm(StVenantKirchhoff(200, 0.4)), bc, source=1.0)
