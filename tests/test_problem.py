"""The composition core: LinearProblem / EnergyProblem and the solve strategies.

The headline is the equivalence the pre-composition architecture could not even
state: LinearSolve and NewtonSolve share no ancestor there, so nothing could
assert they agree. Here a LinearProblem has a constant tangent and an affine
residual, so Newton reaches the LinearSolve answer in one applied step -- from any
seed -- and the two are cross-checked directly.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, BCType
from fem.energies import StVenantKirchhoff
from fem.forms import EnergyForm
from fem.materials import LinearElasticMaterial
from fem.problem import EnergyProblem, linear_elastic, poisson
from fem.regions import everywhere, on_plane
from fem.solve import LinearSolve, NewtonSolve
from fem.solver import LinearElastic, Poisson, Solver
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


def test_poisson_factory_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(15)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    u_factory = LinearSolve().solve(poisson(mesh, _mms_source, bc))
    u_solver = Solver(mesh, Poisson(source=_mms_source), bc).solve().get_values("u")
    np.testing.assert_allclose(u_factory, u_solver, atol=1e-12)


def test_linear_elastic_factory_matches_the_solver_facade(make_unit_square):
    mesh = make_unit_square(12)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [50, 0])

    u_factory = LinearSolve().solve(linear_elastic(mesh, LinearElasticMaterial(200, 0.4), bc))
    u_solver = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve().get_values("u")
    np.testing.assert_allclose(u_factory, u_solver, atol=1e-12)


def test_energy_problem_rejects_a_source(make_unit_square):
    space = FunctionSpace(make_unit_square(6), n_components=2)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    with pytest.raises(NotImplementedError):
        EnergyProblem(space, EnergyForm(StVenantKirchhoff(200, 0.4)), bc, source=1.0)
