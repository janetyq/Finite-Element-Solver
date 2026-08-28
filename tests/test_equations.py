"""An equation names its own physics: the `operator` of its model and the refusals
where the physics does not apply.
"""
import numpy as np
import pytest

from fem.equations import Equation, LinearElastic, Poisson, Projection, FiniteStrainElastic
from fem.forms import EnergyForm, LaplacianForm, LinearElasticForm, MassForm
from fem.problem import LinearProblem, Problem
from fem.space import FunctionSpace


def test_projection_assembles_a_mass_matrix(make_unit_square):
    """An L2 projection solves M u = b, so its operator is the mass form, carrying the
    space's component count."""
    space = FunctionSpace(make_unit_square(3), n_components=2)
    operator = Projection().operator(space)

    assert isinstance(operator, MassForm)
    assert operator.n_components == 2


def test_poisson_names_the_laplacian_and_the_base_equation_names_nothing(make_unit_square):
    """`Equation` is abstract: an operator comes from a named PDE."""
    space = FunctionSpace(make_unit_square(3))
    assert isinstance(Poisson().operator(space), LaplacianForm)
    with pytest.raises(NotImplementedError, match='operator'):
        Equation().operator(space)


def test_linear_elastic_builds_its_form_from_its_own_constants(make_unit_square):
    """The equation carries E and nu and turns them into the form the solver assembles."""
    space = FunctionSpace(make_unit_square(3), n_components=2)
    operator = LinearElastic(E=210.0, nu=0.3).operator(space)

    assert isinstance(operator, LinearElasticForm)
    assert operator.material.E == 210.0
    assert operator.material.nu == 0.3


def test_finite_strain_operator_is_an_energy_form(make_unit_square):
    """A Green-Lagrange energy is not quadratic, so its operator is the St-VK
    `EnergyForm`, its problem a `Problem` with a state-dependent tangent, and a
    `LinearProblem` over it is refused."""
    equation = FiniteStrainElastic(E=200, nu=0.4)
    space = FunctionSpace(make_unit_square(3), n_components=2)

    operator = equation.operator(space)
    assert isinstance(operator, EnergyForm)
    problem = equation.problem(space)
    assert type(problem) is Problem and not problem.is_linear
    assert isinstance(LinearElastic(E=200, nu=0.4).problem(space), LinearProblem)
    with pytest.raises(TypeError, match='state-dependent'):
        LinearProblem(space, operator)


def test_wave_and_diffusion_name_their_operators(make_unit_square):
    """`Wave` is the c²-scaled Laplacian; `Diffusion` the coefficient form."""
    from fem.equations import Diffusion, Wave
    from fem.forms import DiffusionForm, ScaledForm

    space = FunctionSpace(make_unit_square(4))
    w = Wave(3.0).operator(space)
    assert isinstance(w, ScaledForm) and w.factor == 9.0

    assert isinstance(Diffusion(2.0).operator(space), DiffusionForm)


def test_per_element_modulus_has_no_single_energy_density():
    """A density carries one pair of Lame parameters for the whole mesh, so a
    density-scaled modulus (SIMP's) has no scalar answer here."""
    equation = LinearElastic(E=np.full(8, 200.0), nu=0.4)

    with pytest.raises(NotImplementedError, match='scalar Youngs modulus'):
        equation.energy_density()


def test_equation_resolves_its_space_and_problem(make_unit_square):
    """`space` sizes the discretization from the field (one component for Poisson, one
    per spatial dimension for elasticity), and `problem` composes the equation's own
    operator, source, and the given constraints on it."""
    from fem.boundary import BoundaryConditions, Dirichlet
    from fem.elements import QuadraticTriangleElement
    from fem.regions import everywhere

    mesh = make_unit_square(4)
    bc = BoundaryConditions()
    bc = bc + Dirichlet(everywhere(), 0.0)

    assert Poisson().space(mesh).n_components == 1
    assert LinearElastic(E=1.0, nu=0.3).space(mesh).n_components == 2
    assert Poisson().space(mesh, QuadraticTriangleElement).element_type is QuadraticTriangleElement

    space = Poisson(source=2.0).space(mesh)
    problem = Poisson(source=2.0).problem(space, bc)
    assert isinstance(problem.operator, LaplacianForm)
    assert problem.space is space
    np.testing.assert_allclose(problem.load, 2.0 * space.mass_matrix @ np.ones(space.n_dofs))
    assert len(problem.constraints[1]) == len(np.unique(space.boundary_nodes))


def test_default_strategy_follows_the_tangent(make_unit_square):
    """A constant tangent gets `LinearSolve`; a state-dependent one gets line-searched
    `NewtonSolve`, paired with regularization only under an iterative backend."""
    from fem.backends import MinresBackend
    from fem.solve import LinearSolve, NewtonSolve, default_strategy

    mesh = make_unit_square(4)
    linear = LinearElastic(E=200, nu=0.4)
    finite = FiniteStrainElastic(E=200, nu=0.4)

    assert isinstance(default_strategy(linear.problem(mesh)), LinearSolve)
    newton = default_strategy(finite.problem(mesh))
    assert isinstance(newton, NewtonSolve)
    assert newton.line_search is not None and newton.regularization is None
    from fem.backends import DirectBackend
    direct = default_strategy(finite.problem(mesh), DirectBackend())
    assert isinstance(direct, NewtonSolve) and direct.regularization is None
    iterative = default_strategy(finite.problem(mesh), MinresBackend())
    assert isinstance(iterative, NewtonSolve) and iterative.regularization is not None
