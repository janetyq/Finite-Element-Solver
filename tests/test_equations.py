"""An equation names its own physics: the `operator` of its model and the refusals
where the physics does not apply.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet
from fem.equations import Equation, LinearElastic, Poisson, Projection, FiniteStrainElastic
from fem.forms import EnergyForm, DiffusionForm, LinearElasticForm, MassForm
from fem.regions import everywhere
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
    assert isinstance(Poisson().operator(space), DiffusionForm)
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
    """`Poisson`, `Heat`, and `Wave` share the diffusion operator and differ in the time
    orders they have a meaning for."""
    from fem.equations import Heat, Wave

    space = FunctionSpace(make_unit_square(4))
    w = Wave(stiffness=9.0, density=2.0).operator(space)
    assert isinstance(w, DiffusionForm) and w.coefficient == 9.0
    h = Heat(conductivity=2.0, capacity=3.0)
    assert isinstance(h.operator(space), DiffusionForm) and h.capacity == 3.0
    assert Poisson.time_orders == {0} and Heat.time_orders == {1} and Wave.time_orders == {2}
    assert LinearElastic.time_orders == {0, 2}


def test_solves_refuse_a_time_order_the_equation_has_no_meaning_for(make_unit_square):
    """A steady solve needs order 0, `ThetaMethod` order 1, `NewmarkMethod` and
    `ModalAnalysis` order 2; each refusal names the equation to use."""
    from fem.equations import Heat, Wave
    from fem.integrators import NewmarkMethod, ThetaMethod
    from fem.modal import ModalAnalysis

    mesh = make_unit_square(4)
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))
    heat, wave, poisson = Heat().problem(mesh, bc), Wave().problem(mesh, bc), Poisson().problem(mesh, bc)
    zero = np.zeros(heat.space.n_dofs)

    with pytest.raises(TypeError, match='Poisson'):
        heat.solve()
    with pytest.raises(TypeError, match='Poisson'):
        wave.solve()
    with pytest.raises(TypeError, match='Heat'):
        ThetaMethod(dt=0.1, steps=1).solve(poisson, zero)
    with pytest.raises(TypeError, match='Wave'):
        NewmarkMethod(dt=0.1, steps=1).solve(heat, zero, zero)
    with pytest.raises(TypeError, match='second-order'):
        ModalAnalysis(n_modes=1).solve(poisson)
    assert ThetaMethod(dt=0.1, steps=1).solve(heat, zero).t[-1] == 0.1
    assert NewmarkMethod(dt=0.1, steps=1).solve(wave, zero, zero).t[-1] == 0.1
    # A problem composed by hand allows every order.
    assert LinearProblem(heat.space, heat.physics).time_orders == {0, 1, 2}


def test_a_space_with_the_wrong_component_count_is_refused(make_unit_square):
    mesh = make_unit_square(4)
    with pytest.raises(ValueError, match='1-component'):
        Poisson().problem(FunctionSpace(mesh, n_components=2))


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
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))

    assert Poisson().space(mesh).n_components == 1
    assert LinearElastic(E=1.0, nu=0.3).space(mesh).n_components == 2
    assert Poisson().space(mesh, QuadraticTriangleElement).element_type is QuadraticTriangleElement

    space = Poisson(source=2.0).space(mesh)
    problem = Poisson(source=2.0).problem(space, bc)
    assert isinstance(problem.operator, DiffusionForm)
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
