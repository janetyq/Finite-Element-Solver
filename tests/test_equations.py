"""An equation names its own physics: `operator`, `energy_density`, and the refusals
where they do not apply.
"""
import numpy as np
import pytest

from fem.equations import Equation, LinearElastic, Poisson, Projection, StrainMeasure
from fem.forms import LaplacianForm, LinearElasticForm, MassForm
from fem.space import FunctionSpace


def test_projection_assembles_a_mass_matrix(make_unit_square):
    """An L2 projection solves M u = b, so its operator is the mass form, carrying the
    space's component count."""
    space = FunctionSpace(make_unit_square(3), n_components=2)
    operator = Projection().operator(space)

    assert isinstance(operator, MassForm)
    assert operator.n_components == 2


def test_scalar_family_shares_the_material_free_laplacian(make_unit_square):
    """Poisson needs no material, so the base class answer is the right one and Poisson
    does not override it."""
    space = FunctionSpace(make_unit_square(3))
    assert isinstance(Poisson().operator(space), LaplacianForm)
    assert isinstance(Equation().operator(space), LaplacianForm)


def test_linear_elastic_builds_its_form_from_its_own_constants(make_unit_square):
    """The equation carries E and nu and turns them into the form the solver assembles."""
    space = FunctionSpace(make_unit_square(3), n_components=2)
    operator = LinearElastic(E=210.0, nu=0.3).operator(space)

    assert isinstance(operator, LinearElasticForm)
    assert operator.material.E == 210.0
    assert operator.material.nu == 0.3


def test_finite_strain_has_no_bilinear_form(make_unit_square):
    """A Green-Lagrange energy is not quadratic, so there is no constant stiffness
    to assemble."""
    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)

    with pytest.raises(NotImplementedError, match='no constant stiffness'):
        equation.operator(FunctionSpace(make_unit_square(3), n_components=2))


def test_wave_and_diffusion_name_their_operators(make_unit_square):
    """`Wave` is the c²-scaled Laplacian; `Diffusion` the coefficient form."""
    from fem.equations import Diffusion, Wave
    from fem.forms import DiffusionForm, ScaledForm

    space = FunctionSpace(make_unit_square(4))
    w = Wave(3.0).operator(space)
    assert isinstance(w, ScaledForm) and w.factor == 9.0

    assert isinstance(Diffusion(2.0).operator(space), DiffusionForm)


def test_scalar_equations_have_no_energy_density():
    """Only equations with a stored-energy formulation can be solved by minimising
    one; the scalar family raises rather than returning a stand-in density."""
    with pytest.raises(NotImplementedError, match='strain-energy density'):
        Poisson().energy_density()


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
    from fem.boundary import BCType, BoundaryConditions
    from fem.elements import QuadraticTriangleElement
    from fem.regions import everywhere

    mesh = make_unit_square(4)
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    assert Poisson().space(mesh).n_components == 1
    assert LinearElastic(E=1.0, nu=0.3).space(mesh).n_components == 2
    assert Poisson().space(mesh, QuadraticTriangleElement).element_type is QuadraticTriangleElement

    space = Poisson(source=2.0).space(mesh)
    problem = Poisson(source=2.0).problem(space, bc)
    assert isinstance(problem.operator, LaplacianForm)
    assert problem.space is space
    np.testing.assert_allclose(problem.load, 2.0 * space.mass_matrix @ np.ones(space.n_dofs))
    assert len(problem.constraints[1]) == len(np.unique(space.boundary_nodes))


def test_solver_refuses_finite_strain_through_the_equation_itself(make_unit_square):
    """A Green-Lagrange equation has no constant stiffness, so `Solver.solve` refuses with
    a message pointing at the energy path."""
    from fem.solver import Solver

    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    solver = Solver(make_unit_square(4), equation)

    with pytest.raises(NotImplementedError, match='minimising its energy'):
        solver.solve()
