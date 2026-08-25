"""An equation names its own physics: `operator`, `energy_density`, and the refusals
where they do not apply.
"""
import numpy as np
import pytest

from fem.equations import Equation, LinearElastic, Poisson, Projection, StrainMeasure
from fem.forms import LaplacianForm, LinearElasticForm, MassForm


def test_projection_assembles_a_mass_matrix():
    """An L2 projection solves M u = b, so its operator is the mass form, carrying the
    component count."""
    operator = Projection().operator(n_components=2)

    assert isinstance(operator, MassForm)
    assert operator.n_components == 2


def test_scalar_family_shares_the_material_free_laplacian():
    """Poisson (and the Laplacian behind heat/wave) needs no material, so the base
    class answer is the right one and Poisson does not override it."""
    assert isinstance(Poisson().operator(n_components=1), LaplacianForm)
    assert isinstance(Equation().operator(n_components=1), LaplacianForm)


def test_linear_elastic_builds_its_form_from_its_own_constants():
    """The equation carries E and nu and turns them into the form the solver assembles."""
    operator = LinearElastic(E=210.0, nu=0.3).operator(n_components=2)

    assert isinstance(operator, LinearElasticForm)
    assert operator.material.E == 210.0
    assert operator.material.nu == 0.3


def test_finite_strain_has_no_bilinear_form():
    """A Green-Lagrange energy is not quadratic, so there is no constant stiffness
    to assemble."""
    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)

    with pytest.raises(NotImplementedError, match='small-strain'):
        equation.operator(n_components=2)


def test_scalar_equations_have_no_energy_density():
    """Only equations with a stored-energy formulation can be solved by minimising
    one; the scalar family raises rather than returning a stand-in density."""
    with pytest.raises(NotImplementedError, match='strain-energy density'):
        Poisson().energy_density()


def test_per_element_modulus_has_no_single_energy_density():
    """A density carries one pair of Lame parameters for the whole mesh, so a
    density-scaled modulus (TopologyOptimizer's) has no scalar answer here."""
    equation = LinearElastic(E=np.full(8, 200.0), nu=0.4)

    with pytest.raises(NotImplementedError, match='scalar Youngs modulus'):
        equation.energy_density()


def test_solver_refuses_finite_strain_through_the_equation_itself(make_unit_square):
    """A Green-Lagrange equation has no constant stiffness, so `Solver.solve` refuses with
    a message pointing at EnergySolver."""
    from fem.solver import Solver

    equation = LinearElastic(E=200, nu=0.4, kinematics=StrainMeasure.GREEN_LAGRANGE)
    solver = Solver(make_unit_square(4), equation)

    with pytest.raises(NotImplementedError, match='EnergySolver'):
        solver.solve()
