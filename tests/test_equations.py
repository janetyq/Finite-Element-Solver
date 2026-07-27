"""An equation names its own physics.

`Equation.operator` and `Equation.energy_density` are the two questions a solver
used to answer on the equation's behalf, through a module-level `stiffness_form`
and an `EnergySolver._select_energy`. Answering them polymorphically is what lets
`Solver._steady_problem` hold no "which PDE is this?" branch, so these pin the
mapping at the level it now lives.
"""
import numpy as np
import pytest

from fem.equations import Equation, LinearElastic, Poisson, Projection, StrainMeasure
from fem.forms import LaplacianForm, LinearElasticForm, MassForm


def test_projection_assembles_a_mass_matrix():
    """An L2 projection solves M u = b, so its operator is the mass form -- and it
    must carry the component count, which the form needs and the equation knows."""
    operator = Projection().operator(n_components=2)

    assert isinstance(operator, MassForm)
    assert operator.n_components == 2


def test_scalar_family_shares_the_material_free_laplacian():
    """Poisson (and the Laplacian behind heat/wave) needs no material, so the base
    class answer is the right one and Poisson does not override it."""
    assert isinstance(Poisson().operator(n_components=1), LaplacianForm)
    assert isinstance(Equation().operator(n_components=1), LaplacianForm)


def test_linear_elastic_builds_its_form_from_its_own_constants():
    """The equation carries E and nu, so it -- not a solver -- is what turns them
    into a material. A form built here must be the one the solver would assemble."""
    operator = LinearElastic(E=210.0, nu=0.3).operator(n_components=2)

    assert isinstance(operator, LinearElasticForm)
    assert operator.material.E == 210.0
    assert operator.material.nu == 0.3


def test_finite_strain_has_no_bilinear_form():
    """A Green-Lagrange energy is not quadratic, so there is no constant stiffness
    to assemble. Refusing here is what stops a finite-strain problem from being
    silently linearised by whichever solver picked up the equation."""
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
