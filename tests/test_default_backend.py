"""The backend a problem resolves to when given none, and the form hook it reads.

`default_backend` picks AMG-CG for a large system whose operator is symmetric positive
definite by construction, and the direct solve otherwise; `Form.symmetric_positive_definite`
is that construction, answered per form and composed through scaling and sums.
"""
import numpy as np

from fem.algebra import solve as solve_module
from fem.algebra.backends import DirectBackend, IterativeBackend
from fem.algebra.solve import default_backend
from fem.boundary import Robin
from fem.conditions import Conditions
from fem.loads import Source
from fem.physics.energies import StVenantKirchhoff
from fem.physics.equations import FiniteStrainElastic, LinearElastic, Poisson
from fem.physics.forms import (
    BoundaryMassForm, DiffusionForm, EnergyForm, GeometricStiffnessForm, LinearElasticForm,
    MassForm, PrecomputedForm,
)
from fem.physics.materials import LinearElasticMaterial
from fem.regions import everywhere
from helpers import cantilever_bc, pinned


def test_the_forms_that_are_spd_by_construction_say_so():
    material = LinearElasticMaterial(E=200, nu=0.3)
    assert MassForm(1).symmetric_positive_definite
    assert BoundaryMassForm(1, np.ones(4, dtype=bool)).symmetric_positive_definite
    assert DiffusionForm().symmetric_positive_definite
    assert DiffusionForm(3.0).symmetric_positive_definite
    assert DiffusionForm(lambda p: 1.0 + p[:, 0]).symmetric_positive_definite, 'taken on trust'
    assert not DiffusionForm(-1.0).symmetric_positive_definite
    assert LinearElasticForm(material).symmetric_positive_definite
    assert PrecomputedForm(np.zeros((3, 6, 6))).symmetric_positive_definite
    assert not PrecomputedForm(np.zeros((3, 6, 6)), symmetric_positive_definite=False).symmetric_positive_definite
    assert not GeometricStiffnessForm(np.zeros((3, 2, 2))).symmetric_positive_definite
    assert not EnergyForm(StVenantKirchhoff(200, 0.3)).symmetric_positive_definite


def test_scaling_and_sums_compose_the_hook():
    diffusion, mass = DiffusionForm(), MassForm(1)
    assert (2.0 * diffusion).symmetric_positive_definite
    assert not (-2.0 * diffusion).symmetric_positive_definite, 'a negative factor flips the sign'
    assert (diffusion + 0.5 * mass).symmetric_positive_definite
    assert not (diffusion + GeometricStiffnessForm(np.zeros((3, 2, 2)))).symmetric_positive_definite


def test_a_robin_problem_keeps_an_spd_operator(make_unit_square):
    """The Robin term is a scaled boundary mass, semidefinite, so the sum stays SPD."""
    bc = Conditions(Robin(everywhere(), kappa=2.0, g=1.0), Source(1.0))
    problem = Poisson().problem(make_unit_square(4), bc)
    assert problem.operator.symmetric_positive_definite


def test_small_problems_resolve_to_the_direct_backend(make_unit_square):
    mesh = make_unit_square(6)
    assert isinstance(default_backend(Poisson().problem(mesh, pinned())), DirectBackend)
    assert isinstance(default_backend(LinearElastic(E=200, nu=0.3).problem(mesh, cantilever_bc())), DirectBackend)


def test_large_spd_problems_resolve_to_amg_cg_with_the_near_kernel(make_unit_square, monkeypatch):
    """Above the threshold for its dimension an SPD problem is solved iteratively, and the
    problem's resolved backend carries the near-kernel its operator supplies."""
    monkeypatch.setitem(solve_module.ITERATIVE_ABOVE, 2, 10)
    mesh = make_unit_square(6)
    poisson = Poisson().problem(mesh, pinned() + Source(1.0))
    assert len(poisson.constraints[0]) > 10
    assert isinstance(default_backend(poisson), IterativeBackend)
    assert isinstance(poisson.backend, IterativeBackend)
    assert poisson.backend.near_null_space is None

    elastic = LinearElastic(E=200, nu=0.3).problem(mesh, cantilever_bc())
    resolved = elastic.backend
    assert isinstance(resolved, IterativeBackend)
    assert resolved.near_null_space is not None
    assert resolved.near_null_space.shape == (len(elastic.constraints[0]), 3)

    # The default agrees with the direct solve to CG's tolerance.
    direct = elastic.with_backend(DirectBackend()).solve().dofs
    np.testing.assert_allclose(elastic.solve().dofs, direct, atol=1e-8 * np.abs(direct).max())


def test_the_threshold_is_per_dimension(make_unit_square, monkeypatch):
    monkeypatch.setitem(solve_module.ITERATIVE_ABOVE, 2, 10**9)
    problem = Poisson().problem(make_unit_square(6), pinned())
    assert isinstance(default_backend(problem), DirectBackend)
    assert set(solve_module.ITERATIVE_ABOVE) == {2, 3}
    assert solve_module.ITERATIVE_ABOVE[3] < solve_module.ITERATIVE_ABOVE[2], 'direct fill grows faster in 3D'


def test_problems_that_are_not_spd_stay_direct(make_unit_square, monkeypatch):
    monkeypatch.setitem(solve_module.ITERATIVE_ABOVE, 2, 10)
    mesh = make_unit_square(6)
    nonlinear = FiniteStrainElastic(E=200, nu=0.4).problem(mesh, cantilever_bc())
    assert isinstance(default_backend(nonlinear), DirectBackend)
    unsigned = Poisson(coefficient=-1.0).problem(mesh, pinned())
    assert isinstance(default_backend(unsigned), DirectBackend)


def test_a_given_backend_is_honoured_regardless_of_size(make_unit_square, monkeypatch):
    monkeypatch.setitem(solve_module.ITERATIVE_ABOVE, 2, 10)
    problem = Poisson().problem(make_unit_square(6), pinned()).with_backend(DirectBackend())
    assert isinstance(problem.backend, DirectBackend)
