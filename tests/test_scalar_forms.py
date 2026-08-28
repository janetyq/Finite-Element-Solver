"""One diffusion form and one volume source serve the scalar family: a constant
coefficient takes the element's own rule and equals the sampled path, and a `Source`
integrates a constant exactly and samples a callable.
"""
import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet
from fem.equations import Poisson
from fem.forms import DiffusionForm
from fem.loads import NodalSource, Source
from fem.regions import everywhere
from fem.space import FunctionSpace


def test_a_constant_coefficient_takes_the_elements_rule_and_scales_the_laplacian(make_unit_square):
    space = FunctionSpace(make_unit_square(5))
    constant = DiffusionForm(3.0)
    sampled = DiffusionForm(lambda p: 3.0)
    assert not constant.is_sampled and sampled.is_sampled
    assert constant.quadrature_degree(1) == 0 and sampled.quadrature_degree(1) == 2
    K1 = space.assemble(DiffusionForm()).toarray()
    np.testing.assert_allclose(space.assemble(constant).toarray(), 3.0 * K1, atol=1e-12)
    np.testing.assert_allclose(space.assemble(sampled).toarray(), 3.0 * K1, atol=1e-12)


def test_a_source_integrates_a_constant_exactly_and_samples_a_callable(make_unit_square):
    space = FunctionSpace(make_unit_square(6))
    constant = Source(2.0)
    assert not constant.is_sampled
    np.testing.assert_allclose(constant.vector(space), NodalSource(2.0).vector(space), atol=1e-14)
    np.testing.assert_allclose(constant.vector(space).sum(), 2.0, atol=1e-12)

    def peaked(p):
        return float(np.exp(-40 * np.sum((p - 0.5) ** 2)))

    sampled = Source(peaked)
    assert sampled.is_sampled
    assert not np.allclose(sampled.vector(space), NodalSource(peaked).vector(space))
    assert np.allclose(sampled.vector(space), space.assemble_load(sampled))


def test_the_named_equation_and_the_hand_composition_agree_on_a_varying_coefficient(make_unit_square):
    from fem.problem import LinearProblem
    mesh = make_unit_square(6)
    bc = BoundaryConditions(Dirichlet(everywhere(), 0.0))

    def kappa(p):
        return 1.0 + p[0] + p[1]

    named = Poisson(coefficient=kappa, source=1.0).problem(mesh, bc).solve().u
    composed = LinearProblem(FunctionSpace(mesh), DiffusionForm(kappa), 1.0, bc).solve().u
    np.testing.assert_allclose(named, composed, atol=1e-12)
