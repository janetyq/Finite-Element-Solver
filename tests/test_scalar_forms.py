"""One diffusion form and one volume source serve the scalar family: a constant
coefficient takes the element's own rule and equals the sampled path, and a `Source`
integrates a constant exactly and samples a callable.
"""
import numpy as np
from helpers import pinned

from fem.loads import Source
from fem.physics.equations import Poisson
from fem.physics.forms import DiffusionForm
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
    np.testing.assert_allclose(constant.vector(space), Source(2.0, nodal=True).vector(space), atol=1e-14)
    np.testing.assert_allclose(constant.vector(space).sum(), 2.0, atol=1e-12)

    def peaked(p):
        return np.exp(-40 * np.sum((p - 0.5) ** 2, axis=1))

    sampled = Source(peaked)
    assert sampled.is_sampled
    assert not np.allclose(sampled.vector(space), Source(peaked, nodal=True).vector(space))
    assert np.allclose(sampled.vector(space), space.assemble_load(sampled))


def test_the_named_equation_and_the_hand_composition_agree_on_a_varying_coefficient(make_unit_square):
    from fem.problem import LinearProblem
    mesh = make_unit_square(6)
    bc = pinned()

    def kappa(p):
        return 1.0 + p[:, 0] + p[:, 1]

    named = Poisson(coefficient=kappa).problem(mesh, bc + Source(1.0)).solve().dofs
    composed = LinearProblem(FunctionSpace(mesh), DiffusionForm(kappa), bc + Source(1.0)).solve().dofs
    np.testing.assert_allclose(named, composed, atol=1e-12)
