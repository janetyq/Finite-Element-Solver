"""The stress-based quantities of interest: the adjoint load `dJ_du` matches a
central-difference gradient of the measure, and the value is cross-checked against
`invariants.von_mises`.
"""
import numpy as np
import pytest

from fem import invariants
from fem.boundary import BCType, BoundaryConditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.forms import LinearElasticForm, PrecomputedForm
from fem.materials import LinearElasticMaterial
from fem.problem import LinearProblem
from fem.regions import on_plane
from fem.sensitivity import MeanStress, SensitivityAnalysis, SoftMaxStress, _VonMisesStress
from fem.solution import ElasticSolution
from fem.space import FunctionSpace


def _cantilever_bc(w):
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, w), [0.0, -1.0])
    return bc


def _modulus_problem(space, E, nu, bc):
    K0 = LinearElasticForm(LinearElasticMaterial(1.0, nu)).element_matrices(space.geometry)
    return LinearProblem(space, PrecomputedForm(E[:, None, None] * K0), None, bc)


def _solved(space, nu):
    """A cantilever solved on a uniform-modulus mesh; returns (problem, u, material)."""
    E = np.ones(len(space.element_nodes))
    bc = _cantilever_bc(1.0)
    problem = _modulus_problem(space, E, nu, bc)
    u = SensitivityAnalysis(problem).solve_forward()
    return problem, u, LinearElasticMaterial(1.0, nu)


def _fd_dJ_du(qoi, problem, u, eps):
    """Central-difference gradient of the measure with respect to the displacement DOFs."""
    grad = np.zeros(len(u))
    for i in range(len(u)):
        plus, minus = u.copy(), u.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (qoi.value(problem, plus) - qoi.value(problem, minus)) / (2 * eps)
    return grad


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_mean_stress_adjoint_load_matches_finite_differences(make_unit_square, element_type):
    space = FunctionSpace(make_unit_square(4), element_type, n_components=2)
    problem, u, material = _solved(space, nu=0.3)
    qoi = MeanStress(space, material)

    np.testing.assert_allclose(
        qoi.dJ_du(problem, u), _fd_dJ_du(qoi, problem, u, eps=1e-6), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_soft_max_stress_adjoint_load_matches_finite_differences(make_unit_square, element_type):
    space = FunctionSpace(make_unit_square(4), element_type, n_components=2)
    problem, u, material = _solved(space, nu=0.3)
    qoi = SoftMaxStress(space, material, p=6.0)

    np.testing.assert_allclose(
        qoi.dJ_du(problem, u), _fd_dJ_du(qoi, problem, u, eps=1e-6), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_mean_stress_value_matches_von_mises_invariant(make_unit_square, element_type):
    """The measure equals the volume-weighted mean of the invariant von Mises stress
    recovered by ElasticSolution, so it is anchored to the real stress state. On P2 both
    read the element-mean stress, so the same anchor holds."""
    space = FunctionSpace(make_unit_square(6), element_type, n_components=2)
    problem, u, material = _solved(space, nu=0.3)

    solution = ElasticSolution.from_solve(space, u, LinearElasticForm(material))
    weights = space.element_volumes / space.element_volumes.sum()
    expected = float(weights @ invariants.von_mises(solution.stress))

    assert abs(MeanStress(space, material).value(problem, u) - expected) < 1e-9


def test_soft_max_between_mean_and_peak(make_unit_square):
    """The p-norm sits between the mean and the true peak, and rises toward the peak as p
    grows: the property that makes it a peak-stress stand-in."""
    space = FunctionSpace(make_unit_square(6), n_components=2)
    problem, u, material = _solved(space, nu=0.3)

    peak = float(_VonMisesStress(space, material).von_mises(u).max())
    mean = MeanStress(space, material).value(problem, u)
    soft6 = SoftMaxStress(space, material, p=6.0).value(problem, u)
    soft16 = SoftMaxStress(space, material, p=16.0).value(problem, u)

    assert mean <= soft6 <= soft16 <= peak + 1e-9
    assert soft16 > soft6


def test_region_restricts_the_measure(make_unit_square):
    """A region mask limits the measure to selected elements; a mask covering everything
    reproduces the unrestricted measure."""
    space = FunctionSpace(make_unit_square(6), n_components=2)
    problem, u, material = _solved(space, nu=0.3)

    full = MeanStress(space, material).value(problem, u)
    all_true = np.ones(len(space.element_nodes), dtype=bool)
    assert abs(full - MeanStress(space, material, region=all_true).value(problem, u)) < 1e-12

    centers = space.node_coords[space.element_nodes].mean(axis=1)
    left = centers[:, 0] < 0.5   # near the clamp, where the stress is higher
    assert MeanStress(space, material, region=left).value(problem, u) > full
