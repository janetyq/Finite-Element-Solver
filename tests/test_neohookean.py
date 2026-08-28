"""Compressible Neo-Hookean elasticity on the shared `EnergyForm` machinery.

Neo-Hookean is the material the F-derivative interface exists for: written in the
invariants of C = FᵀF, it has no constant strain-measure Hessian to factor the way
St-Venant-Kirchhoff does, so it returns W, P = dW/dF, and A = d²W/dF² directly. These
checks pin the derivatives (finite differences), the frame indifference a finite-strain
law must have, its agreement with small strain in the small-strain limit, and an
end-to-end solve through the existing Newton path.
"""
import numpy as np
import pytest

from fem.boundary import BoundaryConditions, Dirichlet
from fem.elements import QuadraticTriangleElement
from fem.energies import NeohookeanEnergyDensity
from fem.equations import FiniteStrainElastic, LinearElastic
from fem.forms import EnergyForm
from fem.mesh.structured import box_mesh
from fem.numerics import central_difference_order
from fem.regions import on_plane
from fem.solution import ElasticSolution
from fem.solve import BacktrackingLineSearch, NewtonSolve
from fem.solver import Solver


@pytest.mark.parametrize('d', [2, 3])
def test_neohookean_stress_and_tangent_are_the_energy_derivatives(d):
    """P is the gradient of W and A the gradient of P, each to O(eps^2) under central
    differences: the definitive check that the hand-written invariant derivatives are
    consistent, in both 2D and 3D."""
    density = NeohookeanEnergyDensity(E=200.0, nu=0.3)
    rng = np.random.default_rng(0)
    grad0 = 0.1 * rng.standard_normal(d * d)

    def W(g):
        return float(density.evaluate(g.reshape(1, d, d)).W[0])

    def P(g):
        return density.evaluate(g.reshape(1, d, d)).P[0].ravel()

    p0 = density.evaluate(grad0.reshape(1, d, d)).P[0].ravel()
    a0 = density.evaluate(grad0.reshape(1, d, d)).A[0].reshape(d * d, d * d)

    grad_order = central_difference_order(W, lambda dv: p0 @ dv, grad0)
    hess_order = central_difference_order(P, lambda dv: a0 @ dv, grad0)
    assert 1.9 < grad_order < 2.1, f"P disagrees with dW/dF: order {grad_order:.3f}"
    assert 1.9 < hess_order < 2.1, f"A disagrees with dP/dF: order {hess_order:.3f}"


@pytest.mark.parametrize('d', [2, 3])
def test_neohookean_tangent_is_symmetric(d):
    """A = d²W/dF² is a Hessian, so it is symmetric under the (c,i) <-> (k,l) swap."""
    density = NeohookeanEnergyDensity(E=200.0, nu=0.3)
    grad = 0.1 * np.random.default_rng(1).standard_normal((3, d, d))
    A = density.evaluate(grad).A
    np.testing.assert_allclose(A, A.transpose(0, 3, 4, 1, 2), atol=1e-10)


def test_neohookean_is_frame_indifferent(make_unit_square):
    """A rigid rotation is strain-free: F = R gives J = 1 and I1 = d, so W = 0 and the
    stress vanishes. Evaluated directly on the rotation field, with no solve."""
    mesh = make_unit_square(8)
    center = mesh.vertices.mean(axis=0)
    space = FiniteStrainElastic(E=200, nu=0.3).space(mesh)
    form = EnergyForm(NeohookeanEnergyDensity(200, 0.3))

    for theta in (0.4, 0.2, 0.1):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        u = (mesh.vertices - center) @ R.T + center - mesh.vertices
        # Not the exact zero St-VK gives: the log and inverse carry roundoff a polynomial
        # energy does not, so this is machine-zero for an O(1) material rather than 1e-18.
        assert space.total_energy(form, u.flatten()) < 1e-12, (
            f"Neo-Hookean stored energy under a rigid {theta} rotation"
        )


def _stretched(make_unit_square, model, stretch, n=8, **kw):
    mesh = make_unit_square(n)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [stretch, 0]),
    )
    return Solver(mesh, model(E=200, nu=0.4, **kw), bc).solve().u


def test_neohookean_agrees_with_small_strain_to_second_order(make_unit_square):
    """Neo-Hookean and small strain share a linearisation, so their displacements agree
    to O(||grad u||^2): halving the imposed stretch shrinks the gap by ~4x."""
    gaps = []
    for stretch in (0.08, 0.04, 0.02, 0.01):
        u_small = _stretched(make_unit_square, LinearElastic, stretch)
        u_nh = _stretched(make_unit_square, FiniteStrainElastic, stretch,
                          law=NeohookeanEnergyDensity)
        gaps.append(np.linalg.norm(u_small.flatten() - u_nh.flatten()))

    ratios = [a / b for a, b in zip(gaps[:-1], gaps[1:])]
    for r in ratios:
        assert 3.5 < r < 4.5, f"gap ratio {r:.2f} is not the ~4x of a quadratic difference"


def test_neohookean_solve_converges_and_reports_stress(make_unit_square):
    """End to end: a finite Neo-Hookean stretch converges through Newton and comes back an
    ElasticSolution whose recovered von Mises is finite and positive, exercising the
    plane-strain out-of-plane stress reconstruction."""
    mesh = make_unit_square(10)
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 1.0), [0.3, 0.1]),
    )
    equation = FiniteStrainElastic(E=200, nu=0.4, law=NeohookeanEnergyDensity)
    solution = Solver(mesh, equation, bc,
                      strategy=NewtonSolve(line_search=BacktrackingLineSearch())).solve()

    assert isinstance(solution, ElasticSolution)
    vm = solution.nodal_von_mises()
    assert np.all(np.isfinite(vm))
    assert vm.max() > 0.0
    # A large stretch really engages the nonlinearity, so it disagrees with the small-strain
    # answer by well more than roundoff.
    u_small = Solver(mesh, LinearElastic(E=200, nu=0.4), bc).solve().u
    rel = np.linalg.norm(solution.u.flatten() - u_small.flatten()) / np.linalg.norm(u_small)
    assert rel > 1e-3, f"finite-strain answer should differ from small strain, got rel={rel:.2e}"


def test_neohookean_on_p2_converges(make_unit_square):
    """The material carries its own quadrature hint (degree 4), so it also assembles and
    converges on quadratic elements."""
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 4])
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Dirichlet(on_plane(0, 2.0), [0.3, 0.15]),
    )
    equation = FiniteStrainElastic(200.0, 0.3, law=NeohookeanEnergyDensity)
    solution = Solver(mesh, equation, bc, element_type=QuadraticTriangleElement).solve()

    assert isinstance(solution, ElasticSolution)
    assert solution.element_type is QuadraticTriangleElement
    assert np.all(np.isfinite(solution.nodal_von_mises()))
