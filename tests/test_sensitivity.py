"""The adjoint sensitivity core, anchored two ways: the general adjoint pass with a
`Compliance` quantity of interest reproduces the hand-written SIMP compliance
sensitivity, and the adjoint gradient matches a central-difference gradient for both a
self-adjoint and a non-self-adjoint quantity of interest.
"""
import numpy as np

from fem.boundary import BoundaryConditions, Dirichlet, Neumann
from fem.forms import LinearElasticForm, PrecomputedForm
from fem.materials import LinearElasticMaterial
from fem.problem import LinearProblem
from fem.regions import on_plane
from fem.sensitivity import (
    Compliance,
    DensityField,
    ModulusField,
    PointValue,
    SensitivityAnalysis,
)
from fem.space import FunctionSpace


def _cantilever_bc():
    """Clamp the left edge, pull the right edge down: homogeneous supports, so the
    compliance shortcut (lambda = u) is exact."""
    bc = BoundaryConditions(
        Dirichlet(on_plane(0, 0.0), [0.0, 0.0]),
        Neumann(on_plane(0, 1.0), [0.0, -1.0]),
    )
    return bc


def _density_problem(space, rho, base_E, nu, penalty):
    """The elastic problem at density `rho`, stiffness rescaled by rho^p (as SIMP does)."""
    K0 = LinearElasticForm(LinearElasticMaterial(base_E, nu)).element_matrices(space.geometry)
    stiffness = PrecomputedForm((rho**penalty)[:, None, None] * K0)
    return LinearProblem(space, stiffness, None, _cantilever_bc())


def test_compliance_density_gradient_matches_the_hand_written_sensitivity(make_unit_square):
    """The general adjoint pass equals the closed-form SIMP sensitivity p/rho * c_e up to
    sign (the core returns dC/drho, the formula the positive ascent sensitivity)."""
    space = FunctionSpace(make_unit_square(6), n_components=2)
    base_E, nu, penalty = 1.0, 0.3, 3.0
    rho = np.linspace(0.3, 1.0, len(space.element_nodes))

    problem = _density_problem(space, rho, base_E, nu, penalty)
    analysis = SensitivityAnalysis(problem)
    u = analysis.solve_forward()

    parameterization = DensityField.create(space, rho, base_E, nu, penalty)
    core_gradient = analysis.gradient(Compliance(), parameterization, u)

    # The hand-written formula needs the per-element compliance u_e^T K_e u_e.
    form = LinearElasticForm(LinearElasticMaterial(rho**penalty * base_E, nu))
    compliance = form.derived_fields(
        space.geometry, u.reshape(-1, 2)[space.element_nodes]
    ).compliance
    # For E = rho^p E_0 the element compliance is linear in E, so dc_e/drho = p/rho * c_e.
    hand_written = compliance * penalty / rho

    np.testing.assert_allclose(core_gradient, -hand_written, rtol=1e-10, atol=1e-12)


def test_compliance_shortcut_equals_an_explicit_adjoint_solve(make_unit_square):
    """The self-adjoint shortcut (lambda = u) gives the same gradient as actually
    solving the adjoint system, under homogeneous supports."""
    space = FunctionSpace(make_unit_square(5), n_components=2)
    rho = np.full(len(space.element_nodes), 0.7)
    problem = _density_problem(space, rho, 1.0, 0.3, 3.0)
    analysis = SensitivityAnalysis(problem)
    u = analysis.solve_forward()

    shortcut = analysis.adjoint(Compliance(), u)
    explicit = analysis._system.solve_homogeneous(Compliance().dJ_du(problem, u))

    np.testing.assert_allclose(shortcut, explicit, rtol=1e-9, atol=1e-11)


def _fd_gradient(objective_value, p0, eps):
    """Central-difference gradient of a scalar objective over a parameter vector."""
    grad = np.zeros(len(p0))
    for i in range(len(p0)):
        plus, minus = p0.copy(), p0.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (objective_value(plus) - objective_value(minus)) / (2 * eps)
    return grad


def test_compliance_density_gradient_matches_finite_differences(make_unit_square):
    space = FunctionSpace(make_unit_square(4), n_components=2)
    base_E, nu, penalty = 1.0, 0.3, 3.0
    rho0 = np.linspace(0.4, 0.9, len(space.element_nodes))

    def objective(rho):
        problem = _density_problem(space, rho, base_E, nu, penalty)
        analysis = SensitivityAnalysis(problem)
        u = analysis.solve_forward()
        return Compliance().value(problem, u)

    problem = _density_problem(space, rho0, base_E, nu, penalty)
    analysis = SensitivityAnalysis(problem)
    u = analysis.solve_forward()
    adjoint_grad = analysis.gradient(Compliance(), DensityField.create(space, rho0, base_E, nu, penalty), u)

    fd_grad = _fd_gradient(objective, rho0, eps=1e-6)
    np.testing.assert_allclose(adjoint_grad, fd_grad, rtol=1e-5, atol=1e-7)


def test_point_displacement_gradient_matches_finite_differences(make_unit_square):
    """A non-self-adjoint QoI: the adjoint solve is exercised, not the lambda = u shortcut."""
    space = FunctionSpace(make_unit_square(4), n_components=2)
    nu = 0.3
    E0 = np.linspace(0.5, 1.5, len(space.element_nodes))
    # The vertical DOF of a loaded right-edge node: an interior objective of the field.
    tip_dof = _rightmost_vertical_dof(space)
    qoi = PointValue(tip_dof)

    def modulus_problem(E):
        K0 = LinearElasticForm(LinearElasticMaterial(1.0, nu)).element_matrices(space.geometry)
        stiffness = PrecomputedForm(E[:, None, None] * K0)
        return LinearProblem(space, stiffness, None, _cantilever_bc())

    def objective(E):
        problem = modulus_problem(E)
        analysis = SensitivityAnalysis(problem)
        u = analysis.solve_forward()
        return qoi.value(problem, u)

    problem = modulus_problem(E0)
    analysis = SensitivityAnalysis(problem)
    u = analysis.solve_forward()
    adjoint_grad = analysis.gradient(qoi, ModulusField.create(space, E0, nu), u)

    fd_grad = _fd_gradient(objective, E0, eps=1e-6)
    np.testing.assert_allclose(adjoint_grad, fd_grad, rtol=1e-5, atol=1e-7)


def _rightmost_vertical_dof(space):
    coords = space.node_coords
    node = int(np.argmax(coords[:, 0]))
    return node * space.n_components + 1
