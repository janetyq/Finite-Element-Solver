"""SIMP density design: the model's diluted problem, the sensitivity filter, the OC
update, and the optimizer over them."""
import numpy as np
import pytest
from helpers import cantilever_bc
from scipy.sparse import csr_array

from fem.algebra.solve import LinearSolve
from fem.analysis.design import (
    DesignOptimizer,
    SIMPModel,
    TargetCompliance,
    calculate_smoothing_matrix,
    filter_sensitivity,
    optimality_criteria_update,
)
from fem.analysis.sensitivity import Compliance
from fem.physics.equations import LinearElastic, Poisson
from fem.physics.forms import LinearElasticForm, PrecomputedForm
from fem.physics.materials import LinearElasticMaterial
from fem.problem import LinearProblem
from fem.space import FunctionSpace


def _model(mesh, penalty=3.0, radius=None):
    equation = LinearElastic(E=1.0, nu=0.3)
    sensitivity_filter = calculate_smoothing_matrix(mesh, radius) if radius else None
    return SIMPModel(equation.problem(mesh, cantilever_bc()), penalty=penalty,
                     sensitivity_filter=sensitivity_filter)


def _dense_smoothing_matrix(mesh, r):
    """The filter written out as an all-pairs distance matrix, the reference the sparse
    `calculate_smoothing_matrix` is checked against."""
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    weights = np.maximum(0, r - distances)
    return weights / (weights.sum(axis=1)[:, None] + 1e-16)


# -- the model -------------------------------------------------------------------


def test_simp_penalty_drives_the_modulus_scaling(make_unit_square):
    """E(rho) = rho^p E_0, with p the configured exponent."""
    model = _model(make_unit_square(5), penalty=2.0)
    rho = np.full(len(model.space.element_nodes), 0.5)
    assert np.allclose(model.scaled_modulus(rho), 0.5**2.0 * 1.0)


def test_density_scales_the_solid_element_stiffness(make_unit_square):
    """The identity the cached solid stiffness rests on: D is linear in E, so diluting
    the modulus to rho^p E_0 scales each element matrix by rho^p."""
    mesh = make_unit_square(6)
    space = FunctionSpace(mesh, n_components=2)
    rho = np.linspace(0.2, 1.0, len(mesh.elements))
    penalty, E_0, nu = 3.0, 200.0, 0.3

    solid = LinearElasticForm(LinearElasticMaterial(E_0, nu)).element_matrices(space.geometry)
    diluted = LinearElasticForm(
        LinearElasticMaterial(rho**penalty * E_0, nu)
    ).element_matrices(space.geometry)

    np.testing.assert_allclose(rho[:, None, None]**penalty * solid, diluted, rtol=1e-12)


def test_diluted_problem_matches_one_built_from_the_scaled_material(make_unit_square):
    """`SIMPModel.problem(rho)` rescales cached matrices; it solves to the same displacement
    as a `LinearElasticForm` over the scaled modulus assembled from scratch."""
    model = _model(make_unit_square(6))
    rho = np.linspace(0.3, 1.0, len(model.space.element_nodes))

    reference = LinearProblem(model.space, LinearElasticForm(LinearElasticMaterial(model.scaled_modulus(rho), 0.3)), model.template.conditions)
    np.testing.assert_allclose(
        LinearSolve().solve(model.problem(rho)), LinearSolve().solve(reference), atol=1e-10)


def test_precomputed_form_rejects_a_mismatched_geometry(make_unit_square):
    """Precomputed matrices carry no record of the geometry they were built on, so the
    element count is the one guard against assembling them over another mesh."""
    space = FunctionSpace(make_unit_square(6), n_components=2)
    other = FunctionSpace(make_unit_square(8), n_components=2)
    matrices = LinearElasticForm(
        LinearElasticMaterial(200.0, 0.3)
    ).element_matrices(space.geometry)

    np.testing.assert_allclose(
        space.assemble(PrecomputedForm(matrices)).toarray(),
        space.assemble(LinearElasticForm(LinearElasticMaterial(200.0, 0.3))).toarray(),
        atol=1e-9,
    )
    with pytest.raises(ValueError, match='elements'):
        other.assemble(PrecomputedForm(matrices))


def test_model_rejects_a_per_element_modulus(make_unit_square):
    mesh = make_unit_square(4)
    equation = LinearElastic(E=np.ones(len(mesh.elements)), nu=0.3)
    with pytest.raises(ValueError, match='scalar'):
        SIMPModel(equation.problem(mesh, cantilever_bc()))


def test_model_rejects_an_operator_that_is_not_small_strain_elastic(make_unit_square):
    mesh = make_unit_square(4)
    with pytest.raises(TypeError, match='small-strain'):
        SIMPModel(Poisson().problem(mesh))


# -- the sensitivity filter ------------------------------------------------------


def test_smoothing_matrix_matches_the_dense_cone_weights(make_unit_square):
    """Same weights as the all-pairs definition, including the diagonal: an element is at
    distance zero from itself, so it filters at full weight r."""
    mesh = make_unit_square(12)
    smoothing = calculate_smoothing_matrix(mesh, r=0.15)
    assert np.allclose(smoothing.toarray(), _dense_smoothing_matrix(mesh, r=0.15))


def test_smoothing_matrix_leaves_a_uniform_field_alone(make_unit_square):
    """Rows sum to 1, so filtering a constant sensitivity returns it unchanged, including
    along the boundary where the neighbourhood is one-sided."""
    mesh = make_unit_square(12)
    uniform = np.full(len(mesh.elements), 2.5)
    assert np.allclose(calculate_smoothing_matrix(mesh, r=0.15) @ uniform, uniform)


def test_smoothing_matrix_couples_only_within_the_radius(make_unit_square):
    """No weight reaches past r, which keeps the matrix sparse and pins the feature size."""
    mesh = make_unit_square(12)
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    r = 0.15
    smoothing = calculate_smoothing_matrix(mesh, r).tocoo()
    nonzero = smoothing.data != 0
    reach = np.linalg.norm(centers[smoothing.row[nonzero]] - centers[smoothing.col[nonzero]], axis=1)
    assert reach.max() < r


def test_smoothing_matrix_stays_sparse_under_refinement(make_unit_square):
    """A radius tracking the element size keeps the entries per row bounded as the mesh
    refines, so the filter costs O(n_elements)."""
    coarse = calculate_smoothing_matrix(make_unit_square(20), r=3.0 / 20)
    fine = calculate_smoothing_matrix(make_unit_square(40), r=3.0 / 40)
    assert fine.shape[0] > 4 * coarse.shape[0]
    assert fine.nnz / fine.shape[0] < 1.5 * coarse.nnz / coarse.shape[0]


def test_filter_on_a_uniform_density_is_the_plain_weighted_mean(make_unit_square):
    """Sigmund's filter weights each neighbour's sensitivity by its density and divides by
    the element's own; with rho uniform the two cancel and the plain cone mean is left."""
    mesh = make_unit_square(12)
    weights = calculate_smoothing_matrix(mesh, r=0.15)
    rho = np.full(len(mesh.elements), 0.5)
    sensitivity = np.linspace(1.0, 2.0, len(mesh.elements))
    np.testing.assert_allclose(filter_sensitivity(weights, rho, sensitivity), weights @ sensitivity)


def test_filter_matches_the_hand_computed_sigmund_value():
    """s_i = sum_j w_ij rho_j s_j / (rho_i sum_j w_ij), on three elements written out."""
    weights = np.array([[0.5, 0.25, 0.25],
                        [0.25, 0.5, 0.25],
                        [0.25, 0.25, 0.5]])
    rho = np.array([1.0, 0.5, 0.2])
    sensitivity = np.array([2.0, 4.0, 8.0])
    expected = (weights @ (rho * sensitivity)) / rho
    np.testing.assert_allclose(
        filter_sensitivity(csr_array(weights), rho, sensitivity), expected)
    # The first entry by hand: (0.5*1*2 + 0.25*0.5*4 + 0.25*0.2*8) / 1 = 1.9.
    assert np.isclose(expected[0], 1.9)


def test_filter_floors_a_void_density():
    """A density at the OC floor divides by the floor, not by zero."""
    weights = csr_array(np.eye(2))
    rho = np.array([0.0, 1.0])
    filtered = filter_sensitivity(weights, rho, np.array([1.0, 1.0]))
    assert np.all(np.isfinite(filtered)) and filtered[1] == 1.0


# -- the OC update ---------------------------------------------------------------


def test_optimality_criteria_update_moves_a_larger_element_less():
    """The condition balances the sensitivity against the volume it buys, dV/drho_e = v_e,
    so at equal sensitivity the larger element (more volume per unit density) grows less.
    With equal volumes the volumes drop out and both elements move alike."""
    rho = np.array([0.5, 0.5])
    sensitivity = np.array([1.0, 1.0])
    unequal = optimality_criteria_update(rho, sensitivity, np.array([1.0, 4.0]),
                                         volume_frac=0.6, move=0.5)
    assert unequal[0] > unequal[1]
    equal = optimality_criteria_update(rho, sensitivity, np.array([2.0, 2.0]),
                                       volume_frac=0.6, move=0.5)
    np.testing.assert_allclose(equal, [0.6, 0.6], atol=1e-6)


def test_optimality_criteria_update_is_unchanged_by_a_uniform_volume_scale():
    """Volumes enter relative to their mean, so scaling every element by the same factor
    leaves the step exactly alone."""
    rho = np.linspace(0.2, 0.8, 8)
    sensitivity = np.linspace(1.0, 3.0, 8)
    small = optimality_criteria_update(rho, sensitivity, np.ones(8), volume_frac=0.5)
    large = optimality_criteria_update(rho, sensitivity, np.full(8, 7.0), volume_frac=0.5)
    np.testing.assert_allclose(small, large)




def test_optimality_criteria_update_rejects_a_signed_sensitivity():
    """OC needs a nonnegative (compliance-type) sensitivity; a signed one, such as a raw
    point-displacement objective would give, must fail loudly rather than take a NaN
    step. The point-value adjoint gradient itself is validated in test_sensitivity.py."""
    rho = np.full(10, 0.5)
    volumes = np.ones(10)
    sensitivity = np.linspace(-1.0, 1.0, 10)
    with pytest.raises(ValueError, match='nonnegative'):
        optimality_criteria_update(rho, sensitivity, volumes, volume_frac=0.4)


def test_optimality_criteria_update_hits_the_volume_fraction():
    rho = np.full(20, 0.5)
    sensitivity = np.linspace(1.0, 2.0, 20)
    volumes = np.ones(20)
    updated = optimality_criteria_update(rho, sensitivity, volumes, volume_frac=0.4)
    assert abs(updated.mean() - 0.4) < 1e-3
    assert np.all(updated >= 1e-6) and np.all(updated <= 1.0)


# -- the optimizer ---------------------------------------------------------------


def test_design_optimizer_reduces_compliance(make_unit_square):
    model = _model(make_unit_square(10), radius=0.12)
    history = DesignOptimizer(model, Compliance(), volume_frac=0.5, iters=12).run()
    assert history.objective[-1] < history.objective[0]


def test_design_optimizer_reports_each_iteration_to_the_callback(make_unit_square):
    """on_iteration(i, rho, J) fires once per iteration, in order, with that step's
    density (one value per element) and its objective; the reported objectives are the
    history's."""
    model = _model(make_unit_square(6))
    calls = []
    design = DesignOptimizer(model, Compliance(), volume_frac=0.5, iters=4)
    history = design.run(on_iteration=lambda i, rho, j: calls.append((i, np.asarray(rho).copy(), j)))

    assert [i for i, _, _ in calls] == [0, 1, 2, 3]
    assert all(rho.shape == (len(model.volumes),) for _, rho, _ in calls)
    np.testing.assert_allclose([j for _, _, j in calls], history.objective)


def test_design_optimizer_meets_the_volume_target(make_unit_square):
    model = _model(make_unit_square(10))
    design = DesignOptimizer(model, Compliance(), volume_frac=0.4, iters=15)
    design.run()
    volumes = model.volumes
    achieved = float((volumes * design.rho).sum() / volumes.sum())
    assert abs(achieved - 0.4) < 1e-2


def test_design_optimizer_keeps_the_last_iterates_solution(make_unit_square):
    """`solution` is the `ElasticSolution` of the most recent iterate, with the stress of the
    diluted material, so a caller can post-process the design."""
    model = _model(make_unit_square(6))
    design = DesignOptimizer(model, volume_frac=0.5, iters=2)
    history = design.run()
    assert design.solution is not None
    np.testing.assert_allclose(design.solution.dofs, history.dofs[-1])
    assert design.solution.stress.shape == (len(model.space.element_nodes), 3, 3)
    # The history is a series of those solutions, one per iterate.
    assert len(history) == 2 and history.dofs.shape == (2, model.space.n_dofs)
    last = history[-1]
    np.testing.assert_allclose(last.dofs, design.solution.dofs)
    np.testing.assert_allclose(last.stress, design.solution.stress)
    assert all(step.stress.shape == last.stress.shape for step in history)


def test_target_compliance_scores_the_squared_miss(make_unit_square):
    """J = (C - target)^2, with the adjoint load 2 (C - target) f."""
    model = _model(make_unit_square(5))
    problem = model.problem(np.ones(len(model.space.element_nodes)))
    u = LinearSolve().solve(problem)
    compliance = Compliance().value(problem, u)
    qoi = TargetCompliance(target=0.5 * compliance)

    assert np.isclose(qoi.value(problem, u), (0.5 * compliance) ** 2)
    np.testing.assert_allclose(qoi.dJ_du(problem, u), compliance * problem.load)
