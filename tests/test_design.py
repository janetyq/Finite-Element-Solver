"""SIMP density design: the model's diluted problem, the sensitivity filter, the OC
update, and the optimizer over them."""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.design import (
    DesignOptimizer, SIMPModel, TargetCompliance, calculate_smoothing_matrix,
    optimality_criteria_update,
)
from fem.equations import LinearElastic
from fem.forms import LinearElasticForm, PrecomputedForm
from fem.materials import LinearElasticMaterial
from fem.problem import LinearProblem
from fem.regions import on_plane
from fem.sensitivity import Compliance
from fem.solve import LinearSolve
from fem.space import FunctionSpace


def _cantilever_bc():
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])
    return bc


def _model(mesh, penalty=3.0, radius=None):
    equation = LinearElastic(E=1.0, nu=0.3)
    sensitivity_filter = calculate_smoothing_matrix(mesh, radius) if radius else None
    return SIMPModel(equation.space(mesh), equation, _cantilever_bc(), penalty=penalty,
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

    reference = LinearProblem(
        model.space,
        LinearElasticForm(LinearElasticMaterial(model.scaled_modulus(rho), model.equation.nu)),
        None, model.bc,
    )
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
        SIMPModel(equation.space(mesh), equation, _cantilever_bc())


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


# -- the OC update ---------------------------------------------------------------


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
    history = DesignOptimizer(model, Compliance(), volume_frac=0.5, iters=12).solve()
    assert history.objective[-1] < history.objective[0]


def test_design_optimizer_meets_the_volume_target(make_unit_square):
    model = _model(make_unit_square(10))
    design = DesignOptimizer(model, Compliance(), volume_frac=0.4, iters=15)
    design.solve()
    volumes = model.volumes
    achieved = float((volumes * design.rho).sum() / volumes.sum())
    assert abs(achieved - 0.4) < 1e-2


def test_design_optimizer_keeps_the_last_iterates_solution(make_unit_square):
    """`solution` is the `ElasticSolution` of the most recent iterate, with the stress of the
    diluted material, so a caller can post-process the design."""
    model = _model(make_unit_square(6))
    design = DesignOptimizer(model, volume_frac=0.5, iters=2)
    history = design.solve()
    assert design.solution is not None
    np.testing.assert_allclose(design.solution.u, history.u[-1])
    assert design.solution.stress.shape == (len(model.space.element_nodes), 3, 3)


def test_target_compliance_scores_the_squared_miss(make_unit_square):
    """J = (C - target)^2, with the adjoint load 2 (C - target) f."""
    model = _model(make_unit_square(5))
    problem = model.problem(np.ones(len(model.space.element_nodes)))
    u = LinearSolve().solve(problem)
    compliance = Compliance().value(problem, u)
    qoi = TargetCompliance(target=0.5 * compliance)

    assert np.isclose(qoi.value(problem, u), (0.5 * compliance) ** 2)
    np.testing.assert_allclose(qoi.dJ_du(problem, u), compliance * problem.load)
