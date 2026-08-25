"""The density-based topology optimizer. (`test_topology.py` is mesh topology.)"""
import numpy as np
import pytest

from fem.boundary import BCType, BoundaryConditions
from fem.forms import LinearElasticForm, PrecomputedForm
from fem.materials import LinearElasticMaterial
from fem.problem import LinearProblem
from fem.solve import LinearSolve
from fem.space import FunctionSpace
from fem.topology import calculate_smoothing_matrix
from fem.regions import on_plane
from fem.equations import LinearElastic
from fem.topology import TargetCompliance, TopologyOptimizer


def _dense_smoothing_matrix(mesh, r):
    """The filter written out as an all-pairs distance matrix, the reference the sparse
    `calculate_smoothing_matrix` is checked against."""
    centers = mesh.vertices[mesh.elements].mean(axis=1)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    weights = np.maximum(0, r - distances)
    return weights / (weights.sum(axis=1)[:, None] + 1e-16)


def _optimizer(mesh, penalty):
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0.0, 0.0])
    bc.add(BCType.NEUMANN, on_plane(0, 1.0), [0.0, -1.0])
    return TopologyOptimizer(
        mesh, LinearElastic(E=1.0, nu=0.3), bc,
        iters=1, volume_frac=0.5, penalty=penalty,
    )


def test_simp_penalty_drives_the_modulus_scaling(make_unit_square):
    """E(rho) = rho^p E_0, with p the configured exponent rather than a
    literal 3 buried in the density update."""
    optimizer = _optimizer(make_unit_square(5), penalty=2.0)
    rho = np.full(len(optimizer.mesh.elements), 0.5)

    optimizer.set_rho(rho)

    assert np.allclose(optimizer.scaled_modulus, 0.5**2.0 * 1.0)


def test_min_compliance_sensitivity_uses_the_configured_penalty(make_unit_square):
    """The sensitivity p/rho * c is the derivative of the compliance only if p is the
    exponent the modulus scaling used; one configured exponent drives both."""
    penalty = 2.0
    optimizer = _optimizer(make_unit_square(5), penalty=penalty)
    solution = optimizer._solve()

    compliance = solution.compliance
    sensitivity = optimizer.objective.gradient(compliance, optimizer.rho, optimizer.penalty)

    assert np.allclose(sensitivity, compliance * penalty / optimizer.rho)


def test_density_scales_the_solid_element_stiffness(make_unit_square):
    """The identity the cached solid stiffness rests on: D is linear in E, so
    diluting the modulus to rho^p E_0 scales each element matrix by rho^p."""
    mesh = make_unit_square(6)
    space = FunctionSpace(mesh, n_components=2)
    rho = np.linspace(0.2, 1.0, len(mesh.elements))
    penalty, E_0, nu = 3.0, 200.0, 0.3

    solid = LinearElasticForm(LinearElasticMaterial(E_0, nu)).element_matrices(space.geometry)
    diluted = LinearElasticForm(
        LinearElasticMaterial(rho**penalty * E_0, nu)
    ).element_matrices(space.geometry)

    np.testing.assert_allclose(rho[:, None, None]**penalty * solid, diluted, rtol=1e-12)


def test_solve_matches_a_problem_built_from_the_scaled_material(make_unit_square):
    """The optimizer's rescaled cached matrices give the same displacement as one
    `LinearElasticForm` over `scaled_modulus` assembled from scratch."""
    optimizer = _optimizer(make_unit_square(6), penalty=3.0)
    optimizer.set_rho(np.linspace(0.3, 1.0, len(optimizer.mesh.elements)))

    solution = optimizer._solve()

    reference = LinearProblem(
        optimizer.space,
        LinearElasticForm(LinearElasticMaterial(optimizer.scaled_modulus, optimizer.nu)),
        optimizer.source,
        optimizer.bc,
    )
    np.testing.assert_allclose(solution.u, LinearSolve().solve(reference), atol=1e-10)


def test_precomputed_form_rejects_a_mismatched_geometry(make_unit_square):
    """Precomputed matrices carry no record of the geometry they were built on, so
    the element count is the one guard against assembling them over another mesh."""
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


def test_smoothing_matrix_matches_the_dense_cone_weights(make_unit_square):
    """Same weights as the all-pairs definition, including the diagonal: an
    element is at distance zero from itself, so it filters at full weight r."""
    mesh = make_unit_square(12)

    smoothing = calculate_smoothing_matrix(mesh, r=0.15)

    assert np.allclose(smoothing.toarray(), _dense_smoothing_matrix(mesh, r=0.15))


def test_smoothing_matrix_leaves_a_uniform_field_alone(make_unit_square):
    """Rows sum to 1, so filtering a constant sensitivity returns it unchanged, including
    along the boundary where the neighbourhood is one-sided."""
    mesh = make_unit_square(12)
    uniform = np.full(len(mesh.elements), 2.5)

    smoothing = calculate_smoothing_matrix(mesh, r=0.15)

    assert np.allclose(smoothing @ uniform, uniform)


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


def test_target_compliance_gradient_is_finite_per_element():
    """The target-compliance objective yields one finite sensitivity per element."""
    rng = np.random.default_rng(0)
    compliance, rho = rng.random(12) + 0.1, rng.random(12) + 0.1
    gradient = TargetCompliance(target=1.0).gradient(compliance, rho, penalty=3.0)
    assert gradient.shape == (12,) and np.all(np.isfinite(gradient))
