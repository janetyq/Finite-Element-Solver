"""Nodal stress on P2: the within-element variation reaches the nodes.

A P2 displacement has a stress that varies within each element. `ElasticSolution.stress`
keeps one tensor per element (the element mean); `nodal_stress` re-evaluates the form
at the nodes (`'average'`) or quadrature points (`'l2'`) so a rim or corner value is
read from the boundary itself rather than from an interior sample.
"""
import numpy as np
import pytest

from fem.elements import (
    LinearLineElement,
    LinearTetrahedralElement,
    LinearTriangleElement,
    QuadraticLineElement,
    QuadraticTriangleElement,
)
from fem.physics.forms import LinearElasticForm
from fem.physics.materials import Enu_to_Lame, LinearElasticMaterial
from fem.mesh.structured import box_mesh
from fem.post.recovery import average_to_nodal, recover_nodal
from fem.post.solution import ElasticSolution
from fem.space import FunctionSpace


def _quadratic_displacement_solution(element_type, E=200.0, nu=0.3):
    """An `ElasticSolution` whose displacement is a known quadratic field, so its stress
    is linear, which a P2 space carries exactly. Built from the nodal values
    rather than solved, so the discretization error is zero by construction."""
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 4])
    space = FunctionSpace(mesh, element_type, n_components=2)
    x, y = space.node_coords.T
    u = np.column_stack([0.01 * x**2 + 0.02 * x * y, -0.015 * y**2 + 0.005 * x * y]).ravel()
    form = LinearElasticForm(LinearElasticMaterial(E, nu))
    solution = ElasticSolution.from_solve(space, u, form)

    mu, lamb = Enu_to_Lame(E, nu)

    def exact_stress(points):
        px, py = np.asarray(points).T
        exx = 0.02 * px + 0.02 * py
        eyy = -0.03 * py + 0.005 * px
        exy = 0.5 * (0.02 * px + 0.005 * py)
        trace = exx + eyy
        stress = np.zeros((len(px), 3, 3))
        stress[:, 0, 0] = 2 * mu * exx + lamb * trace
        stress[:, 1, 1] = 2 * mu * eyy + lamb * trace
        stress[:, 0, 1] = stress[:, 1, 0] = 2 * mu * exy
        stress[:, 2, 2] = lamb * trace     # plane strain
        return stress

    return solution, exact_stress


@pytest.mark.parametrize('method', ['average', 'l2'])
def test_p2_nodal_stress_is_exact_for_a_linear_stress_field(method):
    """Reading a linear stress at the nodes reproduces it at every node, boundary
    included: the recovery reads the element's own stress at the node."""
    solution, exact_stress = _quadratic_displacement_solution(QuadraticTriangleElement)
    nodal = solution.nodal_stress(method=method)
    np.testing.assert_allclose(nodal, exact_stress(solution.space.node_coords), atol=1e-9)


def test_p2_per_element_stress_is_the_centroid_value():
    """`solution.stress` is the element mean, which for a stress linear within the
    element is its value at the centroid."""
    solution, exact_stress = _quadratic_displacement_solution(QuadraticTriangleElement)
    mesh = solution.mesh
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    np.testing.assert_allclose(solution.stress, exact_stress(centroids), atol=1e-9)


def test_p2_nodal_stress_reaches_a_boundary_extreme_the_element_mean_misses():
    """On a stress that grows toward the domain edge, recovering the per-element tensor
    sits below the true edge value; the nodal evaluation reaches it."""
    solution, exact_stress = _quadratic_displacement_solution(QuadraticTriangleElement)
    space = solution.space
    edge = np.flatnonzero(np.isclose(space.node_coords[:, 0], 2.0))
    exact = exact_stress(space.node_coords[edge])[:, 0, 0]
    from_nodes = solution.nodal_stress('average')[edge, 0, 0]
    from_elements = recover_nodal(space, solution.stress, method='average')[edge, 0, 0]
    np.testing.assert_allclose(from_nodes, exact, atol=1e-9)
    assert np.all(from_elements < exact - 1e-3)


@pytest.mark.parametrize('method', ['average', 'l2'])
def test_p1_nodal_stress_is_unchanged_by_the_form(method):
    """For P1 the stress is constant per element, so evaluating at the nodes and
    recovering the per-element tensor are the same number."""
    solution, _ = _quadratic_displacement_solution(LinearTriangleElement)
    np.testing.assert_allclose(solution.nodal_stress(method=method),
                               recover_nodal(solution.space, solution.stress, method=method))
    np.testing.assert_allclose(solution.nodal_strain(method=method),
                               recover_nodal(solution.space, solution.strain, method=method))


def test_a_loaded_solution_recovers_from_its_per_element_tensors(tmp_path):
    """The form is not persisted; a loaded solution still answers `nodal_stress`, from
    the per-element tensors it does carry."""
    solution, _ = _quadratic_displacement_solution(QuadraticTriangleElement)
    path = tmp_path / 'solution.npz'
    solution.save(str(path))
    loaded = ElasticSolution.load(str(path))
    assert isinstance(loaded, ElasticSolution)
    assert loaded.form is None
    np.testing.assert_allclose(loaded.nodal_stress(),
                               recover_nodal(loaded.space, loaded.stress))
    assert not np.allclose(loaded.nodal_stress(), solution.nodal_stress())


# --- reference nodes and the nodal geometry ---

@pytest.mark.parametrize('element_type', [
    LinearLineElement, LinearTriangleElement, LinearTetrahedralElement,
    QuadraticLineElement, QuadraticTriangleElement,
])
def test_reference_nodes_are_where_the_basis_is_nodal(element_type):
    """Each shape function is 1 at its own reference node and 0 at the others."""
    nodes = element_type.reference_nodes()
    assert nodes.shape == (element_type.N, element_type.reference_dim())
    np.testing.assert_allclose(element_type.shape_values(nodes), np.eye(element_type.N),
                               atol=1e-12)


def test_geometry_at_nodes_places_its_points_on_the_element_nodes():
    """The nodal geometry's points are the element's own nodes, in node order. For P1
    its gradient is the integration geometry's (constant), and its volumes are right."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    space = FunctionSpace(mesh, QuadraticTriangleElement)
    geometry = space.geometry_at_nodes
    np.testing.assert_allclose(geometry.points, space.node_coords[space.element_nodes])

    p1 = FunctionSpace(mesh)
    grad_phi = p1.geometry_at_nodes.grad_phi
    np.testing.assert_allclose(grad_phi, np.broadcast_to(p1.geometry.grad_phi, grad_phi.shape))
    np.testing.assert_allclose(p1.geometry_at_nodes.volumes, p1.element_volumes)


def test_average_to_nodal_agrees_with_recover_nodal_for_an_element_constant_field():
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    space = FunctionSpace(mesh, QuadraticTriangleElement)
    values = np.random.default_rng(0).normal(size=(len(mesh.elements), 2, 2))
    per_node = np.repeat(values[:, None], QuadraticTriangleElement.N, axis=1)
    np.testing.assert_allclose(average_to_nodal(space, per_node), recover_nodal(space, values))
    with pytest.raises(ValueError):
        average_to_nodal(space, values)


# --- the scalar flux: same recipe for a Poisson gradient ---

def _quadratic_scalar_solution(element_type):
    """A `ScalarFieldSolution` whose field is a known quadratic, so its gradient is
    exactly linear and a P2 space carries it exactly."""
    from fem.post.solution import ScalarFieldSolution
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 4])
    space = FunctionSpace(mesh, element_type)
    x, y = space.node_coords.T
    solution = ScalarFieldSolution.from_solve(space, x**2 + x * y - 0.5 * y**2)

    def exact_gradient(points):
        px, py = np.asarray(points).T
        return np.column_stack([2 * px + py, px - py])

    return solution, exact_gradient


@pytest.mark.parametrize('method', ['average', 'l2'])
def test_p2_nodal_flux_is_exact_for_a_linear_gradient(method):
    solution, exact_gradient = _quadratic_scalar_solution(QuadraticTriangleElement)
    np.testing.assert_allclose(solution.nodal_flux(method=method),
                               exact_gradient(solution.space.node_coords), atol=1e-9)


def test_p2_per_element_flux_is_the_centroid_gradient():
    solution, exact_gradient = _quadratic_scalar_solution(QuadraticTriangleElement)
    mesh = solution.mesh
    centroids = mesh.vertices[mesh.elements].mean(axis=1)
    np.testing.assert_allclose(solution.flux, exact_gradient(centroids), atol=1e-9)
    for e in (0, len(mesh.elements) - 1):
        u_e = solution.u[solution.space.element_nodes[e]]
        np.testing.assert_allclose(solution.space.element_gradient(e, u_e), solution.flux[e])


@pytest.mark.parametrize('method', ['average', 'l2'])
def test_p1_nodal_flux_is_unchanged(method):
    solution, _ = _quadratic_scalar_solution(LinearTriangleElement)
    np.testing.assert_allclose(solution.nodal_flux(method=method),
                               recover_nodal(solution.space, solution.flux, method=method))
