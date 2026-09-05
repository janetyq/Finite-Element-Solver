"""Nodal recovery (`fem.post.recovery`): the volume-weighted average and the L2
projection of a per-element field onto the nodes."""
import numpy as np
import pytest
from helpers import two_triangle_square

from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.field import NodalField
from fem.mesh.mesh import Mesh
from fem.mesh.structured import box_mesh
from fem.post.recovery import recover_nodal
from fem.space import FunctionSpace


def test_shared_vertex_combines_the_values_of_its_elements():
    space = FunctionSpace(two_triangle_square())
    values = recover_nodal(space, np.array([10.0, 20.0]))

    # Both triangles have the same area, so the weighting reduces to the mean.
    assert values[0] == 15.0
    assert values[2] == 15.0
    assert values[1] == 10.0
    assert values[3] == 20.0


def test_projection_does_not_depend_on_element_ordering():
    """A shared vertex accumulates every element's contribution, so reversing the elements
    leaves the result unchanged."""
    mesh = two_triangle_square()
    reversed_mesh = Mesh(
        vertices=mesh.vertices, elements=mesh.elements[::-1], boundary=mesh.boundary,
    )
    forward = recover_nodal(FunctionSpace(mesh), np.array([10.0, 20.0]))
    backward = recover_nodal(FunctionSpace(reversed_mesh), np.array([20.0, 10.0]))
    assert np.allclose(forward, backward)


def test_constant_element_field_is_reproduced_at_every_vertex(make_unit_square):
    """The patch test: a constant per-element field must come back as the same
    constant at every vertex, whatever the valence and whatever the weighting."""
    space = FunctionSpace(make_unit_square(6))
    constant = np.full(len(space.mesh.elements), 3.5)
    assert np.allclose(recover_nodal(space, constant), 3.5)


def test_projection_weights_by_element_volume():
    """The projection weights by element measure: on a graded mesh (vertex 0 shared by a
    triangle of area 0.5 and one of area 0.05) an unweighted mean gives a different
    answer."""
    mesh = Mesh(
        vertices=[[0, 0], [1, 0], [0, 1], [-0.1, 0]],
        elements=[[0, 1, 2], [0, 2, 3]],
        boundary=[[0, 1], [1, 2], [2, 3], [3, 0]],
    )
    space = FunctionSpace(mesh)
    areas = space.element_volumes
    np.testing.assert_allclose(areas, [0.5, 0.05])

    values = recover_nodal(space, np.array([10.0, 20.0]))
    expected = (10.0 * areas[0] + 20.0 * areas[1]) / areas.sum()

    np.testing.assert_allclose(values[0], expected)
    assert not np.isclose(values[0], 15.0)  # what an unweighted mean would give


def test_projection_rejects_a_field_of_the_wrong_length(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='one value per element'):
        recover_nodal(space, np.zeros(3))


def test_unknown_recovery_method_is_rejected(make_unit_square):
    space = FunctionSpace(make_unit_square(4))
    with pytest.raises(ValueError, match='unknown recovery method'):
        recover_nodal(space, np.zeros(len(space.mesh.elements)), method='patch')


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_l2_recovery_reproduces_a_constant_field(element_type):
    """The patch test for the L2 projection: a constant per-element field projects to
    that same constant at every node, since the constant lies in the nodal space."""
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [6, 5])
    space = FunctionSpace(mesh, element_type, n_components=1)
    constant = np.full(len(mesh.elements), 3.5)
    assert np.allclose(recover_nodal(space, constant, method='l2'), 3.5)


def test_l2_recovery_conserves_the_field_integral():
    """The L2 projection satisfies ∫ q = ∫ f (test against the constant 1, which the
    nodal space represents), so the recovered field carries the per-element field's
    integral to machine precision on any mesh."""
    uniform = box_mesh([[0.0, 0.0], [1.0, 1.0]], [10, 10])
    grading = np.column_stack([uniform.vertices[:, 0] ** 2 - uniform.vertices[:, 0],
                               np.zeros(uniform.n_vertices)])
    mesh = uniform.displaced(grading)                        # x -> x^2, a graded mesh
    space = FunctionSpace(mesh)
    field = mesh.vertices[mesh.elements].mean(axis=1)[:, 0]  # varies element to element
    exact = float((field * space.element_volumes).sum())

    recovered = recover_nodal(space, field, method='l2')
    assert NodalField(space, recovered).integrate() == pytest.approx(exact, rel=1e-12)


def test_l2_and_average_recovery_differ_on_a_varying_field():
    """The two recoveries are different operators: the local weighted average
    and the global mass projection agree only on a field the space reproduces exactly
    (a constant), and differ on one that varies element to element."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [8, 8])
    space = FunctionSpace(mesh)
    field = mesh.vertices[mesh.elements].mean(axis=1)[:, 0] ** 2

    average = recover_nodal(space, field, method='average')
    l2 = recover_nodal(space, field, method='l2')
    assert not np.allclose(average, l2)


# -- the mass matrix is factored once per space -------------------------------------


def _counting_prepare(monkeypatch):
    """Count `DirectBackend.prepare` calls, the one factorization the recoveries make."""
    from fem.algebra.backends import DirectBackend
    calls = []
    original = DirectBackend.prepare

    def prepare(self, A):
        calls.append(A.shape)
        return original(self, A)

    monkeypatch.setattr(DirectBackend, 'prepare', prepare)
    return calls


def test_l2_recoveries_share_one_factorization_per_space(monkeypatch):
    """Every L2 projection on a space, whatever field it recovers, solves against the
    space's cached `nodal_mass_solver`: a second recovery factors nothing."""
    from fem.analysis.estimators import RecoveryEstimator
    from fem.boundary import Dirichlet
    from fem.conditions import Conditions
    from fem.loads import Source
    from fem.physics.equations import Poisson
    from fem.regions import everywhere
    problem = Poisson().problem(box_mesh([[0.0, 0.0], [1.0, 1.0]], [8, 8]),
                                Conditions(Dirichlet(everywhere(), 0.0), Source(1.0)))
    solution = problem.solve()
    calls = _counting_prepare(monkeypatch)
    solution.nodal_gradient('l2')
    assert len(calls) == 1, 'the first projection factors the mass matrix'
    solution.nodal_gradient('l2')
    RecoveryEstimator().estimate(problem, solution)
    recover_nodal(problem.space, solution.gradient, method='l2')
    assert len(calls) == 1, 'later projections on the space reuse it'


def test_a_supplied_backend_projects_the_same_field(make_unit_square):
    """A `backend` given to a recovery is prepared for that call and gives the same
    projection as the cached direct factorization; it is not cached on the space."""
    from fem.algebra.backends import IterativeBackend
    space = FunctionSpace(make_unit_square(7))
    values = np.sin(3.0 * space.mesh.centroids[:, 0]) * space.mesh.centroids[:, 1]
    cached = recover_nodal(space, values, method='l2')
    iterative = recover_nodal(space, values, method='l2', backend=IterativeBackend(rtol=1e-12))
    np.testing.assert_allclose(iterative, cached, rtol=1e-8, atol=1e-10)
