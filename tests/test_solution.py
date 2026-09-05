"""Nodal recovery of derived fields, the typed solutions that carry them, the form
hook that names them, and their round trip through `fem.post.io`.
"""
import numpy as np
import pytest

from fem.boundary import Dirichlet, Neumann
from fem.conditions import Conditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.loads import Source
from fem.mesh.structured import box_mesh
from fem.physics.derived import GradientFlux, StressFlux
from fem.physics.energies import StVenantKirchhoff
from fem.physics.equations import LinearElastic, Poisson, Projection
from fem.physics.forms import DiffusionForm, EnergyForm, LinearElasticForm, MassForm
from fem.physics.materials import LinearElasticMaterial
from fem.post import invariants
from fem.post.recovery import nodal_gradient, recover_nodal
from fem.post.solution import (
    DiffusionSolution,
    ElasticSolution,
    FieldSolution,
    PathSolution,
)
from fem.regions import everywhere, on_plane
from fem.space import FunctionSpace


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_nodal_flux_of_a_linear_field_is_its_exact_constant_gradient(element_type):
    """A linear field has a constant gradient, which every element reproduces exactly and
    the volume-weighted average carries to every node unchanged: recovery adds no error
    to a field the discretization already represents exactly. Holds for P1 and P2."""
    mesh = box_mesh([[0.0, 0.0], [2.0, 1.0]], [5, 4])
    space = FunctionSpace(mesh, element_type, n_components=1)
    gradient = np.array([3.0, -2.0])
    u = space.node_coords @ gradient                 # a linear field, exact in either space

    solution = DiffusionSolution.from_solve(space, u)

    nodal = solution.nodal_gradient()
    assert nodal.shape == (space.n_nodes, 2)
    assert np.allclose(nodal, gradient)


def test_a_poisson_solve_carries_its_flux_and_recovers_it_to_the_nodes():
    """Poisson comes back as a DiffusionSolution: a per-element flux plus its nodal
    recovery, aligned with the solution's own space (here P2, so edge nodes too)."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [7, 7])
    bc = Conditions(
        Dirichlet(everywhere(), 0.0),
    )

    solution = Poisson().problem(mesh, bc + Source(1.0), element_type=QuadraticTriangleElement).solve()

    assert isinstance(solution, DiffusionSolution)
    assert solution.gradient.shape == (len(mesh.elements), 2)
    nodal = solution.nodal_gradient()
    assert nodal.shape == (solution.space.n_nodes, 2)
    assert np.allclose(nodal, nodal_gradient(solution.space, solution.dofs))
    # Read at the nodes, not averaged from the per-element values: on P2 the two differ.
    assert not np.allclose(nodal, recover_nodal(solution.space, solution.gradient))


def test_nodal_flux_takes_a_recovery_method():
    """The `method` argument threads from the solution's nodal accessor to the space's
    recovery, so a caller can ask for the L2 projection instead of the average."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [8, 8])
    bc = Conditions(
        Dirichlet(everywhere(), 0.0),
    )
    solution = Poisson().problem(mesh, bc + Source(1.0)).solve()  # varying flux (curved u)

    average = solution.nodal_gradient(method='average')
    l2 = solution.nodal_gradient(method='l2')
    assert l2.shape == average.shape == (solution.space.n_nodes, 2)
    assert np.allclose(l2, recover_nodal(solution.space, solution.gradient, method='l2'))
    assert not np.allclose(l2, average)


def test_a_projection_stays_a_bare_field_solution():
    """A projection names no derived field, so it is not upgraded to a DiffusionSolution."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    solution = Projection().problem(mesh, Conditions() + Source(2.0)).solve()
    assert type(solution) is FieldSolution


def test_nodal_von_mises_recovers_the_tensor_then_reduces():
    """The convention: recover the stress tensor to the nodes, then form von Mises there.
    Reducing to von Mises per element and averaging that scalar is a different, less
    faithful number, because the reduction is nonlinear."""
    mesh = box_mesh([[0.0, 0.0], [4.0, 1.0]], [12, 4])
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 4.0), [0, -0.3]),
    )
    solution = LinearElastic(200.0, 0.3).problem(mesh, bc).solve()

    assert isinstance(solution, ElasticSolution)
    assert solution.nodal_stress().shape == (solution.space.n_nodes, 3, 3)

    recover_then_reduce = solution.nodal_von_mises()
    assert np.allclose(recover_then_reduce, invariants.von_mises(solution.nodal_stress()))
    reduce_then_recover = recover_nodal(solution.space, solution.von_mises)
    assert not np.allclose(recover_then_reduce, reduce_then_recover)


def test_solution_carries_its_space_and_deformed_mesh_uses_only_vertex_dofs():
    """A P2 solution knows its space (rebuilt from element_type), and its deformed mesh
    warps by the vertex DOFs alone, dropping the edge-node displacements the mesh has no
    vertices for."""
    mesh = box_mesh([[0.0, 0.0], [4.0, 1.0]], [10, 4])
    bc = Conditions(
        Dirichlet(on_plane(0, 0.0), [0, 0]),
        Neumann(on_plane(0, 4.0), [0, -0.2]),
    )
    solution = LinearElastic(200.0, 0.3).problem(mesh, bc, element_type=QuadraticTriangleElement).solve()

    assert solution.element_type is QuadraticTriangleElement
    assert solution.space.n_nodes > len(mesh.vertices)         # edge nodes exist
    deformed = solution.deformed_mesh()
    assert deformed.vertices.shape == mesh.vertices.shape       # one displacement per vertex
    assert solution.nodal_values.shape == (solution.space.n_nodes, 2)
    np.testing.assert_allclose(
        deformed.vertices - mesh.vertices, solution.nodal_values[:len(mesh.vertices)], atol=1e-15)


def test_scalar_solution_nodal_values_are_one_per_node():
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    solution = Poisson().problem(mesh, Conditions(Source(1.0))).solve()
    assert solution.nodal_values.shape == (len(mesh.vertices),)
    np.testing.assert_array_equal(solution.nodal_values, solution.dofs)


def test_forms_name_their_derived_field():
    """The seam: the Laplacian names a gradient, both elastic forms a stress, and the
    mass form (a projection) none."""
    assert isinstance(DiffusionForm().flux(), GradientFlux)
    assert isinstance(LinearElasticForm(LinearElasticMaterial(1.0, 0.3)).flux(), StressFlux)
    assert isinstance(EnergyForm(StVenantKirchhoff(1.0, 0.3)).flux(), StressFlux)
    assert MassForm().flux() is None


def test_derived_field_reads_the_stored_field_and_checks_its_solution():
    """GradientFlux reads a scalar solution's flux as (n_el, 1, d) and refuses a solution
    that carries none, so a misuse fails loudly rather than recovering nonsense."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    space = FunctionSpace(mesh, n_components=1)
    solution = DiffusionSolution.from_solve(space, space.node_coords[:, 0])

    field = GradientFlux().evaluate(solution)
    assert field.shape == (len(mesh.elements), 1, 2)
    assert np.allclose(field[:, 0, :], solution.gradient)

    with pytest.raises(TypeError, match='scalar solution'):
        GradientFlux().evaluate(FieldSolution(space, space.node_coords[:, 0]))


def test_element_type_round_trips_through_save_and_load(tmp_path):
    """A P2 solution reloads as P2, its flux intact, so nodal recovery works after load
    with no live space to lean on."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [5, 5])
    bc = Conditions(
        Dirichlet(everywhere(), 0.0),
    )
    solution = Poisson().problem(mesh, bc + Source(1.0), element_type=QuadraticTriangleElement).solve()

    path = str(tmp_path / 'solution.npz')
    solution.save(path)
    loaded = FieldSolution.load(path)

    assert isinstance(loaded, DiffusionSolution)
    assert loaded.element_type is QuadraticTriangleElement
    assert np.allclose(loaded.nodal_gradient(), solution.nodal_gradient())


# -- PathSolution: an equilibrium path and its stability -------------------------


def _path(stability, lambdas=None):
    """A three-DOF path on a trivial space, so the container is tested and not a solve."""
    mesh = box_mesh([[0.0, 0.0], [1.0, 1.0]], [2, 2])
    space = FunctionSpace(mesh, LinearTriangleElement, n_components=1)
    n = len(stability)
    lambdas = np.arange(n, dtype=float) if lambdas is None else np.asarray(lambdas, float)
    dofs = np.outer(lambdas, np.ones(space.n_dofs))
    return PathSolution(space, lambdas, dofs, np.asarray(stability))


def test_path_lambdas_alias_the_series_parameter():
    """`t` holds the load factor, and path code reads it under its own name."""
    path = _path([1, 1, 1], lambdas=[0.0, 0.4, 0.9])
    np.testing.assert_allclose(path.lambdas, [0.0, 0.4, 0.9])
    assert path.lambdas is path.t
    assert len(path) == 3
    assert isinstance(path[1], FieldSolution)


def test_path_limit_points_are_the_brackets_where_stability_flips():
    """A sign change between consecutive states brackets an odd number of eigenvalue
    crossings: the fold, or a bifurcation, was passed in that step."""
    path = _path([1, 1, -1, -1, 1])
    np.testing.assert_array_equal(path.limit_points, [1, 3])


def test_path_without_stability_flags_reports_no_limit_points():
    """An unknown flag (0) is not a crossing either way, so it brackets nothing."""
    assert _path([0, 0, 0]).limit_points.size == 0


def test_path_needs_one_stability_flag_per_step():
    with pytest.raises(ValueError, match='one flag per step'):
        _path([1, 1, 1, 1], lambdas=[0.0, 1.0, 2.0])


def test_path_round_trips_through_save_and_load(tmp_path):
    """The stability flags are a stored field like any other: `fem.post.io` reflects
    over the dataclass, so the subclass needs no I/O of its own."""
    path = _path([1, -1, -1, 1], lambdas=[0.0, 0.7, 0.6, 0.9])

    file = str(tmp_path / 'path.npz')
    path.save(file)
    loaded = PathSolution.load(file)

    assert isinstance(loaded, PathSolution)
    np.testing.assert_allclose(loaded.lambdas, path.lambdas)
    np.testing.assert_allclose(loaded.dofs, path.dofs)
    np.testing.assert_array_equal(loaded.stability, path.stability)
    np.testing.assert_array_equal(loaded.limit_points, [0, 2])
