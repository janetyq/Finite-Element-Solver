"""Nodal recovery of derived fields, and the DerivedField seam that names them.

The solve stores a per-element derived field (a scalar flux, an elastic stress); these
tests cover recovering it to a continuous per-node field, the typed solutions that carry
it, the equation hook that names it, and that it round-trips through `fem.io`.
"""
import numpy as np
import pytest

from fem import invariants
from fem.boundary import BCType, BoundaryConditions
from fem.elements import LinearTriangleElement, QuadraticTriangleElement
from fem.equations import LinearElastic, Poisson, Projection
from fem.mesh.structured import create_rect_mesh
from fem.postprocess import GradientField, StressField
from fem.regions import everywhere, on_plane
from fem.solution import ElasticSolution, FieldSolution, ScalarFieldSolution
from fem.solver import Solver
from fem.space import FunctionSpace


@pytest.mark.parametrize('element_type', [LinearTriangleElement, QuadraticTriangleElement])
def test_nodal_flux_of_a_linear_field_is_its_exact_constant_gradient(element_type):
    """A linear field has a constant gradient, which every element reproduces exactly and
    the volume-weighted average carries to every node unchanged: recovery adds no error
    to a field the discretization already represents exactly. Holds for P1 and P2."""
    mesh = create_rect_mesh([[0.0, 0.0], [2.0, 1.0]], [5, 4])
    space = FunctionSpace(mesh, element_type, n_components=1)
    gradient = np.array([3.0, -2.0])
    u = space.node_coords @ gradient                 # a linear field, exact in either space

    solution = ScalarFieldSolution.from_solve(space, u)

    nodal = solution.nodal_flux()
    assert nodal.shape == (space.n_nodes, 2)
    assert np.allclose(nodal, gradient)


def test_a_poisson_solve_carries_its_flux_and_recovers_it_to_the_nodes():
    """Poisson comes back as a ScalarFieldSolution: a per-element flux plus its nodal
    recovery, aligned with the solution's own space (here P2, so edge nodes too)."""
    mesh = create_rect_mesh([[0.0, 0.0], [1.0, 1.0]], [7, 7])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)

    solution = Solver(mesh, Poisson(source=1.0), bc,
                      element_type=QuadraticTriangleElement).solve()

    assert isinstance(solution, ScalarFieldSolution)
    assert solution.flux.shape == (len(mesh.elements), 2)
    assert solution.nodal_flux().shape == (solution.space.n_nodes, 2)
    assert np.allclose(solution.nodal_flux(), solution.space.recover_nodal(solution.flux))


def test_nodal_flux_takes_a_recovery_method():
    """The `method` argument threads from the solution's nodal accessor to the space's
    recovery, so a caller can ask for the L2 projection instead of the average."""
    mesh = create_rect_mesh([[0.0, 0.0], [1.0, 1.0]], [8, 8])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solution = Solver(mesh, Poisson(source=1.0), bc).solve()  # varying flux (curved u)

    average = solution.nodal_flux(method='average')
    l2 = solution.nodal_flux(method='l2')
    assert l2.shape == average.shape == (solution.space.n_nodes, 2)
    assert np.allclose(l2, solution.space.recover_nodal(solution.flux, method='l2'))
    assert not np.allclose(l2, average)


def test_a_projection_stays_a_bare_field_solution():
    """A projection names no derived field, so it is not upgraded to a ScalarFieldSolution."""
    mesh = create_rect_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    solution = Solver(mesh, Projection(source=2.0), BoundaryConditions()).solve()
    assert type(solution) is FieldSolution


def test_nodal_von_mises_recovers_the_tensor_then_reduces():
    """The convention: recover the stress tensor to the nodes, then form von Mises there.
    Reducing to von Mises per element and averaging that scalar is a different, less
    faithful number, because the reduction is nonlinear."""
    mesh = create_rect_mesh([[0.0, 0.0], [4.0, 1.0]], [12, 4])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 4.0), [0, -0.3])
    solution = Solver(mesh, LinearElastic(200.0, 0.3), bc).solve()

    assert isinstance(solution, ElasticSolution)
    assert solution.nodal_stress().shape == (solution.space.n_nodes, 3, 3)

    recover_then_reduce = solution.nodal_von_mises()
    assert np.allclose(recover_then_reduce, invariants.von_mises(solution.nodal_stress()))
    reduce_then_recover = solution.space.recover_nodal(solution.von_mises)
    assert not np.allclose(recover_then_reduce, reduce_then_recover)


def test_solution_carries_its_space_and_deformed_mesh_uses_only_vertex_dofs():
    """A P2 solution knows its space (rebuilt from element_type), and its deformed mesh
    warps by the vertex DOFs alone, dropping the edge-node displacements the mesh has no
    vertices for."""
    mesh = create_rect_mesh([[0.0, 0.0], [4.0, 1.0]], [10, 4])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, on_plane(0, 0.0), [0, 0])
    bc.add(BCType.NEUMANN, on_plane(0, 4.0), [0, -0.2])
    solution = Solver(mesh, LinearElastic(200.0, 0.3), bc,
                      element_type=QuadraticTriangleElement).solve()

    assert solution.element_type is QuadraticTriangleElement
    assert solution.space.n_nodes > len(mesh.vertices)         # edge nodes exist
    deformed = solution.deformed_mesh()
    assert deformed.vertices.shape == mesh.vertices.shape       # one displacement per vertex


def test_equations_name_their_derived_field():
    """The seam: Poisson names a gradient, elasticity a stress, a projection none."""
    assert isinstance(Poisson().derived_field(), GradientField)
    assert isinstance(LinearElastic(1.0, 0.3).derived_field(), StressField)
    assert Projection().derived_field() is None


def test_derived_field_reads_the_stored_field_and_checks_its_solution():
    """GradientField reads a scalar solution's flux as (n_el, 1, d) and refuses a solution
    that carries none, so a misuse fails loudly rather than recovering nonsense."""
    mesh = create_rect_mesh([[0.0, 0.0], [1.0, 1.0]], [4, 4])
    space = FunctionSpace(mesh, n_components=1)
    solution = ScalarFieldSolution.from_solve(space, space.node_coords[:, 0])

    field = GradientField().evaluate(solution)
    assert field.shape == (len(mesh.elements), 1, 2)
    assert np.allclose(field[:, 0, :], solution.flux)

    with pytest.raises(TypeError, match='scalar solution'):
        GradientField().evaluate(FieldSolution(mesh, 1, space.node_coords[:, 0]))


def test_element_type_round_trips_through_save_and_load(tmp_path):
    """A P2 solution reloads as P2, its flux intact, so nodal recovery works after load
    with no live space to lean on."""
    mesh = create_rect_mesh([[0.0, 0.0], [1.0, 1.0]], [5, 5])
    bc = BoundaryConditions()
    bc.add(BCType.DIRICHLET, everywhere(), 0.0)
    solution = Solver(mesh, Poisson(source=1.0), bc,
                      element_type=QuadraticTriangleElement).solve()

    path = str(tmp_path / 'solution.npz')
    solution.save(path)
    loaded = FieldSolution.load(path)

    assert isinstance(loaded, ScalarFieldSolution)
    assert loaded.element_type is QuadraticTriangleElement
    assert np.allclose(loaded.nodal_flux(), solution.nodal_flux())
