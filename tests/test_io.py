"""Round-trip tests for fem.io persistence.

Meshes go to JSON, solutions to a pickle-free npz archive. The point of these
tests is that a save/load cycle preserves everything a caller depends on --
value arrays, mesh geometry, mesh class and n_components -- and that the load path never
falls back to pickle.
"""
import numpy as np
import pytest

from fem.integrators import ThetaMethod
from fem.io import load_mesh, save_mesh, save_solution
from fem.mesh.mesh import Mesh
from fem.numerics import bump_function
from fem.problem import heat
from fem.solution import ElasticSolution, FieldSolution, Solution, TransientSolution


def test_mesh_json_round_trip(make_unit_square, tmp_path):
    """Geometry survives a JSON save/load unchanged."""
    mesh = make_unit_square(6)
    path = tmp_path / "mesh.json"

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert np.allclose(loaded.vertices, mesh.vertices)
    assert np.array_equal(loaded.elements, mesh.elements)
    assert np.array_equal(loaded.boundary, mesh.boundary)


def test_mesh_load_rebuilds_the_requested_class(make_unit_square, tmp_path):
    """`cls` controls the reconstructed type."""
    mesh = make_unit_square(6)
    path = tmp_path / "mesh.json"
    mesh.save(path)

    assert type(load_mesh(path)) is Mesh
    assert type(Mesh.load(path)) is Mesh


def test_elastic_solution_round_trip_preserves_fields_mesh_and_dim(make_unit_square, tmp_path):
    """An ElasticSolution comes back the same class, with identical fields."""
    mesh = make_unit_square(6)
    n_el = len(mesh.elements)
    solution = ElasticSolution(
        mesh, 2,
        np.arange(len(mesh.vertices) * 2, dtype=float),
        np.linspace(0, 1, n_el),
        np.linspace(1, 2, n_el),
        np.linspace(2, 3, n_el),
    )
    path = tmp_path / "solution.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert type(loaded) is ElasticSolution
    assert loaded.n_components == 2
    assert np.allclose(loaded.u, solution.u)
    assert np.allclose(loaded.compliance, solution.compliance)
    assert np.allclose(loaded.stress, solution.stress)
    assert type(loaded.mesh) is Mesh
    assert np.allclose(loaded.mesh.vertices, mesh.vertices)
    assert np.array_equal(loaded.mesh.elements, mesh.elements)


def test_transient_solution_round_trip_after_solve(make_unit_square, tmp_path):
    """A time series (times + a list of per-step fields) reloads as a TransientSolution."""
    mesh = make_unit_square(8)
    u0 = bump_function(mesh.vertices, mesh.vertices.max(axis=0), mag=50, size=0.3) + 300
    solution = ThetaMethod(dt=0.01, steps=3).run(heat(mesh), u0.copy())
    path = tmp_path / "heat.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert type(loaded) is TransientSolution
    assert np.allclose(loaded.t, solution.t)
    assert np.allclose(loaded.u, solution.u)
    # Geometry round-trips; a solve rebuilds element data into its own space.
    assert np.allclose(loaded.mesh.vertices, mesh.vertices)
    assert np.array_equal(loaded.mesh.elements, mesh.elements)


def test_solution_load_does_not_unpickle(make_unit_square, tmp_path):
    """The archive must be readable with allow_pickle=False -- that is the whole
    point of moving off pickle, so pin it rather than trusting the default."""
    mesh = make_unit_square(6)
    solution = FieldSolution(mesh, 1, np.zeros(len(mesh.vertices)))
    path = tmp_path / "solution.npz"
    solution.save(path)

    with np.load(path, allow_pickle=False) as data:
        assert "value.u" in data.files


def test_saving_a_ragged_field_fails_loudly(make_unit_square, tmp_path):
    """A ragged field can only be stored as an object array (i.e. pickle), so it
    must raise at save time rather than silently degrade. Typed solves never
    produce one; a hand-built series with unequal-length steps does."""
    mesh = make_unit_square(6)
    ragged = TransientSolution(mesh, 1, np.array([0.0, 1.0]), [np.zeros(3), np.zeros(5)])

    with pytest.raises(ValueError):
        save_solution(ragged, tmp_path / "ragged.npz")
