"""Round-trip tests for fem.io persistence.

Meshes go to JSON, solutions to a pickle-free npz archive. The point of these
tests is that a save/load cycle preserves everything a caller depends on --
value arrays, mesh geometry, mesh class and n_components -- and that the load path never
falls back to pickle.
"""
import numpy as np
import pytest

from fem.elements import LinearTriangleElement
from fem.io import load_mesh, save_mesh, save_solution
from fem.mesh.femesh import FEMesh
from fem.mesh.mesh import Mesh
from fem.numerics import bump_function
from fem.solution import Solution
from fem.solver import Heat, Solver


def test_mesh_json_round_trip(make_unit_square, tmp_path):
    """Geometry survives a JSON save/load unchanged."""
    femesh = make_unit_square(6)
    path = tmp_path / "mesh.json"

    save_mesh(femesh, path)
    loaded = load_mesh(path)

    assert np.allclose(loaded.vertices, femesh.vertices)
    assert np.array_equal(loaded.elements, femesh.elements)
    assert np.array_equal(loaded.boundary, femesh.boundary)


def test_mesh_load_rebuilds_the_requested_class(make_unit_square, tmp_path):
    """`cls` controls the reconstructed type, so FEMesh.load returns an FEMesh."""
    femesh = make_unit_square(6)
    path = tmp_path / "mesh.json"
    femesh.save(path)

    assert type(load_mesh(path)) is Mesh
    assert type(FEMesh.load(path)) is FEMesh


def test_solution_round_trip_preserves_values_mesh_and_dim(make_unit_square, tmp_path):
    """A hand-built solution comes back with identical values, geometry and component count."""
    femesh = make_unit_square(6)
    solution = Solution(femesh, n_components=2)
    solution.set_values("u", np.arange(len(femesh.vertices) * 2, dtype=float))
    solution.set_values("compliance", np.linspace(0, 1, len(femesh.elements)))
    path = tmp_path / "solution.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert loaded.n_components == 2
    assert set(loaded.values) == {"u", "compliance"}
    assert np.allclose(loaded.get_values("u"), solution.get_values("u"))
    assert np.allclose(loaded.get_values("compliance"), solution.get_values("compliance"))
    assert type(loaded.femesh) is FEMesh
    assert loaded.femesh.element_type is LinearTriangleElement
    assert np.allclose(loaded.femesh.vertices, femesh.vertices)
    assert np.array_equal(loaded.femesh.elements, femesh.elements)


def test_solution_round_trip_after_solve(make_unit_square, tmp_path):
    """Per-timestep values (a list of arrays) stack and reload intact."""
    femesh = make_unit_square(8)
    u0 = bump_function(femesh.vertices, femesh.vertices.max(axis=0), mag=50, size=0.3) + 300
    solution = Solver(femesh, Heat(u_initial=u0.copy(), iters=3, dt=0.01)).solve()
    path = tmp_path / "heat.npz"

    solution.save(path)
    loaded = Solution.load(path)

    assert np.allclose(loaded.get_values("t_values"), solution.get_values("t_values"))
    assert np.allclose(loaded.get_values("u_values"), solution.get_values("u_values"))
    # The deformed-mesh path needs a working femesh, not just the raw arrays.
    assert len(loaded.femesh.element_objs) == len(femesh.element_objs)


def test_solution_load_does_not_unpickle(make_unit_square, tmp_path):
    """The archive must be readable with allow_pickle=False -- that is the whole
    point of moving off pickle, so pin it rather than trusting the default."""
    femesh = make_unit_square(6)
    solution = Solution(femesh, n_components=1)
    solution.set_values("u", np.zeros(len(femesh.vertices)))
    path = tmp_path / "solution.npz"
    solution.save(path)

    with np.load(path, allow_pickle=False) as data:
        assert "value.u" in data.files


def test_saving_a_ragged_value_fails_loudly(make_unit_square, tmp_path):
    """Ragged values can't be stored without object arrays (which would mean
    pickle), so they must raise at save time rather than silently degrade."""
    solution = Solution(make_unit_square(6), n_components=1)
    solution.set_values("ragged", [np.zeros(3), np.zeros(5)])

    with pytest.raises(ValueError):
        save_solution(solution, tmp_path / "ragged.npz")
